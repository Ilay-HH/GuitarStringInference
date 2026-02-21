"""
Single-window annotator: N/P = next/prev suspect, Space = play, Arrows = frame step.
On suspect: SYNC top left, V = good, 1-6 = wrong string. On any frame: 1-6 = manual tag.
"""
import cv2
import numpy as np
import json
import subprocess
import tempfile
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import (
    detectStringLinesAngled,
    detectStringLinesInHandsRegion,
    detectSuspects,
    findVisibleXRange,
    shrinkToCenter,
    fallbackStringLines,
)
from scripts.hands_region.handsCalibrator import getHandsBbox
from scripts.hands_region.handsRegionDetector import HandsRegionDetector, getProcessingRoi


def stopAudio():
    try:
        import pygame
        pygame.mixer.music.stop()
    except Exception:
        pass


def getFfmpegPath():
    """Get ffmpeg path: imageio-ffmpeg (bundled) or system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    return "ffmpeg"


def initAudio(videoPath):
    """Extract audio via ffmpeg, init pygame mixer. Returns True if audio ready."""
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    except Exception:
        return False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    ffmpegExe = getFfmpegPath()
    try:
        subprocess.run(
            [ffmpegExe, "-y", "-i", str(videoPath), "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", tmp.name],
            capture_output=True, timeout=60, check=True
        )
        import pygame
        pygame.mixer.music.load(tmp.name)
        return True
    except FileNotFoundError:
        print("ffmpeg not found - run: pip install imageio-ffmpeg")
        return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Audio extraction failed: {e}")
        return False


def colorEdgesByString(edges, stringLines, colors):
    """Assign edges to strings using band boundaries (non-overlapping, respects angle)."""
    h, w = edges.shape
    ey, ex = np.where(edges > 0)
    if len(ey) == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    n = len(stringLines)
    yCoords = np.zeros((len(ey), n))
    for i, (x1, y1, x2, y2) in enumerate(stringLines):
        denom = x2 - x1
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        t = np.clip((ex - x1) / denom, 0, 1)
        yCoords[:, i] = y1 + t * (y2 - y1)
    mids = (yCoords[:, :-1] + yCoords[:, 1:]) / 2
    bounds = np.column_stack([np.zeros(len(ey)), mids, np.full(len(ey), h)])
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(n):
        mask = (bounds[:, i] <= ey) & (ey < bounds[:, i + 1])
        out[ey[mask], ex[mask]] = colors[i]
    return out


def drawLegend(overlay, colors, numStrings=6):
    """Draw color-to-string legend in bottom right."""
    h, w = overlay.shape[:2]
    pad = 10
    boxH = numStrings * 22 + pad * 2
    boxW = 100
    x1, y1 = w - boxW - pad, h - boxH - pad
    cv2.rectangle(overlay, (x1, y1), (x1 + boxW, y1 + boxH), (40, 40, 40), -1)
    cv2.rectangle(overlay, (x1, y1), (x1 + boxW, y1 + boxH), (200, 200, 200), 1)
    for i in range(numStrings):
        y = y1 + pad + i * 22 + 14
        cv2.circle(overlay, (x1 + 18, y - 4), 6, colors[i], -1)
        cv2.putText(overlay, f"Str {i + 1}", (x1 + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def buildOverlay(frame, gray, edges, stringLines, colors, roiY1=None, roiY2=None, handsBbox=None, videoPath=None, frameIdx=None, algoDetector=None):
    """Overlay colored edges only within used region. algoDetector or calibration or fallback."""
    h, w = frame.shape[:2]
    if roiY1 is None or roiY2 is None:
        from scripts.hands_region.handsRegionDetector import getRoiVerticalBounds
        ry1, ry2 = getRoiVerticalBounds(h)
        if roiY1 is None:
            roiY1 = ry1
        if roiY2 is None:
            roiY2 = ry2
    if handsBbox is None and algoDetector is not None:
        handsBbox = algoDetector.detect(frame, gray, roiY1, roiY2, h, w, stringLines)
    if handsBbox is None and videoPath is not None and frameIdx is not None:
        handsBbox = getHandsBbox(videoPath, frameIdx, h, w)
    if handsBbox is not None:
        x1, y1, x2, y2 = handsBbox
    else:
        rawRange = findVisibleXRange(gray, stringLines)
        x1, x2 = shrinkToCenter(rawRange, w)
        y1, y2 = roiY1, roiY2
    coloredEdges = colorEdgesByString(edges, stringLines, colors)
    overlay = cv2.convertScaleAbs(frame, alpha=0.5, beta=0)
    roiFrame = frame[y1:y2, x1:x2]
    roiColored = coloredEdges[y1:y2, x1:x2]
    overlay[y1:y2, x1:x2] = cv2.addWeighted(roiFrame, 0.7, roiColored, 0.5, 0)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(overlay, "used region", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    overlay = drawLegend(overlay, colors, len(stringLines))
    return overlay


def runAnnotator(videoPath, numStrings=6, outputPath=None, dropThreshold=0.18, useAlgorithm=False):
    """Single-window annotator with sound, N/P suspect nav, Space play, Arrows step."""
    videoPath = Path(videoPath)
    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videoPath}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    h, w = first.shape[:2]
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    result = detectStringLinesInHandsRegion(first, gray, numStrings, returnCrop=True)
    stringLines, handsX1, handsY1, handsX2, handsY2 = result[0], result[1], result[2], result[3], result[4]
    if stringLines is None:
        roiEdges = result[6]
        stringLines = detectStringLinesAngled(roiEdges, numStrings, 0, roiEdges.shape[0], yOffset=handsY1)
        if stringLines is not None:
            stringLines = [(l[0] + handsX1, l[1], l[2] + handsX1, l[3]) for l in stringLines]
    if stringLines is None:
        stringLines = fallbackStringLines(h, w, numStrings, handsY1, handsY2)
        handsX1, handsY1, handsX2, handsY2 = 0, int(h * 0.2), w, int(h * 0.8)
    else:
        bbox = getProcessingRoi(first, gray, h, w, stringLines)
        handsX1, handsY1, handsX2, handsY2 = bbox
    print("Scanning video for suspect frames...")
    algoDetector = HandsRegionDetector() if useAlgorithm else None
    suspects = detectSuspects(videoPath, stringLines, numStrings, dropThreshold=dropThreshold,
                             handsBboxGetter=None if useAlgorithm else getHandsBbox, useAlgorithm=useAlgorithm)
    suspectsByFrame = {f: s for f, s in suspects}
    print(f"Found {len(suspects)} suspect frames")
    hasAudio = False
    try:
        hasAudio = initAudio(videoPath)
        if hasAudio:
            print("Audio loaded")
        else:
            print("No audio (install ffmpeg for sound)")
    except Exception:
        pass
    colors = [
        (100, 100, 255), (50, 150, 255), (100, 255, 100),
        (255, 200, 50), (255, 100, 200), (100, 200, 255)
    ][:numStrings]
    annotations = {}
    currentFrame = 0 if not suspects else suspects[0][0]
    suspectIdx = 0 if suspects else -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    scale = min(1.0, 1100 / w)
    playing = False
    playStartFrame = 0
    playStartTime = 0.0
    print("N/P = next/prev suspect | Space = play | Arrows = frame step | V = good | 1-6 = annotate | Q = quit")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
        ret, frame = cap.read()
        if not ret:
            currentFrame = max(0, min(currentFrame, totalFrames - 1))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roiCropped = gray[handsY1:handsY2, handsX1:handsX2]
        roiEdges = cv2.Canny(roiCropped, 50, 150)
        edges = np.zeros_like(gray)
        edges[handsY1:handsY2, handsX1:handsX2] = roiEdges
        overlay = buildOverlay(frame, gray, edges, stringLines, colors, handsY1, handsY2,
                              videoPath=videoPath, frameIdx=currentFrame, algoDetector=algoDetector)
        isSuspect = currentFrame in suspectsByFrame
        suspectedStrings = suspectsByFrame.get(currentFrame, [])
        if isSuspect:
            cv2.putText(overlay, "SYNC", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(overlay, f"Suspect: Str {suspectedStrings}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        ann = annotations.get(currentFrame, {})
        if ann:
            fb = ann.get("feedback", "")
            actual = ann.get("actualStrings", [])
            if fb == "correct":
                cv2.putText(overlay, "Your: Correct", (10, 95 if isSuspect else 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif fb == "wrong":
                cv2.putText(overlay, f"Your: Wrong, actual Str {actual}", (10, 95 if isSuspect else 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            elif ann.get("manual"):
                cv2.putText(overlay, f"Manual: Str {actual}", (10, 95 if isSuspect else 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        status = "PLAYING" if playing else "Frame " + str(currentFrame)
        cv2.putText(overlay, f"{status} | Time {currentFrame/fps:.2f}s | N/P suspects Arrows step Space play",
                    (10, overlay.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        disp = cv2.resize(overlay, None, fx=scale, fy=scale) if scale < 1.0 else overlay
        cv2.imshow("Annotator", disp)
        pollMs = 35 if playing else 50
        key = cv2.waitKeyEx(pollMs)
        if playing:
            if hasAudio:
                try:
                    import pygame
                    posMs = pygame.mixer.music.get_pos()
                    currentFrame = min(int(playStartFrame + (posMs / 1000.0) * fps), totalFrames - 1)
                    if not pygame.mixer.music.get_busy():
                        playing = False
                except Exception:
                    playing = False
            else:
                currentFrame = min(int(playStartFrame + (time.time() - playStartTime) * fps), totalFrames - 1)
                if currentFrame >= totalFrames - 1:
                    playing = False
        if key in (ord("q"), 27):
            if hasAudio:
                stopAudio()
            break
        elif key == ord(" "):
            if playing:
                playing = False
                stopAudio()
            else:
                playStartFrame = currentFrame
                playStartTime = time.time()
                if hasAudio:
                    try:
                        import pygame
                        pygame.mixer.music.play(start=currentFrame / fps)
                        playing = True
                    except Exception:
                        playing = False
                else:
                    playing = True
        elif key == 2555904:
            if playing:
                playing = False
                stopAudio()
            currentFrame = min(currentFrame + 1, totalFrames - 1)
        elif key == 2424832:
            if playing:
                playing = False
                stopAudio()
            currentFrame = max(0, currentFrame - 1)
        elif key == ord("n"):
            if playing:
                playing = False
                stopAudio()
            if suspects and suspectIdx < len(suspects) - 1:
                suspectIdx += 1
                currentFrame = suspects[suspectIdx][0]
        elif key == ord("p"):
            if playing:
                playing = False
                stopAudio()
            if suspects and suspectIdx > 0:
                suspectIdx -= 1
                currentFrame = suspects[suspectIdx][0]
        elif key == ord("v"):
            if playing:
                playing = False
                stopAudio()
            if isSuspect:
                annotations[currentFrame] = {"suspects": suspectedStrings, "feedback": "correct"}
        elif ord("1") <= key <= ord("6"):
            if playing:
                playing = False
                stopAudio()
            actual = key - ord("0")
            if isSuspect:
                annotations[currentFrame] = {"suspects": suspectedStrings, "feedback": "wrong", "actualStrings": [actual]}
            else:
                annotations[currentFrame] = {"manual": True, "actualStrings": [actual]}
        if suspects and currentFrame in suspectsByFrame:
            for i, (f, _) in enumerate(suspects):
                if f == currentFrame:
                    suspectIdx = i
                    break
    cap.release()
    cv2.destroyAllWindows()
    if outputPath:
        out = Path(outputPath)
        out.parent.mkdir(parents=True, exist_ok=True)
        def toJson(obj):
            if isinstance(obj, dict):
                return {str(k): toJson(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [toJson(x) for x in obj]
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            return obj
        manualFrames = [int(k) for k, v in annotations.items() if v.get("manual")]
        data = {
            "video": str(videoPath),
            "stringLines": [[int(x) for x in L] for L in stringLines],
            "fps": float(fps),
            "dropThreshold": float(dropThreshold),
            "suspects": [(int(f), [int(x) for x in s]) for f, s in suspects],
            "manualTags": manualFrames,
            "annotations": toJson(annotations),
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(annotations)} annotations to {outputPath}")
    return annotations, stringLines


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Single-window annotator: N/P suspects, Space play, V=good, 1-6=annotate"
    )
    parser.add_argument("-v", "--video", type=str, default=None, help="Video path (default: shortest in data)")
    parser.add_argument("-n", "--strings", type=int, default=6)
    parser.add_argument("-o", "--output", type=str, default="output/annotations.json")
    parser.add_argument("-t", "--threshold", type=float, default=0.18,
                        help="Drop threshold for suspect detection (default 0.18 = 18%% below baseline)")
    parser.add_argument("-a", "--algorithm", action="store_true", help="Use algorithm for hands region (no calibration)")
    args = parser.parse_args()
    dataDir = PROJECT_ROOT / "data"
    if args.video:
        videoPath = Path(args.video)
        if not videoPath.is_absolute():
            videoPath = dataDir / videoPath.name
    else:
        videos = sorted(dataDir.glob("*.mp4"), key=lambda p: p.stat().st_size)
        if not videos:
            raise FileNotFoundError("No videos in data folder")
        videoPath = videos[0]
        print(f"Using shortest video: {videoPath.name}")
    runAnnotator(videoPath, numStrings=args.strings, outputPath=args.output, dropThreshold=args.threshold, useAlgorithm=args.algorithm)


if __name__ == "__main__":
    main()
