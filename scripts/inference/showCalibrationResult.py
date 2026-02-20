"""
Shows calibration result at sample frames. Run after handsCalibrator --sequence.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import detectStringLinesAngled, fallbackStringLines
from scripts.inference.frameAnnotator import colorEdgesByString
from scripts.hands_region.handsCalibrator import getHandsBbox


def main():
    dataDir = PROJECT_ROOT / "data"
    videoPath = dataDir / "The most beautiful melody line.mp4"
    if not videoPath.exists():
        videos = list(dataDir.glob("*.mp4"))
        videoPath = videos[0] if videos else None
    if not videoPath or not videoPath.exists():
        print("No video found in data/")
        return
    cap = cv2.VideoCapture(str(videoPath))
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first = cap.read()
    if not ret:
        print("Cannot read video")
        return
    h, w = first.shape[:2]
    roiY1, roiY2 = int(h * 0.2), int(h * 0.8)
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    roiEdges = cv2.Canny(gray[roiY1:roiY2, :], 50, 150)
    stringLines = detectStringLinesAngled(roiEdges, 6, 0, roiEdges.shape[0], yOffset=roiY1)
    if stringLines is None:
        stringLines = fallbackStringLines(h, w, 6, roiY1, roiY2)
    colors = [
        (100, 100, 255), (50, 150, 255), (100, 255, 100),
        (255, 200, 50), (255, 100, 200), (100, 200, 255)
    ]
    fullEdges = np.zeros_like(gray)
    fullEdges[roiY1:roiY2, :] = roiEdges
    coloredEdges = colorEdgesByString(fullEdges, stringLines, colors)

    outDir = PROJECT_ROOT / "output" / "calibration_preview"
    outDir.mkdir(parents=True, exist_ok=True)

    sampleFrames = [0, totalFrames // 4, totalFrames // 2, 3 * totalFrames // 4, totalFrames - 1]
    for frameIdx in sampleFrames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roiEdges = cv2.Canny(gray[roiY1:roiY2, :], 50, 150)
        fullEdges = np.zeros_like(gray)
        fullEdges[roiY1:roiY2, :] = roiEdges
        coloredEdges = colorEdgesByString(fullEdges, stringLines, colors)

        bbox = getHandsBbox(str(videoPath), frameIdx, h, w)
        if bbox is None:
            print(f"Frame {frameIdx}: no calibration")
            continue
        xLo, yLo, xHi, yHi = bbox

        overlay = cv2.convertScaleAbs(frame, alpha=0.5, beta=0)
        overlay[yLo:yHi, xLo:xHi] = cv2.addWeighted(
            frame[yLo:yHi, xLo:xHi], 0.7,
            coloredEdges[yLo:yHi, xLo:xHi], 0.5, 0
        )
        cv2.rectangle(overlay, (xLo, yLo), (xHi, yHi), (0, 255, 255), 2)
        cv2.putText(overlay, f"Frame {frameIdx} (interpolated)", (xLo, yLo - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        path = outDir / f"calib_frame_{frameIdx:04d}.png"
        cv2.imwrite(str(path), overlay)
        print(f"Saved {path}")

    cap.release()
    print(f"Preview saved to {outDir}")


if __name__ == "__main__":
    main()
