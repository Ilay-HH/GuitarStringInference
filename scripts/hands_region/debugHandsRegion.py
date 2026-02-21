"""
Debug skin and string detection for hands region. Outputs visualizations to understand what each signal returns.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
PROJECT_ROOT = _PROJECT_ROOT if (_PROJECT_ROOT / "config").exists() else _THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.string_tracking.stringEdgeTracker import detectStringLinesAngled, findVisibleXRange, fallbackStringLines
except ImportError:
    try:
        from ._string_utils import detectStringLinesAngled, findVisibleXRange, fallbackStringLines
    except ImportError:
        from _string_utils import detectStringLinesAngled, findVisibleXRange, fallbackStringLines
from scripts.hands_region.handsRegionDetector import getSkinMask, findLowSkinXRange, detectHandsRegion, getRoiVerticalBounds


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
    h, w = 0, 0
    roiY1, roiY2 = 0, 0
    stringLines = None

    outDir = PROJECT_ROOT / "output" / "debug_hands"
    outDir.mkdir(parents=True, exist_ok=True)

    sampleFrames = [0, 30, 60, 90, 180, 360]
    for frameIdx in sampleFrames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        roiY1, roiY2 = getRoiVerticalBounds(h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if stringLines is None:
            roiEdges = cv2.Canny(gray[roiY1:roiY2, :], 50, 150)
            stringLines = detectStringLinesAngled(roiEdges, 6, 0, roiEdges.shape[0], yOffset=roiY1)
            if stringLines is None:
                stringLines = fallbackStringLines(h, w, 6, roiY1, roiY2)

        skinMask = getSkinMask(frame, roiY1, roiY2)
        skinRoi = skinMask[roiY1:roiY2, :]
        xRangeSkin = findLowSkinXRange(skinMask, roiY1, roiY2)

        for skinFracThresh in [0.15, 0.25, 0.35]:
            testRange = findLowSkinXRange(skinMask, roiY1, roiY2, skinFracThresh=skinFracThresh)
            if testRange and frameIdx == 0:
                print(f"  Skin thresh {skinFracThresh}: {testRange}")

        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        xRangeStrings = findVisibleXRange(gray, stringLines)
        rawScores = []
        for x in range(0, w, 4):
            minInt = float("inf")
            for x1, y1, x2, y2 in stringLines:
                if abs(x2 - x1) < 1e-6:
                    sy = (y1 + y2) / 2
                else:
                    t = np.clip((x - x1) / (x2 - x1), 0, 1)
                    sy = y1 + t * (y2 - y1)
                y = int(sy)
                y1b, y2b = max(0, y - 5), min(h, y + 6)
                x1b, x2b = max(0, x - 2), min(w, x + 3)
                v = np.mean(np.abs(gy[y1b:y2b, x1b:x2b]))
                minInt = min(minInt, v)
            rawScores.append(minInt)
        if rawScores and frameIdx == 0:
            peak = max(rawScores)
            good = sum(1 for s in rawScores if s >= peak * 0.5)
            print(f"  String visibility: peak={peak:.1f} good@0.5={good} w//6={w//6}")

        bbox = detectHandsRegion(frame, gray, stringLines, roiY1, roiY2, h, w)
        skinDensity = []
        for x in range(0, w, 4):
            col = skinRoi[:, max(0, x - 4):min(w, x + 5)]
            frac = np.sum(col > 0) / (col.size + 1e-6)
            skinDensity.append(frac)
        skinDensity = np.array(skinDensity)
        if frameIdx == 0 and len(skinDensity) > 0:
            low = np.sum(skinDensity <= 0.15)
            print(f"  Skin density: min={skinDensity.min():.3f} max={skinDensity.max():.3f} cols<=0.15={low}")
        xs = np.arange(0, w, 4)[:len(skinDensity)]

        stringScores = []
        for x in range(0, w, 4):
            minInt = float("inf")
            for x1, y1, x2, y2 in stringLines:
                if abs(x2 - x1) < 1e-6:
                    sy = (y1 + y2) / 2
                else:
                    t = np.clip((x - x1) / (x2 - x1), 0, 1)
                    sy = y1 + t * (y2 - y1)
                y = int(sy)
                y1b = max(0, y - 5)
                y2b = min(h, y + 6)
                x1b = max(0, x - 2)
                x2b = min(w, x + 3)
                band = gy[y1b:y2b, x1b:x2b]
                v = np.mean(np.abs(band))
                minInt = min(minInt, v)
            stringScores.append(minInt)
        stringScores = np.array(stringScores)
        peak = np.max(stringScores) if len(stringScores) > 0 else 1
        stringNorm = stringScores / (peak + 1e-6)

        plotH = 80
        plotW = w
        skinPlot = np.zeros((plotH, plotW, 3), dtype=np.uint8)
        skinPlot[:] = (30, 30, 30)
        for i, x in enumerate(xs):
            if i >= len(skinDensity):
                break
            val = int(skinDensity[i] * 255)
            cv2.line(skinPlot, (x, plotH), (x, plotH - int(skinDensity[i] * plotH)), (0, 0, 255), 2)
        cv2.putText(skinPlot, "Skin density (red=high, between hands=low)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if xRangeSkin:
            cv2.rectangle(skinPlot, (xRangeSkin[0], 0), (xRangeSkin[1], plotH), (0, 255, 0), 2)
            cv2.putText(skinPlot, f"Skin: {xRangeSkin}", (xRangeSkin[0], plotH - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        strPlot = np.zeros((plotH, plotW, 3), dtype=np.uint8)
        strPlot[:] = (30, 30, 30)
        for i, x in enumerate(xs):
            if i >= len(stringNorm):
                break
            cv2.line(strPlot, (x, plotH), (x, plotH - int(stringNorm[i] * plotH)), (0, 255, 255), 2)
        cv2.putText(strPlot, "String visibility (yellow=high, between hands=high)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if xRangeStrings:
            cv2.rectangle(strPlot, (xRangeStrings[0], 0), (xRangeStrings[1], plotH), (0, 255, 0), 2)
            cv2.putText(strPlot, f"Strings: {xRangeStrings}", (xRangeStrings[0], plotH - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        skinVis = frame.copy()
        skinOverlay = np.zeros_like(frame)
        skinOverlay[:, :, 2] = skinMask
        skinVis = cv2.addWeighted(skinVis, 0.7, skinOverlay, 0.3, 0)

        combined = np.vstack([
            frame,
            skinVis,
            skinPlot,
            strPlot
        ])
        if bbox:
            cv2.rectangle(combined, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
            cv2.putText(combined, f"Frame {frameIdx} bbox={bbox}", (bbox[0], bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        path = outDir / f"debug_frame_{frameIdx:04d}.png"
        cv2.imwrite(str(path), combined)
        print(f"Frame {frameIdx}: skin={xRangeSkin} strings={xRangeStrings} bbox={bbox}")

    cap.release()
    print(f"Debug images saved to {outDir}")


if __name__ == "__main__":
    main()
