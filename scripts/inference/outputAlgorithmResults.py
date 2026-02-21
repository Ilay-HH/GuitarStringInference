"""
Output 100 consecutive frames as separate images with the algorithm-detected used region.
Uses handsRegionDetector (skin + string visibility), no calibration.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import detectStringLinesAngled, detectStringLinesInHandsRegion, fallbackStringLines
from scripts.inference.frameAnnotator import colorEdgesByString
from scripts.hands_region.handsRegionDetector import detectHandsRegionForFrame, HandsRegionDetector


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
    ret, first = cap.read()
    if not ret:
        print("Cannot read video")
        return
    h, w = first.shape[:2]
    roiY1, roiY2 = int(h * 0.2), int(h * 0.8)
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    result = detectStringLinesInHandsRegion(first, gray, 6, roiY1, roiY2, returnCrop=True)
    stringLines, handsX1, handsX2 = result[0], result[1], result[2]
    if stringLines is None:
        roiEdges = result[4]
        stringLines = detectStringLinesAngled(roiEdges, 6, 0, roiEdges.shape[0], yOffset=roiY1)
        if stringLines is not None:
            stringLines = [(l[0] + handsX1, l[1], l[2] + handsX1, l[3]) for l in stringLines]
    if stringLines is None:
        stringLines = fallbackStringLines(h, w, 6, roiY1, roiY2)
        handsX1, handsX2 = 0, w

    colors = [
        (100, 100, 255), (50, 150, 255), (100, 255, 100),
        (255, 200, 50), (255, 100, 200), (100, 200, 255)
    ]

    outDir = PROJECT_ROOT / "output" / "algorithm_100_frames"
    outDir.mkdir(parents=True, exist_ok=True)
    detector = HandsRegionDetector(alpha=0.7)

    for i in range(100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roiCropped = gray[roiY1:roiY2, handsX1:handsX2]
        roiEdges = cv2.Canny(roiCropped, 50, 150)
        fullEdges = np.zeros_like(gray)
        fullEdges[roiY1:roiY2, handsX1:handsX2] = roiEdges
        coloredEdges = colorEdgesByString(fullEdges, stringLines, colors)

        bbox = detectHandsRegionForFrame(frame, gray, stringLines, roiY1, roiY2, h, w, detector=detector)
        xLo, yLo, xHi, yHi = bbox

        overlay = cv2.convertScaleAbs(frame, alpha=0.5, beta=0)
        overlay[yLo:yHi, xLo:xHi] = cv2.addWeighted(
            frame[yLo:yHi, xLo:xHi], 0.7,
            coloredEdges[yLo:yHi, xLo:xHi], 0.5, 0
        )
        cv2.rectangle(overlay, (xLo, yLo), (xHi, yHi), (0, 255, 255), 2)
        cv2.putText(overlay, f"Frame {i} (algorithm)", (xLo, yLo - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        path = outDir / f"frame_{i:04d}.png"
        cv2.imwrite(str(path), overlay)

    cap.release()
    print(f"Saved 100 images to {outDir}")


if __name__ == "__main__":
    main()
