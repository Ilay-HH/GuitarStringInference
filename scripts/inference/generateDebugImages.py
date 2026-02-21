"""
Generates debug images for PROCESS.md - each step of calibration and visualization.
"""
import cv2
import numpy as np
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import detectStringLinesAngled, detectStringLinesInHandsRegion, fallbackStringLines
from scripts.hands_region.handsRegionDetector import getProcessingRoi
from scripts.inference.frameAnnotator import colorEdgesByString


def main():
    dataDir = PROJECT_ROOT / "data"
    videoPath = dataDir / "The most beautiful melody line.mp4"
    if not videoPath.exists():
        videos = list(dataDir.glob("*.mp4"))
        videoPath = videos[0] if videos else None
    if not videoPath or not videoPath.exists():
        print("No video found in data/")
        return
    outDir = PROJECT_ROOT / "output" / "debug"
    outDir.mkdir(parents=True, exist_ok=True)
    for f in outDir.glob("*.png"):
        f.unlink()
    cap = cv2.VideoCapture(str(videoPath))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read frame")
        return
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = detectStringLinesInHandsRegion(frame, gray, 6, returnCrop=True)
    stringLines, handsX1, handsY1, handsX2, handsY2, roiGray, roiEdges = result
    if stringLines is None:
        stringLines = detectStringLinesAngled(roiEdges, 6, 0, roiEdges.shape[0], yOffset=handsY1)
        if stringLines is not None:
            stringLines = [(l[0] + handsX1, l[1], l[2] + handsX1, l[3]) for l in stringLines]
    if stringLines is None:
        stringLines = fallbackStringLines(h, w, 6, handsY1, handsY2)
        handsX1, handsY1, handsX2, handsY2 = 0, int(h * 0.2), w, int(h * 0.8)
        roiGray = gray[handsY1:handsY2, :]
        roiEdges = cv2.Canny(roiGray, 50, 150)
    else:
        bbox = getProcessingRoi(frame, gray, h, w, stringLines)
        handsX1, handsY1, handsX2, handsY2 = bbox
        roiGray = gray[handsY1:handsY2, handsX1:handsX2]
        roiEdges = cv2.Canny(roiGray, 50, 150)
    colors = [
        (100, 100, 255), (50, 150, 255), (100, 255, 100),
        (255, 200, 50), (255, 100, 200), (100, 200, 255)
    ]
    roiRect = frame.copy()
    cv2.rectangle(roiRect, (handsX1, handsY1), (handsX2, handsY2), (0, 255, 255), 2)
    cv2.putText(roiRect, "processing ROI (hands region)", (handsX1, handsY1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.imwrite(str(outDir / "01_original.png"), frame)
    cv2.imwrite(str(outDir / "02_roi_marked.png"), roiRect)
    cv2.imwrite(str(outDir / "03_roi_grayscale.png"), roiGray)
    cv2.imwrite(str(outDir / "04_canny_edges.png"), roiEdges)
    lines = cv2.HoughLinesP(roiEdges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=max(roiEdges.shape[1] // 4, 50), maxLineGap=25)
    houghVis = cv2.cvtColor(roiEdges, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(houghVis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.imwrite(str(outDir / "05_hough_lines.png"), houghVis)
    linesVis = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(stringLines):
        cv2.line(linesVis, (int(x1), int(y1)), (int(x2), int(y2)), colors[i], 2)
        cv2.putText(linesVis, str(i + 1), (int(x1) + 5, int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    cv2.imwrite(str(outDir / "06_string_lines.png"), linesVis)
    bandVis = frame.copy()
    xs = np.arange(0, w, max(1, w // 100))
    for i, (x1, y1, x2, y2) in enumerate(stringLines):
        if abs(x2 - x1) < 1e-6:
            ys = np.full_like(xs, (y1 + y2) / 2)
        else:
            t = np.clip((xs - x1) / (x2 - x1), 0, 1)
            ys = y1 + t * (y2 - y1)
        pts = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(bandVis, [pts], False, colors[i], 2)
        cv2.putText(bandVis, str(i + 1), (int(pts[0, 0]) + 5, int(pts[0, 1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    boundaryPts = []
    for b in range(5):
        mids = []
        for x in xs:
            yCoords = []
            for (x1, y1, x2, y2) in stringLines:
                if abs(x2 - x1) < 1e-6:
                    yCoords.append((y1 + y2) / 2)
                else:
                    t = np.clip((x - x1) / (x2 - x1), 0, 1)
                    yCoords.append(y1 + t * (y2 - y1))
            mid = (yCoords[b] + yCoords[b + 1]) / 2
            mids.append((x, mid))
        pts = np.array(mids, dtype=np.int32)
        cv2.polylines(bandVis, [pts], False, (255, 255, 255), 1)
    cv2.imwrite(str(outDir / "07_band_boundaries.png"), bandVis)
    fullEdges = np.zeros_like(gray)
    fullEdges[handsY1:handsY2, handsX1:handsX2] = roiEdges
    coloredEdges = colorEdgesByString(fullEdges, stringLines, colors)
    cv2.imwrite(str(outDir / "08_colored_edges.png"), coloredEdges)
    overlay = cv2.addWeighted(frame, 0.7, coloredEdges, 0.5, 0)
    pad, boxW, boxH = 10, 100, 6 * 22 + 20
    x1, y1 = w - boxW - pad, h - boxH - pad
    cv2.rectangle(overlay, (x1, y1), (x1 + boxW, y1 + boxH), (40, 40, 40), -1)
    for i in range(6):
        y = y1 + pad + i * 22 + 14
        cv2.circle(overlay, (x1 + 18, y - 4), 6, colors[i], -1)
        cv2.putText(overlay, f"Str {i + 1}", (x1 + 30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(outDir / "09_final_overlay.png"), overlay)
    docsDir = PROJECT_ROOT / "docs" / "images" / "string_tracking"
    docsDir.mkdir(parents=True, exist_ok=True)
    generated = [
        "01_original.png", "02_roi_marked.png", "03_roi_grayscale.png", "04_canny_edges.png",
        "05_hough_lines.png", "06_string_lines.png", "07_band_boundaries.png",
        "08_colored_edges.png", "09_final_overlay.png"
    ]
    for name in generated:
        shutil.copy(outDir / name, docsDir / name)
    edgesSrc = PROJECT_ROOT / "output" / "test_edges.png"
    if edgesSrc.exists():
        shutil.copy(edgesSrc, docsDir / "test_edges.png")
    for old in ["02_grayscale.png", "03_canny_edges.png", "04_hough_lines.png", "05_string_lines.png",
                "06_band_boundaries.png", "07_colored_edges.png", "08_final_overlay.png"]:
        (docsDir / old).unlink(missing_ok=True)
    print(f"Saved {len(generated)} debug images to {outDir} and {docsDir}")


if __name__ == "__main__":
    main()
