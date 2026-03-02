"""
Temp example: run getStringLinesFromImage and getStringLinesFromEdgemap on an image.
Outputs result images to output/temp_string_lines_example/.
"""
import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import getStringLinesFromImage, getStringLinesFromEdgemap

COLORS = [
    (100, 100, 255), (50, 150, 255), (100, 255, 100),
    (255, 200, 50), (255, 100, 200), (100, 200, 255),
]


def main():
    imgPath = PROJECT_ROOT / "edited_guitar_image_1.png"
    if not imgPath.exists():
        print(f"Image not found: {imgPath}")
        return
    frame = cv2.imread(str(imgPath))
    if frame is None:
        print("Cannot read image")
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    outDir = PROJECT_ROOT / "output" / "temp_string_lines_example"
    outDir.mkdir(parents=True, exist_ok=True)

    linesFromImage, handsX1, handsY1, handsX2, handsY2 = getStringLinesFromImage(frame, 6)
    roiGray = gray[handsY1:handsY2, handsX1:handsX2]
    roiEdges = cv2.Canny(roiGray, 50, 150)

    linesFromEdgemap = getStringLinesFromEdgemap(
        roiEdges, numStrings=6, roiY1=0, roiY2=roiEdges.shape[0],
        yOffset=handsY1, xOffset=handsX1
    )

    def drawStringLines(vis, stringLines, label):
        for i, (x1, y1, x2, y2) in enumerate(stringLines):
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[i], 2)
            cv2.putText(vis, str(i + 1), (int(x1) + 5, int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[i], 2)
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return vis

    visImage = drawStringLines(frame.copy(), linesFromImage, "getStringLinesFromImage")
    cv2.imwrite(str(outDir / "01_from_image.png"), visImage)

    if linesFromEdgemap:
        visEdgemap = drawStringLines(frame.copy(), linesFromEdgemap, "getStringLinesFromEdgemap")
        cv2.imwrite(str(outDir / "02_from_edgemap.png"), visEdgemap)
    else:
        visEdgemap = frame.copy()
        cv2.putText(visEdgemap, "getStringLinesFromEdgemap: None", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(outDir / "02_from_edgemap.png"), visEdgemap)

    cv2.imwrite(str(outDir / "03_edgemap.png"), roiEdges)

    print(f"getStringLinesFromImage: {len(linesFromImage)} strings")
    print(f"getStringLinesFromEdgemap: {len(linesFromEdgemap) if linesFromEdgemap else 0} strings")
    print(f"Saved to {outDir}")


if __name__ == "__main__":
    main()
