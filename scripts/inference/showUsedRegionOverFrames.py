"""
Shows the used region (from calibration interpolation) over 100 consecutive frames.
Creates a 10x10 grid so you can see how the bbox changes.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    numFrames = 100
    cellW, cellH = 170, 96
    gridCols, gridRows = 10, 10
    outW, outH = gridCols * cellW, gridRows * cellH
    grid = np.zeros((outH, outW, 3), dtype=np.uint8)
    grid[:] = (40, 40, 40)

    for i in range(numFrames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        bbox = getHandsBbox(str(videoPath), i, h, w)
        if bbox is None:
            bbox = (0, 0, w, h)
        x1, y1, x2, y2 = bbox
        disp = frame.copy()
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(disp, f"f{i}", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        disp = cv2.resize(disp, (cellW, cellH))
        row, col = i // gridCols, i % gridCols
        y0, x0 = row * cellH, col * cellW
        grid[y0:y0 + cellH, x0:x0 + cellW] = disp

    cap.release()

    outPath = PROJECT_ROOT / "output" / "used_region_100_frames.png"
    outPath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outPath), grid)
    print(f"Saved {outPath}")
    print("Each cell = 1 frame. Yellow box = interpolated used region from your 9 calibrations.")


if __name__ == "__main__":
    main()
