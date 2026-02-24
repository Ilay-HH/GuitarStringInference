"""
Run 15 Hough parameter variations on the same ROI and render a comparison grid.
3 groups of 5: sweep threshold, maxLineGap, minLineLength independently.
Yellow = raw Hough lines. Blue = selected string (merged cluster). Green = single-Hough-line cluster.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import detectStringLinesInHandsRegion
from scripts.hands_region.handsRegionDetector import getProcessingRoi


def _runVariant(roiEdges, threshold, minLineLength, maxLineGap, handsY1, numStrings=6):
    """Return (houghVis, nRawLines, nCandidates, nSelected) for one parameter set."""
    CLUSTER_DIST = 15
    X_START_MAX_FRAC = 0.5
    MAX_ANGLE_DEG = 35

    h, w = roiEdges.shape
    midX = w / 2

    def heightAtMid(x1, y1, x2, y2):
        if abs(x2 - x1) < 1e-6:
            return (y1 + y2) / 2
        t = np.clip((midX - x1) / (x2 - x1), 0, 1)
        return y1 + t * (y2 - y1)

    lines = cv2.HoughLinesP(
        roiEdges, rho=1, theta=np.pi / 180,
        threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap
    )
    vis = cv2.cvtColor(roiEdges, cv2.COLOR_GRAY2BGR)

    if lines is None:
        return vis, 0, 0, 0, 0

    nRaw = len(lines)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)

    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        y1f, y2f = y1 + handsY1, y2 + handsY1
        angle = np.degrees(np.arctan2(y2f - y1f, x2 - x1))
        if abs(angle) <= MAX_ANGLE_DEG or abs(abs(angle) - 180) <= MAX_ANGLE_DEG:
            candidates.append((x1, y1f, x2, y2f))

    if len(candidates) < numStrings:
        return vis, nRaw, len(candidates), 0, 0

    candidates.sort(key=lambda L: heightAtMid(*L))

    clusters = []
    current = [candidates[0]]
    for seg in candidates[1:]:
        if heightAtMid(*seg) - heightAtMid(*current[-1]) <= CLUSTER_DIST:
            current.append(seg)
        else:
            clusters.append(current)
            current = [seg]
    clusters.append(current)

    merged = []
    for cluster in clusters:
        avg_x1 = int(np.mean([l[0] for l in cluster]))
        avg_y1 = int(np.mean([l[1] for l in cluster]))
        avg_x2 = int(np.mean([l[2] for l in cluster]))
        avg_y2 = int(np.mean([l[3] for l in cluster]))
        x_start = min(min(l[0], l[2]) for l in cluster)
        merged.append({'line': (avg_x1, avg_y1, avg_x2, avg_y2), 'x_start': x_start, 'members': cluster})

    nClusters = len(merged)
    x_threshold = w * X_START_MAX_FRAC
    filtered = [m for m in merged if m['x_start'] <= x_threshold]
    if len(filtered) < numStrings:
        filtered = merged

    filtered.sort(key=lambda m: heightAtMid(*m['line']))
    n = len(filtered)
    if n < numStrings:
        return vis, nRaw, len(candidates), 0, nClusters

    indices = [int(i * (n - 1) / (numStrings - 1)) for i in range(numStrings)] if numStrings > 1 else [0]
    selected = [filtered[i] for i in indices]

    # Refine outer strings from inner spacing
    innerYs = [heightAtMid(*selected[i]['line']) for i in range(1, numStrings - 1)]
    avgSpacing = float(np.mean([innerYs[j + 1] - innerYs[j] for j in range(len(innerYs) - 1)]))
    extraY0 = innerYs[0] - avgSpacing
    extraYN = innerYs[-1] + avgSpacing
    innerIds = {id(m) for m in selected[1:-1]}

    def _nearest(targetY):
        best, bestDist = None, float('inf')
        for m in merged:
            if id(m) in innerIds:
                continue
            d = abs(heightAtMid(*m['line']) - targetY)
            if d < bestDist:
                bestDist, best = d, m
        return best

    cand0 = _nearest(extraY0)
    if cand0 is not None:
        innerIds.add(id(cand0))
    candN = _nearest(extraYN)
    if cand0 is not None:
        selected[0] = cand0
    if candN is not None:
        selected[-1] = candN
    selected.sort(key=lambda m: heightAtMid(*m['line']))

    for m in selected:
        members = m['members']
        avgX1 = int(np.mean([l[0] for l in members]))
        avgY1 = int(np.mean([l[1] for l in members])) - handsY1
        avgX2 = int(np.mean([l[2] for l in members]))
        avgY2 = int(np.mean([l[3] for l in members])) - handsY1
        color = (0, 255, 0) if len(members) == 1 else (255, 100, 0)
        cv2.line(vis, (avgX1, avgY1), (avgX2, avgY2), color, 2)

    return vis, nRaw, len(candidates), len(selected), nClusters


def _labelImage(img, label, sublabel, nRaw, nCandidates, nClusters, nSelected):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(out, sublabel, (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    ok = nSelected == 6
    selColor = (80, 220, 80) if ok else (60, 60, 220)
    info = f"raw:{nRaw} cand:{nCandidates} clust:{nClusters} sel:{nSelected}"
    cv2.putText(out, info, (w - 230, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, selColor, 1)
    if not ok:
        cv2.putText(out, "CLUST FAIL", (w - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 220), 1)
    return out


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
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read frame")
        return

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = detectStringLinesInHandsRegion(frame, gray, 6, returnCrop=True)
    stringLines, handsX1, handsY1, handsX2, handsY2, roiGray, roiEdges = result
    detectionY1 = handsY1

    print(f"ROI: x=[{handsX1},{handsX2}] y=[{handsY1},{handsY2}]  edges shape: {roiEdges.shape}")

    roiW = roiEdges.shape[1]
    BASE_THRESHOLD = 50
    BASE_MIN_LEN = roiW // 4
    BASE_MAX_GAP = 25

    # 15 variants: 3 groups of 5, each sweeping one parameter
    variants = []

    # Group 1: threshold sweep
    for t in [30, 40, 50, 60, 70]:
        label = f"thresh={t}"
        sublabel = f"minLen={BASE_MIN_LEN}  gap={BASE_MAX_GAP}"
        variants.append((label, sublabel, t, BASE_MIN_LEN, BASE_MAX_GAP))

    # Group 2: maxLineGap sweep (threshold fixed to 40 as a better baseline)
    for g in [10, 20, 35, 50, 70]:
        label = f"gap={g}"
        sublabel = f"thresh=40  minLen={BASE_MIN_LEN}"
        variants.append((label, sublabel, 40, BASE_MIN_LEN, g))

    # Group 3: minLineLength sweep (threshold=40, gap=35)
    for d in [6, 5, 4, 3, 2]:
        minLen = roiW // d
        label = f"minLen=w//{d} ({minLen}px)"
        sublabel = f"thresh=40  gap=35"
        variants.append((label, sublabel, 40, minLen, 35))

    CELL_W, CELL_H = 630, 210
    COLS, ROWS = 5, 3
    grid = np.zeros((ROWS * CELL_H, COLS * CELL_W, 3), dtype=np.uint8)
    grid[:] = (30, 30, 30)

    outDir = PROJECT_ROOT / "output" / "hough_param_compare"
    outDir.mkdir(parents=True, exist_ok=True)

    for idx, (label, sublabel, threshold, minLen, maxGap) in enumerate(variants):
        vis, nRaw, nCand, nSel, nClust = _runVariant(roiEdges, threshold, minLen, maxGap, detectionY1)
        vis = _labelImage(vis, label, sublabel, nRaw, nCand, nClust, nSel)
        cv2.imwrite(str(outDir / f"variant_{idx+1:02d}_{label.replace('=','').replace('/','_').replace(' ','_')}.png"), vis)

        cell = cv2.resize(vis, (CELL_W, CELL_H))
        row, col = idx // COLS, idx % COLS
        grid[row * CELL_H:(row + 1) * CELL_H, col * CELL_W:(col + 1) * CELL_W] = cell

        print(f"  [{idx+1:2d}] {label:25s} | raw={nRaw:3d} cand={nCand:3d} clust={nClust:3d} sel={nSel}")

    # Group labels on left margin separator lines
    for r, grpLabel in enumerate(["Group 1: threshold (minLen=w//4, gap=25)",
                                   "Group 2: maxLineGap (thresh=40, minLen=w//4)",
                                   "Group 3: minLineLength (thresh=40, gap=35)"]):
        y = r * CELL_H + 48
        cv2.putText(grid, grpLabel, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 60), 1)
        if r > 0:
            cv2.line(grid, (0, r * CELL_H), (COLS * CELL_W, r * CELL_H), (80, 80, 80), 2)

    gridPath = outDir / "grid_all_15.png"
    cv2.imwrite(str(gridPath), grid)
    print(f"\nGrid saved to {gridPath}")
    print(f"Individual images in {outDir}")


if __name__ == "__main__":
    main()
