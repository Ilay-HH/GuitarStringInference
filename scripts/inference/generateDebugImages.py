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
    _initLines, handsX1, handsY1, handsX2, handsY2, roiGray, roiEdges = result
    # Single authoritative detection call: returnDebug=True gives us both the
    # string lines AND the cluster debug info so 05 and 06 are always in sync.
    detectionRoiEdges = roiEdges
    detectionY1 = handsY1
    _roiResult, dbgInfo = detectStringLinesAngled(
        detectionRoiEdges, 6, 0, detectionRoiEdges.shape[0],
        yOffset=detectionY1, returnDebug=True)
    if _roiResult is not None:
        # Convert ROI-local x back to full-frame x
        stringLines = [
            (l[0] + handsX1, l[1], l[2] + handsX1, l[3]) for l in _roiResult
        ]
    else:
        stringLines = fallbackStringLines(h, w, 6, handsY1, handsY2)
        handsX1, handsY1, handsX2, handsY2 = 0, int(h * 0.2), w, int(h * 0.8)
        roiGray = gray[handsY1:handsY2, :]
        roiEdges = cv2.Canny(roiGray, 50, 150)
        detectionRoiEdges = roiEdges
        detectionY1 = handsY1
        dbgInfo = {'hough_candidates': [], 'selected_members': [], 'outer_refined': []}
    if _roiResult is not None:
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
    lines = cv2.HoughLinesP(detectionRoiEdges, rho=1, theta=np.pi / 180, threshold=50,
                            minLineLength=max(detectionRoiEdges.shape[1] // 4, 50), maxLineGap=25)
    houghVis = cv2.cvtColor(detectionRoiEdges, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(houghVis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    x_ref = dbgInfo.get('inner_x_ref')
    if x_ref is not None:
        cv2.line(houghVis, (x_ref, 0), (x_ref, houghVis.shape[0] - 1), (0, 0, 255), 1)
        cv2.putText(houghVis, f"x={x_ref}", (x_ref + 3, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                xs, xe = min(x1, x2), max(x1, x2)
                if xs <= x_ref <= xe:
                    t = (x_ref - x1) / (x2 - x1) if abs(x2 - x1) > 0 else 0.5
                    iy = int(y1 + t * (y2 - y1))
                    cv2.circle(houghVis, (x_ref, iy), 3, (0, 0, 255), -1)
    cv2.imwrite(str(outDir / "05_hough_lines.png"), houghVis)

    # 05b: pair each hit line with its string-edge partner and draw midpoint string lines
    _PAIR_DIST = 20  # max y-gap between the two edges of one string

    def _y_at(seg, x):
        x1, y1, x2, y2 = seg
        xs, xe = min(x1, x2), max(x1, x2)
        if x < xs or x > xe:
            return None
        if abs(x2 - x1) < 1:
            return (y1 + y2) / 2.0
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    # Angle-filtered candidates in ROI-local coords
    _MAX_ANGLE = 35
    acands = []
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(ang) <= _MAX_ANGLE or abs(abs(ang) - 180) <= _MAX_ANGLE:
                acands.append((x1, y1, x2, y2))

    # Lines that span x_ref, sorted by y there
    hit = []
    if x_ref is not None:
        for idx, l in enumerate(acands):
            y = _y_at(l, x_ref)
            if y is not None:
                hit.append((y, idx))
        hit.sort(key=lambda t: t[0])

    used_idx = set()
    string_segs = []

    for y_i, idx_i in hit:
        if idx_i in used_idx:
            continue
        l_i = acands[idx_i]
        xe_i = max(l_i[0], l_i[2])
        best_jdx = None
        best_pair_x = None

        for jdx, l_j in enumerate(acands):
            if jdx == idx_i or jdx in used_idx:
                continue
            xs_j = min(l_j[0], l_j[2])
            xe_j = max(l_j[0], l_j[2])
            scan_start = max(x_ref, xs_j)
            scan_end = min(xe_i, xe_j)
            if scan_start > scan_end:
                continue
            ya_s = _y_at(l_i, scan_start)
            yb_s = _y_at(l_j, scan_start)
            if ya_s is None or yb_s is None:
                continue
            d_s = abs(ya_s - yb_s)
            if d_s <= _PAIR_DIST:
                pair_x = scan_start
            else:
                ya_e = _y_at(l_i, scan_end)
                yb_e = _y_at(l_j, scan_end)
                if ya_e is None or yb_e is None:
                    continue
                d_e = abs(ya_e - yb_e)
                if d_e >= d_s or d_e > _PAIR_DIST:
                    continue
                frac = (d_s - _PAIR_DIST) / (d_s - d_e)
                pair_x = int(scan_start + frac * (scan_end - scan_start))
            if best_pair_x is None or pair_x < best_pair_x:
                best_pair_x = pair_x
                best_jdx = jdx

        if best_jdx is not None:
            used_idx.add(idx_i)
            used_idx.add(best_jdx)
            string_segs.append((l_i, acands[best_jdx], y_i))

    _pair_colors = [
        (0, 255, 0), (0, 165, 255), (0, 0, 255),
        (255, 0, 255), (0, 255, 255), (255, 128, 0),
        (128, 0, 255), (0, 200, 100), (200, 200, 0), (255, 0, 128),
    ]
    # Build midpoint lines in ROI-local coords, then convert to full-frame
    # Each midpoint line: from x_ref (on l_i) to end of pair overlap
    pairMidLines = []
    for la, lb, y_hit in string_segs:
        xe_a = max(la[0], la[2])
        xe_b = max(lb[0], lb[2])
        end_x = min(xe_a, xe_b)
        ya_end = _y_at(la, end_x)
        yb_end = _y_at(lb, end_x)
        if ya_end is None or yb_end is None:
            continue
        mid_y_end = (ya_end + yb_end) / 2.0
        # Convert ROI-local -> full-frame
        fx1 = x_ref + handsX1
        fy1 = int(y_hit) + detectionY1
        fx2 = end_x + handsX1
        fy2 = int(mid_y_end) + detectionY1
        pairMidLines.append((fx1, fy1, fx2, fy2))

    pairVis = houghVis.copy()
    for pi, (la, lb, _) in enumerate(string_segs):
        col = _pair_colors[pi % len(_pair_colors)]
        x1a, y1a, x2a, y2a = la
        x1b, y1b, x2b, y2b = lb
        cv2.line(pairVis, (x1a, y1a), (x2a, y2a), col, 2)
        cv2.line(pairVis, (x1b, y1b), (x2b, y2b), col, 2)
    cv2.imwrite(str(outDir / "05b_string_pairs.png"), pairVis)

    drawLines = pairMidLines if len(pairMidLines) == len(colors) else stringLines
    linesVis = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(drawLines):
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
        "05_hough_lines.png", "05b_string_pairs.png", "06_string_lines.png", "07_band_boundaries.png",
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
