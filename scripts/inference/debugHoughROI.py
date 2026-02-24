"""
Non-interactive Hough debug tool.
Reconstructs the detection-time Canny edges (same image used for Hough in the pipeline),
then for each of the 6 strings analyses a left-start patch and sweeps Hough params.
Produces annotated images saved to output/debug_hough/.

Usage: python -m scripts.inference.debugHoughROI [--x-end N]
  --x-end N  right edge of the "start region" to inspect (default: first 200px of ROI)
"""
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



def _rowProfile(patch):
    """Return list of (row_idx, edge_count) for non-empty rows."""
    return [(r, int(np.sum(patch[r] > 0))) for r in range(patch.shape[0])
            if np.sum(patch[r] > 0) > 0]


def _edgeBands(patch):
    """Group consecutive rows that have edges into bands."""
    row_has = [np.sum(patch[r] > 0) > 0 for r in range(patch.shape[0])]
    bands, in_band, start = [], False, 0
    for i, has in enumerate(row_has):
        if has and not in_band:
            in_band, start = True, i
        elif not has and in_band:
            in_band = False
            bands.append((start, i - 1))
    if in_band:
        bands.append((start, patch.shape[0] - 1))
    return bands


def _sweepHough(patch, label, full_roi_w):
    h, w = patch.shape
    n_edge = int(np.sum(patch > 0))
    print(f"\n  [{label}]  patch {w}x{h} px  edge_px={n_edge}")
    if n_edge == 0:
        print("    (no edge pixels)")
        return

    configs = [
        (50, full_roi_w // 4, 25, "current baseline"),
        (50, full_roi_w // 6, 25, "shorter minLen"),
        (30, full_roi_w // 6, 25, "lower thr + shorter"),
        (20, full_roi_w // 8, 25, "very low thr + short"),
        (10, full_roi_w // 8, 25, "minimal thr"),
        (10, max(5, w // 3),  25, "patch-relative minLen"),
        (10, max(5, w // 4),  50, "patch-relative + big gap"),
        (5,  max(5, w // 4),  50, "tiny thr + patch-relative"),
    ]
    first_hit = True
    for thr, mll, gap, note in configs:
        lines = cv2.HoughLinesP(patch, 1, np.pi / 180,
                                threshold=thr, minLineLength=mll, maxLineGap=gap)
        n = len(lines) if lines is not None else 0
        hit = n > 0
        arrow = " <<< FIRST DETECTION" if hit and first_hit else ""
        if hit:
            first_hit = False
        print(f"    thresh={thr:3d}  minLen={mll:4d}  gap={gap:2d}  lines={n:3d}  ({note}){arrow}")


def _annotateVis(vis, patch_x0, patch_y0, bands, string_idx, color):
    """Draw the detected edge bands on the vis image as coloured rectangles."""
    for b_start, b_end in bands:
        y0 = patch_y0 + b_start
        y1 = patch_y0 + b_end
        cv2.rectangle(vis, (patch_x0, y0), (patch_x0 + 1, y1), color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-end", type=int, default=200,
                        help="Right edge of start-region to inspect in ROI x-coords (default 200)")
    args = parser.parse_args()

    dataDir = PROJECT_ROOT / "data"
    videoPath = next(dataDir.glob("*.mp4"), None)
    if not videoPath:
        print("No video found in data/")
        return

    cap = cv2.VideoCapture(str(videoPath))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Cannot read frame")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    debugDir = PROJECT_ROOT / "output" / "debug"
    cannyPath = debugDir / "05_detection_canny.png"
    statePath = debugDir / "detection_state.json"

    if not cannyPath.exists() or not statePath.exists():
        print("Run scripts/inference/generateDebugImages.py first to generate detection state.")
        return

    import json
    with open(statePath) as f:
        state = json.load(f)

    handsX1 = state["handsX1"]
    handsY1 = state["handsY1"]
    handsX2 = state["handsX2"]
    handsY2 = state["handsY2"]
    stringLines = [tuple(l) for l in state["stringLines"]]

    detectionRoiEdges = cv2.imread(str(cannyPath), cv2.IMREAD_GRAYSCALE)
    if detectionRoiEdges is None:
        print(f"Cannot load {cannyPath}")
        return

    roiH, roiW = detectionRoiEdges.shape
    print(f"Detection ROI: x=[{handsX1},{handsX2}] y=[{handsY1},{handsY2}]  "
          f"edges shape: {roiW}x{roiH}")
    print(f"Inspecting left-start region: x=0..{args.x_end}\n")

    outDir = PROJECT_ROOT / "output" / "debug_hough"
    outDir.mkdir(parents=True, exist_ok=True)

    # Build annotated overlay: Canny edges coloured by string, with edge-band markers
    vis = cv2.cvtColor(detectionRoiEdges, cv2.COLOR_GRAY2BGR)
    colors = [(0,0,255),(0,128,255),(0,255,128),(0,255,0),(255,200,0),(255,0,200)]

    x_end = min(args.x_end, roiW)

    for si, (sx1, sy1, sx2, sy2) in enumerate(stringLines):
        # Convert string line from full-frame to ROI coords
        rx1 = int(sx1 - handsX1)
        ry1 = int(sy1 - handsY1)
        rx2 = int(sx2 - handsX1)
        ry2 = int(sy2 - handsY1)

        # y at x=0 in ROI
        if abs(rx2 - rx1) > 0:
            y_at_x0 = int(ry1 + (0 - rx1) * (ry2 - ry1) / (rx2 - rx1))
        else:
            y_at_x0 = (ry1 + ry2) // 2

        # Build a generous patch around this string's y at the start region
        pad = 20
        py0 = max(0, y_at_x0 - pad)
        py1 = min(roiH, y_at_x0 + pad + 1)

        patch = detectionRoiEdges[py0:py1, 0:x_end]

        print(f"=== String {si+1}  ROI line ({rx1},{ry1})->({rx2},{ry2})  "
              f"y_at_x0={y_at_x0}  patch y=[{py0},{py1}] ===")

        profile = _rowProfile(patch)
        bands = _edgeBands(patch)

        print(f"  Edge bands (groups of rows with edges): {len(bands)}")
        for b in bands:
            total_px = sum(c for _, c in profile if b[0] <= _ <= b[1])
            print(f"    rows {b[0]:3d}-{b[1]:3d}  (h={b[1]-b[0]+1})  total_edge_px={total_px}")

        _sweepHough(patch, f"String {si+1} start patch", roiW)

        # Annotate vis: draw patch bounds and edge bands
        color = colors[si % len(colors)]
        cv2.rectangle(vis, (0, py0), (x_end, py1), color, 1)
        for b_start, b_end in bands:
            cv2.line(vis, (0, py0 + b_start), (5, py0 + b_start), color, 1)
            cv2.line(vis, (0, py0 + b_end),   (5, py0 + b_end),   color, 1)

        # Save zoomed patch (8x)
        patch_vis = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        zoom = cv2.resize(patch_vis, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        # Annotate bands on zoom
        for b_start, b_end in bands:
            zy = b_start * 8
            cv2.rectangle(zoom, (0, zy), (zoom.shape[1]-1, b_end*8+7), color, 1)
        cv2.putText(zoom, f"String {si+1}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(outDir / f"str{si+1}_start_zoom8x.png"), zoom)

    cv2.imwrite(str(outDir / "detection_roi_annotated.png"), vis)
    zoom_vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(outDir / "detection_roi_annotated_2x.png"), zoom_vis)
    print(f"\nAnnotated ROI saved to {outDir}/detection_roi_annotated.png")
    print(f"Per-string zoomed patches in {outDir}/str*_start_zoom8x.png")


if __name__ == "__main__":
    main()
