"""
Tests hypothesis: played strings smear due to motion blur, reducing edge visibility.
Tracks edge intensity per string over time - drops indicate possible string activation.
Runnable standalone.
"""
import cv2
import numpy as np
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
PROJECT_ROOT = _PROJECT_ROOT if (_PROJECT_ROOT / "config").exists() else _THIS_DIR.parent


def detectStringLines(edgeImg, numStrings=6, roiY1=0, roiY2=None):
    """Find horizontal lines (strings) via Hough transform, returns y-positions."""
    if roiY2 is None:
        roiY2 = edgeImg.shape[0]
    roi = edgeImg[roiY1:roiY2, :]
    lines = cv2.HoughLinesP(
        roi, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=roi.shape[1] // 3, maxLineGap=20
    )
    if lines is None:
        return None
    ys = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 15 or angle > 165:
            ys.append((y1 + y2) // 2)
    if len(ys) < numStrings:
        return None
    ys = np.array(ys)
    ys = np.sort(ys)
    clustered = []
    for y in ys:
        if not clustered or y - clustered[-1] > 8:
            clustered.append(y)
    if len(clustered) >= numStrings:
        return np.array(clustered[:numStrings]) + roiY1
    return None


def _estimateDominantAngle(roi, maxAngleDeg):
    """Pass-1 Hough with loose params to estimate the dominant string angle in degrees.

    minLineLength is intentionally short so short start-of-string segments are captured.
    Returns the median angle of all near-horizontal candidates.
    """
    w = roi.shape[1]
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=30,
                            minLineLength=w // 8, maxLineGap=30)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        if angle_deg <= maxAngleDeg:
            angles.append(angle_deg)
    return float(np.median(angles)) if angles else 0.0


def detectStringLinesAngled(edgeImg, numStrings=6, roiY1=0, roiY2=None, maxAngleDeg=35, yOffset=None, returnDebug=False):
    """Find string lines via Hough with pair-clustering and x-start filtering.

    Assumes Hough outputs two lines per string (top and bottom edge). Clusters nearby
    lines by y at mid-x to merge each pair into one representative line, then filters
    out clusters that start too far right (neck edges start later than string lines).

    minLineLength scales with cos(dominantAngle): the same horizontal coverage corresponds
    to a shorter Euclidean segment at steeper angles. Base w//4 preserves existing behavior
    for near-horizontal strings; the correction only meaningfully kicks in above ~15 deg.
    """
    CLUSTER_DIST = 15

    if roiY2 is None:
        roiY2 = edgeImg.shape[0]
    if yOffset is None:
        yOffset = roiY1
    h, w = edgeImg.shape
    roi = edgeImg[roiY1:roiY2, :]

    # Pass 1: estimate dominant angle so minLineLength can account for it.
    # Base w//4 preserves existing behavior for near-horizontal strings; cos(angle)
    # reduces the requirement at steeper angles where the same horizontal coverage
    # corresponds to a shorter Euclidean segment.
    dominantAngleDeg = _estimateDominantAngle(roi, maxAngleDeg)
    minLineLength = max(roi.shape[1] // 8,
                        int(roi.shape[1] // 4 * np.cos(np.radians(dominantAngleDeg))))

    # Pass 2: main detection with angle-corrected minLineLength.
    lines = cv2.HoughLinesP(
        roi, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=minLineLength, maxLineGap=25
    )
    empty_debug = {'hough_candidates': [], 'selected_members': []}
    if lines is None:
        return (None, empty_debug) if returnDebug else None

    midX = w / 2

    def heightAtMid(x1, y1, x2, y2):
        if abs(x2 - x1) < 1e-6:
            return (y1 + y2) / 2
        t = np.clip((midX - x1) / (x2 - x1), 0, 1)
        return y1 + t * (y2 - y1)

    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        y1f, y2f = y1 + yOffset, y2 + yOffset
        angle = np.degrees(np.arctan2(y2f - y1f, x2 - x1))
        if abs(angle) <= maxAngleDeg or abs(abs(angle) - 180) <= maxAngleDeg:
            candidates.append((x1, y1f, x2, y2f))

    if len(candidates) < numStrings:
        return (None, {'hough_candidates': candidates, 'selected_members': []}) if returnDebug else None

    candidates.sort(key=lambda L: heightAtMid(*L))

    # Cluster consecutive lines whose y at mid-x are within CLUSTER_DIST pixels.
    # Each cluster represents one string (top + bottom edges merged).
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
        # Fit a regression line through all member endpoints so the representative line
        # spans the full cluster x-range and reflects the piecewise center.
        pts_x = np.array([l[0] for l in cluster] + [l[2] for l in cluster], dtype=float)
        pts_y = np.array([l[1] for l in cluster] + [l[3] for l in cluster], dtype=float)
        x_start = int(pts_x.min())
        x_end = int(pts_x.max())
        if x_end > x_start:
            slope_r, intercept_r = np.linalg.lstsq(
                np.column_stack([pts_x, np.ones(len(pts_x))]), pts_y, rcond=None)[0]
            rep_line = (x_start, int(slope_r * x_start + intercept_r),
                        x_end,   int(slope_r * x_end   + intercept_r))
        else:
            rep_line = (x_start, int(pts_y.mean()), x_end, int(pts_y.mean()))
        merged.append({'line': rep_line, 'x_start': x_start, 'members': cluster})

    # Soft x-start selection: divide the y-range into numStrings evenly-spaced slots;
    # within each slot prefer the cluster whose members begin earliest on the x-axis.
    # This naturally rejects late-starting neck-edge clusters without a hard threshold.
    merged.sort(key=lambda m: heightAtMid(*m['line']))
    n = len(merged)
    if n < numStrings:
        return (None, {'hough_candidates': candidates, 'selected_members': []}) if returnDebug else None

    y_vals = [heightAtMid(*m['line']) for m in merged]
    y_min, y_max = y_vals[0], y_vals[-1]
    slot_h = (y_max - y_min) / max(numStrings - 1, 1)

    selected = []
    used = set()
    for i in range(numStrings):
        target_y = y_min + slot_h * i
        nearby = [(j, merged[j]) for j in range(n)
                  if j not in used and abs(y_vals[j] - target_y) <= slot_h * 0.7]
        if not nearby:
            nearby = [(j, merged[j]) for j in range(n) if j not in used]
        if not nearby:
            break
        j_best, m_best = min(nearby, key=lambda jm: jm[1]['x_start'])
        selected.append(m_best)
        used.add(j_best)

    if len(selected) < numStrings:
        return (None, {'hough_candidates': candidates, 'selected_members': []}) if returnDebug else None

    selected.sort(key=lambda m: heightAtMid(*m['line']))
    result = [m['line'] for m in selected]

    if returnDebug:
        return result, {'hough_candidates': candidates, 'selected_members': [m['members'] for m in selected]}
    return result


def fallbackStringLines(h, w, numStrings=6, roiY1=None, roiY2=None):
    """Horizontal lines when angled detection fails."""
    if roiY1 is None:
        roiY1 = int(h * 0.25)
    if roiY2 is None:
        roiY2 = int(h * 0.75)
    step = (roiY2 - roiY1) / (numStrings + 1)
    return [(0, int(roiY1 + step * (i + 1)), w, int(roiY1 + step * (i + 1))) for i in range(numStrings)]


def getHandsXRangeFromSkin(frame, roiY1, roiY2, w):
    """Get x-range between hands from skin detection only. Returns (x1, x2) or None."""
    from scripts.hands_region.handsRegionDetector import getSkinMask, findLowSkinXRange
    skinMask = getSkinMask(frame, roiY1, roiY2)
    return findLowSkinXRange(skinMask, roiY1, roiY2)


def detectStringLinesInHandsRegion(frame, gray, numStrings=6, roiY1=None, roiY2=None, returnCrop=False):
    """
    Path 1: Use getProcessingRoi (config-driven, hands_region when auto) for bbox, then run string detection.
    If returnCrop=True, returns (stringLines, x1, y1, x2, y2, croppedGray, croppedEdges).
    """
    from scripts.hands_region.handsRegionDetector import getProcessingRoi
    h, w = gray.shape
    bbox = getProcessingRoi(frame, gray, h, w)
    x1, y1, x2, y2 = bbox
    roiCropped = gray[y1:y2, x1:x2]
    roiEdges = cv2.Canny(roiCropped, 50, 150)
    stringLines = detectStringLinesAngled(roiEdges, numStrings, 0, roiEdges.shape[0], yOffset=y1)
    if stringLines is None:
        if returnCrop:
            return None, x1, y1, x2, y2, roiCropped, roiEdges
        return None
    linesFullFrame = [(line[0] + x1, line[1], line[2] + x1, line[3]) for line in stringLines]
    if returnCrop:
        return linesFullFrame, x1, y1, x2, y2, roiCropped, roiEdges
    return linesFullFrame


def getStringBands(frameHeight, numStrings=6, roiY1=None, roiY2=None, bandHeight=8):
    """Fallback: evenly spaced bands when Hough fails."""
    if roiY1 is None:
        roiY1 = int(frameHeight * 0.25)
    if roiY2 is None:
        roiY2 = int(frameHeight * 0.75)
    roiHeight = roiY2 - roiY1
    step = roiHeight / (numStrings + 1)
    centers = [int(roiY1 + step * (i + 1)) for i in range(numStrings)]
    return np.array(centers)


def computeEdgeIntensity(gray, stringY, bandHeight=10, useGradient=True):
    """Mean edge intensity in a horizontal band around string position."""
    h, w = gray.shape
    y1 = max(0, stringY - bandHeight // 2)
    y2 = min(h, stringY + bandHeight // 2 + 1)
    band = gray[y1:y2, :]
    if useGradient:
        gy = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(np.abs(gy))
    edges = cv2.Canny(band, 50, 150)
    return np.mean(edges)


def shrinkToCenter(xRange, w, usedFrac=0.4, biasLeft=True):
    """Shrink x-range to a segment of the string. biasLeft=True uses left portion (excludes right hand)."""
    if xRange is None:
        return (int(w * 0.2), int(w * 0.45))
    x1, x2 = xRange
    span = x2 - x1
    used = max(20, int(span * usedFrac))
    if biasLeft:
        return (x1, x1 + used)
    margin = (span - used) // 2
    return (x1 + margin, x1 + margin + used)


def findVisibleXRange(gray, stringLines, bandHeight=10, sampleStep=4, minFrac=0.4, minRunFrac=0.08):
    """Infer x-range where all 6 strings have edges (between hands). Returns (x1, x2) or None."""
    h, w = gray.shape
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    scores = []
    for x in range(0, w, sampleStep):
        minInt = float("inf")
        for x1, y1, x2, y2 in stringLines:
            if abs(x2 - x1) < 1e-6:
                sy = (y1 + y2) / 2
            else:
                t = np.clip((x - x1) / (x2 - x1), 0, 1)
                sy = y1 + t * (y2 - y1)
            y = int(sy)
            y1b = max(0, y - bandHeight // 2)
            y2b = min(h, y + bandHeight // 2 + 1)
            x1b = max(0, x - 2)
            x2b = min(w, x + 3)
            band = gy[y1b:y2b, x1b:x2b]
            v = np.mean(np.abs(band))
            minInt = min(minInt, v)
        scores.append((x, minInt))
    if not scores:
        return None
    xs, vals = np.array([s[0] for s in scores]), np.array([s[1] for s in scores])
    peak = np.max(vals)
    if peak < 1e-6:
        return None
    thresh = peak * minFrac
    good = vals >= thresh
    if not np.any(good):
        return None
    runs = []
    inRun = False
    start = 0
    for i in range(len(good)):
        if good[i] and not inRun:
            inRun = True
            start = i
        elif not good[i] and inRun:
            inRun = False
            runs.append((xs[start], xs[i - 1]))
    if inRun:
        runs.append((xs[start], xs[-1]))
    if not runs:
        return None
    best = max(runs, key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2 - w / 2)))
    left, right = best
    if right - left < w * minRunFrac:
        return None
    return (int(left), int(right))


def computeEdgeIntensityForLine(gray, line, bandHeight=10, xRange=None):
    """Mean edge intensity in band around angled line. If xRange given, only uses that x span."""
    h, w = gray.shape
    x1, y1, x2, y2 = line
    if xRange is not None:
        xLo, xHi = xRange
        xLo, xHi = max(0, xLo), min(w, xHi)
        if xHi <= xLo:
            return 0.0
        midX = (xLo + xHi) / 2
    else:
        midX = w / 2
    if abs(x2 - x1) < 1e-6:
        sy = (y1 + y2) / 2
    else:
        t = np.clip((midX - x1) / (x2 - x1), 0, 1)
        sy = y1 + t * (y2 - y1)
    band = gray[max(0, int(sy) - bandHeight // 2):min(h, int(sy) + bandHeight // 2 + 1), :]
    if xRange is not None:
        band = band[:, xLo:xHi]
    if band.size == 0:
        return 0.0
    gy = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.abs(gy))


def detectSuspects(videoPath, stringLines, numStrings=6, dropThreshold=0.18, baselineFrames=25, sampleEvery=1, handsBboxGetter=None, useAlgorithm=False):
    """Find frames where edge intensity drops suggest string play. useAlgorithm=True uses handsRegionDetector."""
    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videoPath}")
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    h, w = first.shape[:2]
    from scripts.hands_region.handsRegionDetector import getRoiVerticalBounds
    roiY1, roiY2 = getRoiVerticalBounds(h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    algoDetector = None
    if useAlgorithm:
        from scripts.hands_region.handsRegionDetector import HandsRegionDetector
        algoDetector = HandsRegionDetector()
    intensities = [[] for _ in range(numStrings)]
    frameIndices = []
    frameIdx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frameIdx % sampleEvery != 0:
            frameIdx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if useAlgorithm and algoDetector:
            bbox = algoDetector.detect(frame, gray, roiY1, roiY2, h, w, stringLines)
            xRange = (bbox[0], bbox[2])
        elif handsBboxGetter:
            bbox = handsBboxGetter(videoPath, frameIdx, h, w)
            xRange = (bbox[0], bbox[2]) if bbox else None
        else:
            xRange = None
        if xRange is None:
            rawRange = findVisibleXRange(gray, stringLines)
            xRange = shrinkToCenter(rawRange, gray.shape[1])
        for i, line in enumerate(stringLines):
            if i < numStrings:
                intensities[i].append(computeEdgeIntensityForLine(gray, line, xRange=xRange))
        frameIndices.append(frameIdx)
        frameIdx += 1
    cap.release()
    intensities = np.array(intensities)
    suspects = []
    for t in range(baselineFrames, len(frameIndices)):
        frameIdx = frameIndices[t]
        suspected = []
        for s in range(numStrings):
            baseline = np.median(intensities[s, max(0, t - baselineFrames):t])
            current = intensities[s, t]
            if baseline > 1e-6 and current < baseline * (1 - dropThreshold):
                suspected.append(s + 1)
        if suspected:
            suspects.append((frameIdx, suspected))
    return suspects


def runTracking(videoPath, numStrings=6, sampleEvery=2, maxFrames=500, outputPath=None):
    """Process video and track edge intensity per string over time."""
    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videoPath}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    from scripts.hands_region.handsRegionDetector import getRoiVerticalBounds
    roiY1, roiY2 = getRoiVerticalBounds(gray.shape[0])
    stringYs = detectStringLines(edges, numStrings, roiY1, roiY2)
    if stringYs is None:
        stringYs = getStringBands(gray.shape[0], numStrings, roiY1, roiY2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frameIdx = 0
    results = [[] for _ in range(numStrings)]
    frameCount = 0
    while frameCount < maxFrames:
        ret, frame = cap.read()
        if not ret:
            break
        if frameIdx % sampleEvery != 0:
            frameIdx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for i, sy in enumerate(stringYs):
            intensity = computeEdgeIntensity(gray, int(sy), useGradient=True)
            results[i].append(intensity)
        frameCount += 1
        frameIdx += 1
    cap.release()
    results = np.array(results)
    if outputPath:
        np.save(outputPath, {"intensities": results, "stringYs": stringYs, "fps": fps})
    return results, stringYs, fps


def visualizeResults(results, stringYs, fps, outputPath=None, show=True):
    """Plot edge intensity over time per string for hypothesis validation."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    numStrings = results.shape[0]
    t = np.arange(results.shape[1]) / (fps / 2)
    fig, ax = plt.subplots(numStrings, 1, figsize=(12, 2 * numStrings), sharex=True)
    if numStrings == 1:
        ax = [ax]
    colors = plt.cm.viridis(np.linspace(0, 1, numStrings))
    for i in range(numStrings):
        ax[i].plot(t, results[i], color=colors[i], linewidth=0.8)
        ax[i].set_ylabel(f"Str {i+1}")
        ax[i].set_ylim(0, None)
        ax[i].grid(True, alpha=0.3)
    ax[-1].set_xlabel("Time (s)")
    fig.suptitle("Edge intensity per string (drops = possible play)")
    plt.tight_layout()
    if outputPath:
        plt.savefig(outputPath, dpi=150)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Track string edges to test play-detection hypothesis. "
        "Drops in edge intensity indicate possible string activation (motion blur)."
    )
    parser.add_argument("video", type=str, help="Path to video in data folder")
    parser.add_argument("-n", "--strings", type=int, default=6, help="Number of strings (default 6)")
    parser.add_argument("-s", "--sample", type=int, default=2, help="Sample every N frames")
    parser.add_argument("-m", "--max-frames", type=int, default=500, help="Max frames to process")
    parser.add_argument("-o", "--output", type=str, help="Output base path for .npy and .png")
    parser.add_argument("--no-show", action="store_true", help="Save plot only, do not display")
    args = parser.parse_args()
    videoPath = Path(args.video)
    if not videoPath.is_absolute():
        videoPath = PROJECT_ROOT / "data" / videoPath.name
    if not videoPath.exists():
        videoPath = Path(args.video)
    if not videoPath.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    base = Path(args.output) if args.output else videoPath.with_suffix("")
    results, stringYs, fps = runTracking(
        videoPath, numStrings=args.strings,
        sampleEvery=args.sample, maxFrames=args.max_frames,
        outputPath=str(base) + "_edges.npy" if args.output else None
    )
    print(f"Processed {results.shape[1]} frames, {args.strings} strings")
    print("String Y positions:", stringYs.tolist())
    visPath = str(base) + "_edges.png" if args.output else None
    visualizeResults(results, stringYs, fps, visPath, show=not args.no_show)


if __name__ == "__main__":
    main()
