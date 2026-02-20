"""
Minimal string detection utils for standalone hands_region runs.
When string_tracking is available, handsRegionDetector uses that instead.
"""
import cv2
import numpy as np


def detectStringLinesAngled(edgeImg, numStrings=6, roiY1=0, roiY2=None, maxAngleDeg=35, yOffset=None):
    """Find string lines (possibly angled) via Hough."""
    if roiY2 is None:
        roiY2 = edgeImg.shape[0]
    if yOffset is None:
        yOffset = roiY1
    h, w = edgeImg.shape
    roi = edgeImg[roiY1:roiY2, :]
    lines = cv2.HoughLinesP(
        roi, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=roi.shape[1] // 4, maxLineGap=25
    )
    if lines is None:
        return None
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        y1, y2 = y1 + yOffset, y2 + yOffset
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= maxAngleDeg or abs(abs(angle) - 180) <= maxAngleDeg:
            candidates.append((x1, y1, x2, y2))
    if len(candidates) < numStrings:
        return None
    midX = w / 2
    def heightAtMid(x1, y1, x2, y2):
        if abs(x2 - x1) < 1e-6:
            return (y1 + y2) / 2
        t = (midX - x1) / (x2 - x1)
        t = np.clip(t, 0, 1)
        return y1 + t * (y2 - y1)
    candidates.sort(key=lambda L: heightAtMid(L[0], L[1], L[2], L[3]))
    n = len(candidates)
    if n < numStrings:
        return None
    indices = [int(i * (n - 1) / (numStrings - 1)) for i in range(numStrings)] if numStrings > 1 else [0]
    return [candidates[i] for i in indices]


def fallbackStringLines(h, w, numStrings=6, roiY1=None, roiY2=None):
    """Horizontal lines when angled detection fails."""
    if roiY1 is None:
        roiY1 = int(h * 0.25)
    if roiY2 is None:
        roiY2 = int(h * 0.75)
    step = (roiY2 - roiY1) / (numStrings + 1)
    return [(0, int(roiY1 + step * (i + 1)), w, int(roiY1 + step * (i + 1))) for i in range(numStrings)]


def findVisibleXRange(gray, stringLines, bandHeight=10, sampleStep=4, minFrac=0.4, minRunFrac=0.08):
    """Infer x-range where all strings have edges. Returns (x1, x2) or None."""
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
    inRun, start = False, 0
    for i in range(len(good)):
        if good[i] and not inRun:
            inRun, start = True, i
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
