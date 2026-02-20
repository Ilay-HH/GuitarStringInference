"""
Automatic detection of the region between the two hands (used region) per frame.
Combines skin detection (hands = high skin) with string visibility (between hands = strings visible).
No per-video calibration required. Reads params from config/hands_region.json.
Runnable standalone (uses _string_utils) or with string_tracking.
"""
import cv2
import json
import numpy as np
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if (_PROJECT_ROOT / "config").exists():
    PROJECT_ROOT = _PROJECT_ROOT
else:
    PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.string_tracking.stringEdgeTracker import detectStringLinesAngled, findVisibleXRange
except ImportError:
    from ._string_utils import detectStringLinesAngled, findVisibleXRange

CONFIG_PATH = PROJECT_ROOT / "config" / "hands_region.json"


def loadConfig():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def getSkinMask(frame, roiY1, roiY2):
    """HSV-based skin mask in ROI. Returns binary mask (0-255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def findLowSkinXRange(skinMask, roiY1, roiY2, sampleStep=4, skinFracThresh=None, minRunFrac=None):
    """
    Find x-range where skin density is low (between hands).
    For each column, compute fraction of skin pixels in ROI. Low = between hands.
    """
    cfg = loadConfig()
    skinFracThresh = skinFracThresh if skinFracThresh is not None else cfg.get("skinFracThresh", 0.30)
    minRunFrac = minRunFrac if minRunFrac is not None else cfg.get("minRunFrac", 0.08)
    h, w = skinMask.shape
    roi = skinMask[roiY1:roiY2, :]
    scores = []
    for x in range(0, w, sampleStep):
        col = roi[:, max(0, x - 4):min(w, x + 5)]
        frac = np.sum(col > 0) / (col.size + 1e-6)
        scores.append((x, frac))
    if not scores:
        return None
    xs = np.array([s[0] for s in scores])
    vals = np.array([s[1] for s in scores])
    low = vals <= skinFracThresh
    if not np.any(low):
        return None
    runs = []
    inRun, start = False, 0
    for i in range(len(low)):
        if low[i] and not inRun:
            inRun, start = True, i
        elif not low[i] and inRun:
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


def detectHandsRegion(frame, gray, stringLines, roiY1, roiY2, h, w,
                     leftStretchFrac=None, rightStretchFrac=None):
    """
    Detect used region (between hands) from frame. Returns (x1, y1, x2, y2) or None.
    """
    cfg = loadConfig()
    leftStretchFrac = leftStretchFrac if leftStretchFrac is not None else cfg.get("leftStretchFrac", 0.30)
    rightStretchFrac = rightStretchFrac if rightStretchFrac is not None else cfg.get("rightStretchFrac", 0.10)
    skinMask = getSkinMask(frame, roiY1, roiY2)
    xRangeSkin = findLowSkinXRange(skinMask, roiY1, roiY2)
    xRangeStrings = findVisibleXRange(gray, stringLines)

    candidates = []
    if xRangeSkin is not None:
        candidates.append((xRangeSkin[0], xRangeSkin[1]))
    if xRangeStrings is not None:
        candidates.append((xRangeStrings[0], xRangeStrings[1]))

    if not candidates:
        return None

    if len(candidates) == 2:
        interLeft = max(candidates[0][0], candidates[1][0])
        interRight = min(candidates[0][1], candidates[1][1])
        if interRight - interLeft >= w * 0.08:
            x1, x2 = interLeft, interRight
        else:
            best = max(candidates, key=lambda c: c[1] - c[0])
            x1, x2 = best[0], best[1]
    else:
        best = max(candidates, key=lambda c: c[1] - c[0])
        x1, x2 = best[0], best[1]

    x1 = max(0, int(x1 - w * leftStretchFrac))
    x2 = min(w, int(x2 + w * rightStretchFrac))
    return (x1, roiY1, x2, roiY2)


class HandsRegionDetector:
    """Stateful detector with temporal smoothing."""

    def __init__(self, alpha=None):
        cfg = loadConfig()
        self.alpha = alpha if alpha is not None else cfg.get("smoothingAlpha", 0.70)
        self.prevBbox = None

    def detect(self, frame, gray, stringLines, roiY1, roiY2, h, w):
        bbox = detectHandsRegion(frame, gray, stringLines, roiY1, roiY2, h, w)
        if bbox is None:
            bbox = (int(w * 0.2), roiY1, int(w * 0.8), roiY2)
        if self.prevBbox is not None:
            smoothed = []
            for j in range(4):
                v = int(self.alpha * bbox[j] + (1 - self.alpha) * self.prevBbox[j])
                smoothed.append(v)
            bbox = tuple(smoothed)
        self.prevBbox = bbox
        return bbox


def detectHandsRegionForFrame(frame, gray, stringLines, roiY1, roiY2, h, w, detector=None):
    """Convenience: detect hands region, return bbox. Pass detector for temporal smoothing."""
    if detector is not None:
        return detector.detect(frame, gray, stringLines, roiY1, roiY2, h, w)
    bbox = detectHandsRegion(frame, gray, stringLines, roiY1, roiY2, h, w)
    if bbox is not None:
        return bbox
    return (int(w * 0.2), roiY1, int(w * 0.8), roiY2)
