"""
Automatic detection of the region between the two hands (used region) per frame.
Uses skin detection: hands = high skin density, between hands = low skin.
No per-video calibration required. Reads params from config/hands_region.json.
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


def _yAtX(line, x):
    x1, y1, x2, y2 = line
    if abs(x2 - x1) < 1e-6:
        return (y1 + y2) / 2
    t = np.clip((x - x1) / (x2 - x1), 0, 1)
    return y1 + t * (y2 - y1)


def _refineHeightFromStrings(bbox, stringLines, roiY1, roiY2):
    """Refine y1,y2 from string span at mid-x. Returns (x1, y1, x2, y2)."""
    if not stringLines or len(stringLines) < 2:
        return bbox
    cfg = loadConfig()
    padding = cfg.get("stringHeightPadding", 0.15)
    x1, y1, x2, y2 = bbox
    midX = (x1 + x2) / 2
    yTop = _yAtX(stringLines[0], midX)
    yBot = _yAtX(stringLines[-1], midX)
    pad = max(1, (yBot - yTop) * padding)
    y1 = max(roiY1, int(yTop - pad))
    y2 = min(roiY2, int(yBot + pad))
    return (x1, y1, x2, y2)


def detectHandsRegionSkinOnly(frame, roiY1, roiY2, h, w):
    """Bbox from skin detection: low skin density = between hands."""
    cfg = loadConfig()
    leftStretchFrac = cfg.get("leftStretchFrac", 0.30)
    rightStretchFrac = cfg.get("rightStretchFrac", 0.10)
    skinMask = getSkinMask(frame, roiY1, roiY2)
    xRange = findLowSkinXRange(skinMask, roiY1, roiY2)
    if xRange is None:
        return (int(w * 0.2), roiY1, int(w * 0.8), roiY2)
    x1, x2 = xRange
    x1 = max(0, int(x1 - w * leftStretchFrac))
    x2 = min(w, int(x2 + w * rightStretchFrac))
    return (x1, roiY1, x2, roiY2)


def getRoiVerticalBounds(h, cfg=None):
    """Returns (roiY1, roiY2) from config for skin/hands search area."""
    cfg = cfg or loadConfig()
    roiHeightCfg = cfg.get("roi_height", "auto")
    heightFixed = cfg.get("roi_height_fixed", [0.2, 0.8])
    if roiHeightCfg == "auto":
        return int(h * heightFixed[0]), int(h * heightFixed[1])
    fracs = roiHeightCfg if isinstance(roiHeightCfg, (list, tuple)) else heightFixed
    return int(h * fracs[0]), int(h * fracs[1])


def getProcessingRoi(frame, gray, h, w, stringLines=None):
    """
    Resolve ROI from config. Returns (x1, y1, x2, y2).
    roi_height/roi_width: "auto" uses skin-based hands_region; [minFrac, maxFrac] uses fixed fractions.
    stringLines: optional, when provided refines height to actual string span.
    """
    cfg = loadConfig()
    roiHeightCfg = cfg.get("roi_height", "auto")
    roiWidthCfg = cfg.get("roi_width", "auto")
    heightFixed = cfg.get("roi_height_fixed", [0.2, 0.8])
    widthFixed = cfg.get("roi_width_fixed", [0.2, 0.8])

    if roiHeightCfg == "auto":
        roiY1 = int(h * heightFixed[0])
        roiY2 = int(h * heightFixed[1])
    else:
        fracs = roiHeightCfg if isinstance(roiHeightCfg, (list, tuple)) else heightFixed
        roiY1 = int(h * fracs[0])
        roiY2 = int(h * fracs[1])

    if roiWidthCfg == "auto":
        bbox = detectHandsRegionSkinOnly(frame, roiY1, roiY2, h, w)
        if bbox is None:
            bbox = (int(w * widthFixed[0]), roiY1, int(w * widthFixed[1]), roiY2)
        bbox = _refineHeightFromStrings(bbox, stringLines, roiY1, roiY2)
        return bbox

    fracs = roiWidthCfg if isinstance(roiWidthCfg, (list, tuple)) else widthFixed
    x1 = int(w * fracs[0])
    x2 = int(w * fracs[1])
    bbox = (x1, roiY1, x2, roiY2)
    bbox = _refineHeightFromStrings(bbox, stringLines, roiY1, roiY2)
    return bbox


class HandsRegionDetector:
    """Stateful detector with temporal smoothing."""

    def __init__(self, alpha=None):
        cfg = loadConfig()
        self.alpha = alpha if alpha is not None else cfg.get("smoothingAlpha", 0.70)
        self.prevBbox = None

    def detect(self, frame, gray, roiY1, roiY2, h, w, stringLines=None):
        bbox = detectHandsRegionSkinOnly(frame, roiY1, roiY2, h, w)
        if bbox is None:
            bbox = (int(w * 0.2), roiY1, int(w * 0.8), roiY2)
        bbox = _refineHeightFromStrings(bbox, stringLines, roiY1, roiY2)
        if self.prevBbox is not None:
            smoothed = []
            for j in range(4):
                v = int(self.alpha * bbox[j] + (1 - self.alpha) * self.prevBbox[j])
                smoothed.append(v)
            bbox = tuple(smoothed)
        self.prevBbox = bbox
        return bbox


def detectHandsRegionForFrame(frame, gray, roiY1, roiY2, h, w, stringLines=None, detector=None):
    """Convenience: detect hands region, return bbox. Pass detector for temporal smoothing."""
    if detector is not None:
        return detector.detect(frame, gray, roiY1, roiY2, h, w, stringLines)
    bbox = detectHandsRegionSkinOnly(frame, roiY1, roiY2, h, w)
    if bbox is not None:
        bbox = _refineHeightFromStrings(bbox, stringLines, roiY1, roiY2)
        return bbox
    return (int(w * 0.2), roiY1, int(w * 0.8), roiY2)
