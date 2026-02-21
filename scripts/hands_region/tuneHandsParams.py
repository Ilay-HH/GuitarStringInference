"""
Tune hands region algorithm parameters by comparing to user-annotated calibration.
Runs 10 parameter variations and reports IoU + boundary errors vs ground truth.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
PROJECT_ROOT = _PROJECT_ROOT if (_PROJECT_ROOT / "config").exists() else _THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hands_region.handsRegionDetector import getSkinMask, findLowSkinXRange, getRoiVerticalBounds


def loadCalibration():
    path = PROJECT_ROOT / "output" / "calibration.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("entries", [])


def getGtBbox(entries, frameIdx, videoKey):
    entries = sorted([e for e in entries if e["video"] == videoKey], key=lambda e: e["frame"])
    if not entries:
        return None
    if frameIdx <= entries[0]["frame"]:
        return tuple(entries[0]["bbox"])
    if frameIdx >= entries[-1]["frame"]:
        return tuple(entries[-1]["bbox"])
    for i in range(len(entries) - 1):
        if entries[i]["frame"] <= frameIdx <= entries[i + 1]["frame"]:
            t = (frameIdx - entries[i]["frame"]) / (entries[i + 1]["frame"] - entries[i]["frame"])
            a, b = entries[i]["bbox"], entries[i + 1]["bbox"]
            return tuple(int(a[j] + t * (b[j] - a[j])) for j in range(4))
    return tuple(entries[-1]["bbox"])


def detectWithParams(frame, gray, roiY1, roiY2, h, w, params):
    skinFracThresh = params.get("skinFracThresh", 0.25)
    minRunFrac = params.get("minRunFrac", 0.08)
    rightStretchFrac = params.get("rightStretchFrac", 0)
    leftStretchFrac = params.get("leftStretchFrac", 0)

    skinMask = getSkinMask(frame, roiY1, roiY2)
    xRange = findLowSkinXRange(skinMask, roiY1, roiY2, skinFracThresh=skinFracThresh, minRunFrac=minRunFrac)
    if xRange is None:
        return None
    x1, x2 = xRange
    x1 = max(0, int(x1 - w * leftStretchFrac))
    x2 = min(w, int(x2 + w * rightStretchFrac))
    return (x1, roiY1, x2, roiY2)


def iou(bboxA, bboxB):
    ax1, ay1, ax2, ay2 = bboxA
    bx1, by1, bx2, by2 = bboxB
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter
    return inter / (union + 1e-6)


def boundaryErrors(pred, gt):
    """Returns (leftErr, rightErr) - positive means pred is to the right of gt (too tight on left/right)."""
    leftErr = pred[0] - gt[0]
    rightErr = pred[2] - gt[2]
    return leftErr, rightErr


def main():
    dataDir = PROJECT_ROOT / "data"
    videoPath = dataDir / "The most beautiful melody line.mp4"
    if not videoPath.exists():
        videos = list(dataDir.glob("*.mp4"))
        videoPath = videos[0] if videos else None
    if not videoPath or not videoPath.exists():
        print("No video found")
        return

    entries = loadCalibration()
    videoKey = str(videoPath.resolve())
    if not any(e["video"] == videoKey for e in entries):
        print("No calibration for this video")
        return

    cap = cv2.VideoCapture(str(videoPath))
    ret, first = cap.read()
    if not ret:
        print("Cannot read video")
        return
    h, w = first.shape[:2]
    roiY1, roiY2 = getRoiVerticalBounds(h)
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    calibFrames = sorted(set(e["frame"] for e in entries if e["video"] == videoKey))
    evalFrames = calibFrames + [int((a + b) / 2) for a, b in zip(calibFrames[:-1], calibFrames[1:])]
    evalFrames = sorted(set(evalFrames))

    paramSets = [
        {"name": "baseline", "skinFracThresh": 0.25, "rightStretchFrac": 0},
        {"name": "skin0.30 right+10%", "skinFracThresh": 0.30, "rightStretchFrac": 0.10},
        {"name": "skin0.30 right+15%", "skinFracThresh": 0.30, "rightStretchFrac": 0.15},
        {"name": "skin0.30 right+20%", "skinFracThresh": 0.30, "rightStretchFrac": 0.20},
        {"name": "skin0.30 left-20% right+10%", "skinFracThresh": 0.30, "leftStretchFrac": 0.20, "rightStretchFrac": 0.10},
        {"name": "skin0.30 left-25% right+15%", "skinFracThresh": 0.30, "leftStretchFrac": 0.25, "rightStretchFrac": 0.15},
        {"name": "skin0.35 left-20% right+15%", "skinFracThresh": 0.35, "leftStretchFrac": 0.20, "rightStretchFrac": 0.15},
        {"name": "skin0.35 left-25% right+20%", "skinFracThresh": 0.35, "leftStretchFrac": 0.25, "rightStretchFrac": 0.20},
        {"name": "skin0.30 left-30% right+10%", "skinFracThresh": 0.30, "leftStretchFrac": 0.30, "rightStretchFrac": 0.10},
        {"name": "skin0.35 left-30% right+15%", "skinFracThresh": 0.35, "leftStretchFrac": 0.30, "rightStretchFrac": 0.15},
    ]

    results = []
    for params in paramSets:
        ious = []
        rightErrs = []
        leftErrs = []
        for frameIdx in evalFrames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gt = getGtBbox(entries, frameIdx, videoKey)
            if gt is None:
                continue
            pred = detectWithParams(frame, gray, roiY1, roiY2, h, w, params)
            if pred is None:
                pred = (int(w * 0.2), roiY1, int(w * 0.8), roiY2)
            ious.append(iou(pred, gt))
            le, re = boundaryErrors(pred, gt)
            leftErrs.append(le)
            rightErrs.append(re)

        avgIou = np.mean(ious) if ious else 0
        avgRightErr = np.mean(rightErrs) if rightErrs else 0
        avgLeftErr = np.mean(leftErrs) if leftErrs else 0
        results.append({
            "name": params.get("name", str(params)),
            "iou": avgIou,
            "rightErr": avgRightErr,
            "leftErr": avgLeftErr,
            "params": params
        })

    cap.release()

    results.sort(key=lambda r: (-r["iou"], r["rightErr"]))
    print("Results (sorted by IoU, then rightErr):")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:30} IoU={r['iou']:.3f}  rightErr={r['rightErr']:+.0f}px  leftErr={r['leftErr']:+.0f}px")
    print("-" * 70)
    best = results[0]
    print(f"Best: {best['name']} with IoU={best['iou']:.3f}")
    print(f"Params: {best['params']}")


if __name__ == "__main__":
    main()
