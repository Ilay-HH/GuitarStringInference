"""
Manual calibration tool for the bounding box between the two hands.
Run on example frames, save calibrations to output/calibration.json for algorithm tuning.
Runnable standalone.
"""
import cv2
import json
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
PROJECT_ROOT = _PROJECT_ROOT if (_PROJECT_ROOT / "config").exists() else _THIS_DIR.parent
CALIBRATION_PATH = PROJECT_ROOT / "output" / "calibration.json"


def loadCalibration():
    """Load calibration entries from JSON."""
    if not CALIBRATION_PATH.exists():
        return {"entries": []}
    with open(CALIBRATION_PATH, "r") as f:
        return json.load(f)


def saveCalibration(data):
    """Save calibration to JSON."""
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(data, f, indent=2)


def runCalibrationSequence(videoPath, numFrames=10):
    """
    Run 10-frame calibration sequence. Frames evenly spaced. For each: user selects ROI, Enter to confirm and advance.
    Saves all to calibration.json. getHandsBbox then interpolates between them.
    """
    videoPath = Path(videoPath)
    if not videoPath.is_absolute():
        videoPath = PROJECT_ROOT / "data" / videoPath.name
    if not videoPath.exists():
        raise FileNotFoundError(f"Video not found: {videoPath}")

    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videoPath}")
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoKey = str(videoPath.resolve())

    frameIndices = [int(i * (totalFrames - 1) / (numFrames - 1)) for i in range(numFrames)] if numFrames > 1 else [0]
    entries = []

    cv2.namedWindow("Hands calibration", cv2.WINDOW_NORMAL)
    print(f"Calibration sequence: {numFrames} frames. Select ROI for each, press Enter to confirm and advance.")
    for i, frameIdx in enumerate(frameIndices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
        ret, frame = cap.read()
        if not ret:
            print(f"Cannot read frame {frameIdx}")
            continue
        disp = frame.copy()
        cv2.putText(disp, f"Frame {i+1}/{numFrames} (idx {frameIdx}) | Select ROI, Enter to confirm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Hands calibration", disp)
        cv2.waitKey(1)
        roi = cv2.selectROI("Hands calibration - select region between hands", frame, fromCenter=False)
        if roi[2] <= 0 or roi[3] <= 0:
            print(f"Frame {i+1}: invalid selection, skipping")
            continue
        x, y, w, h = roi
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        entries.append({"video": videoKey, "frame": frameIdx, "bbox": bbox})
        print(f"Frame {i+1}/{numFrames}: bbox {bbox}")

    cap.release()
    cv2.destroyAllWindows()

    if entries:
        data = loadCalibration()
        existing = [e for e in data["entries"] if e["video"] != videoKey]
        data["entries"] = existing + entries
        saveCalibration(data)
        print(f"Saved {len(entries)} calibrations. Algorithm will interpolate between them.")
    else:
        print("No valid calibrations saved.")


def runCalibrator(videoPath, frameIdx=0, interactive=True):
    """
    Calibrate bounding box between hands. Interactive: Left/Right = frame step, S = select ROI, Q = quit.
    Non-interactive: use -f frame and select ROI directly.
    """
    videoPath = Path(videoPath)
    if not videoPath.is_absolute():
        videoPath = PROJECT_ROOT / "data" / videoPath.name
    if not videoPath.exists():
        raise FileNotFoundError(f"Video not found: {videoPath}")

    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videoPath}")
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoKey = str(videoPath.resolve())

    def readFrame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        return cap.read()

    if interactive:
        idx = max(0, min(frameIdx, totalFrames - 1))
        cv2.namedWindow("Hands calibration", cv2.WINDOW_NORMAL)
        print("A/D or Left/Right = frame step | S = select ROI on current frame | Q = quit")
        while True:
            ret, frame = readFrame(idx)
            if not ret:
                break
            disp = frame.copy()
            cv2.putText(disp, f"Frame {idx}/{totalFrames} | S=select Q=quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Hands calibration", disp)
            k = cv2.waitKeyEx(0)
            if k in (ord("q"), 27):
                break
            elif k in (ord("s"), ord(" "), 13):
                roi = cv2.selectROI("Hands calibration - select region between hands", frame, fromCenter=False)
                if roi[2] > 0 and roi[3] > 0:
                    x, y, w, h = roi
                    bbox = [int(x), int(y), int(x + w), int(y + h)]
                    data = loadCalibration()
                    existing = [e for e in data["entries"] if e["video"] != videoKey or e["frame"] != idx]
                    existing.append({"video": videoKey, "frame": idx, "bbox": bbox})
                    data["entries"] = existing
                    saveCalibration(data)
                    print(f"Saved: frame {idx} -> bbox {bbox}")
                continue
            elif k in (ord("a"), 2, 2424832):
                idx = max(0, idx - 1)
            elif k in (ord("d"), 3, 2555904):
                idx = min(totalFrames - 1, idx + 1)
        cv2.destroyAllWindows()
    else:
        ret, frame = readFrame(frameIdx)
        cap.release()
        if not ret:
            raise RuntimeError("Cannot read frame")
        print("Select the bounding box BETWEEN the two hands (drag rectangle).")
        roi = cv2.selectROI("Hands calibration - select region between hands", frame, fromCenter=False)
        cv2.destroyAllWindows()
        if roi[2] <= 0 or roi[3] <= 0:
            print("Invalid selection (zero size), not saved.")
            return
        x, y, w, h = roi
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        data = loadCalibration()
        existing = [e for e in data["entries"] if e["video"] != videoKey or e["frame"] != frameIdx]
        existing.append({"video": videoKey, "frame": frameIdx, "bbox": bbox})
        data["entries"] = existing
        saveCalibration(data)
        print(f"Saved: {videoKey} frame {frameIdx} -> bbox {bbox}")
    cap.release()


def getHandsBbox(videoPath, frameIdx, h, w, padding=0):
    """
    Get hands bbox from calibration. Returns (x1, y1, x2, y2) or None.
    Interpolates between calibrated frames; applies optional inward padding.
    """
    videoKey = str(Path(videoPath).resolve())
    data = loadCalibration()
    entries = sorted([e for e in data["entries"] if e["video"] == videoKey], key=lambda e: e["frame"])
    if not entries:
        return None
    if len(entries) == 1:
        bbox = list(entries[0]["bbox"])
    elif frameIdx <= entries[0]["frame"]:
        bbox = list(entries[0]["bbox"])
    elif frameIdx >= entries[-1]["frame"]:
        bbox = list(entries[-1]["bbox"])
    else:
        for i in range(len(entries) - 1):
            if entries[i]["frame"] <= frameIdx <= entries[i + 1]["frame"]:
                t = (frameIdx - entries[i]["frame"]) / (entries[i + 1]["frame"] - entries[i]["frame"])
                a, b = entries[i]["bbox"], entries[i + 1]["bbox"]
                bbox = [int(a[j] + t * (b[j] - a[j])) for j in range(4)]
                break
        else:
            bbox = list(entries[-1]["bbox"])
    if padding > 0:
        bbox[0] = min(bbox[0] + padding, bbox[2] - 10)
        bbox[1] = min(bbox[1] + padding, bbox[3] - 10)
        bbox[2] = max(bbox[2] - padding, bbox[0] + 10)
        bbox[3] = max(bbox[3] - padding, bbox[1] + 10)
    return tuple(bbox)


def main():
    parser = argparse.ArgumentParser(
        description="Manually calibrate bounding box between hands on example frames."
    )
    parser.add_argument("video", type=str, help="Path to video")
    parser.add_argument("-f", "--frame", type=int, default=0, help="Start frame (default 0)")
    parser.add_argument("-n", "--num-frames", type=int, default=10, help="Number of frames in sequence mode (default 10)")
    parser.add_argument("--sequence", action="store_true", help="10-frame sequence: select ROI for each, Enter to advance")
    parser.add_argument("--no-interactive", action="store_true", help="Skip frame browser, select ROI directly")
    args = parser.parse_args()
    if args.sequence:
        runCalibrationSequence(args.video, args.num_frames)
    else:
        runCalibrator(args.video, args.frame, interactive=not args.no_interactive)


if __name__ == "__main__":
    main()
