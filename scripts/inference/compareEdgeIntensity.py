"""
Compare edge intensity per string: with algorithm (filtered to relevant area) vs without (full width).
Outputs a single graph with both traces per string.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.string_tracking.stringEdgeTracker import detectStringLinesAngled, detectStringLinesInHandsRegion, computeEdgeIntensityForLine, fallbackStringLines
from scripts.hands_region.handsRegionDetector import HandsRegionDetector, getRoiVerticalBounds


def computeIntensitiesOverVideo(videoPath, stringLines, numStrings=6, useAlgorithm=False, sampleEvery=1):
    """Returns (frameIndices, intensities) where intensities shape is (numStrings, numFrames)."""
    cap = cv2.VideoCapture(str(videoPath))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {videoPath}")
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")
    h, w = first.shape[:2]
    roiY1, roiY2 = getRoiVerticalBounds(h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    algoDetector = HandsRegionDetector() if useAlgorithm else None
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
        else:
            xRange = None

        for i, line in enumerate(stringLines):
            if i < numStrings:
                intensities[i].append(computeEdgeIntensityForLine(gray, line, xRange=xRange))
        frameIndices.append(frameIdx)
        frameIdx += 1

    cap.release()
    return np.array(frameIndices), np.array(intensities)


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
    ret, first = cap.read()
    cap.release()
    if not ret:
        print("Cannot read video")
        return

    h, w = first.shape[:2]
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    result = detectStringLinesInHandsRegion(first, gray, 6, returnCrop=True)
    stringLines = result[0]
    if stringLines is None:
        handsY1, roiEdges = result[2], result[6]
        stringLines = detectStringLinesAngled(roiEdges, 6, 0, roiEdges.shape[0], yOffset=handsY1)
        if stringLines is not None:
            stringLines = [(l[0] + result[1], l[1], l[2] + result[1], l[3]) for l in stringLines]
    if stringLines is None:
        ry1, ry2 = getRoiVerticalBounds(h)
        stringLines = fallbackStringLines(h, w, 6, ry1, ry2)

    fps = cv2.VideoCapture(str(videoPath)).get(cv2.CAP_PROP_FPS) or 30
    cv2.VideoCapture(str(videoPath)).release()

    print("Computing intensities WITH algorithm (filtered)...")
    framesAlgo, intAlgo = computeIntensitiesOverVideo(videoPath, stringLines, useAlgorithm=True)
    print("Computing intensities WITHOUT algorithm (full width)...")
    framesNoAlgo, intNoAlgo = computeIntensitiesOverVideo(videoPath, stringLines, useAlgorithm=False)

    nFrames = min(len(framesAlgo), len(framesNoAlgo))
    t = np.arange(nFrames) / fps

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numStrings = 6
    fig, axes = plt.subplots(numStrings, 1, figsize=(14, 3 * numStrings), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 0.8, numStrings))

    for i in range(numStrings):
        ax = axes[i]
        ax.plot(t[:nFrames], intAlgo[i, :nFrames], color=colors[i], linewidth=1.0, label="Filtered (algorithm)")
        ax.plot(t[:nFrames], intNoAlgo[i, :nFrames], color=colors[i], linewidth=0.7, linestyle="--", alpha=0.7, label="Full width")
        ax.set_ylabel(f"Str {i+1}")
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Edge intensity per string: with algorithm (filtered) vs without (full width)")
    plt.tight_layout()

    outPath = PROJECT_ROOT / "output" / "compare_edge_intensity.png"
    outPath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outPath, dpi=150)
    print(f"Saved comparison graph to {outPath}")


if __name__ == "__main__":
    main()
