# Guitar Tab Inference - Process Documentation

Work-in-progress documentation of our approach to infer which guitar strings are played from video using edge detection and motion blur.

---

## Hypothesis

When a guitar string is played, it vibrates at high speed. This motion causes **blur** in the video frame: the string "smears" across multiple pixels. A stationary string has sharp edges; a vibrating string has softer, spread-out edges. Therefore: **a drop in edge intensity** around a string indicates it was likely played.

---

## Pipeline Overview

1. **String Tracking** - Detect the 6 string positions and track edge intensity per string
2. **Hands Region** - Find the relevant zone between the two hands (filter out fretting/strumming hands)
3. **Suspect Detection** - Scan video for frames where edge intensity drops (within the relevant zone)
4. **Annotation** - Human validates/corrects suspects and tags missed frames
5. **Evaluation** (future) - Compare algorithm output to annotations

---

## Part 1: String Tracking

### Calibration: Finding the Strings

We use the first frame to locate the 6 strings. The process:

#### Step 1: Original Frame

![Original frame](images/string_tracking/01_original.png)

#### Step 2: Region of Interest

Define the fretboard area (20%-80% of frame height) to avoid hands, headstock, etc. We crop to this region before edge detection.

![ROI marked on frame](images/string_tracking/02_roi_marked.png)

#### Step 3: Crop and Edge Detection

Crop the grayscale to the ROI, then apply Canny edge detection (thresholds 50, 150). Only the fretboard region is processed - this reduces noise from hands, face, and background.

![Cropped grayscale](images/string_tracking/03_roi_grayscale.png)

![Canny edges on ROI](images/string_tracking/04_canny_edges.png)

#### Step 4: Hough Line Transform

Run `cv2.HoughLinesP` on the cropped edges to find line segments. We keep lines that are roughly horizontal (within 35 degrees) since guitar strings run across the frame.

![Hough lines on edges](images/string_tracking/05_hough_lines.png)

#### Step 5: Select 6 String Lines

- Sort candidate lines by vertical position at frame center
- Pick 6 evenly spaced lines (top = string 1, bottom = string 6)
- If Hough fails, fallback to 6 evenly spaced horizontal bands in the ROI
- Convert line coordinates back to full-frame for overlay

![Detected string lines](images/string_tracking/06_string_lines.png)

#### Iteration: Actual Cropping

**Originally** we applied Canny to the full frame and only used the ROI for Hough. The edge image was full-size. **Changed** to crop the grayscale to the ROI before Canny - edge detection now runs only on the fretboard, reducing noise from hands and background.

#### Iteration: Angled Strings

**Originally** we assumed strings were strictly horizontal. **Changed** because the camera angle or fretboard perspective often makes strings appear at a slight angle. We now use `detectStringLinesAngled` which allows lines up to 35 degrees from horizontal and returns full line segments `(x1,y1,x2,y2)` instead of just y-positions.

---

### Edge-to-String Assignment

We need to assign each edge pixel to a string for visualization and intensity tracking.

#### Iteration: Distance vs Band Boundaries

**Originally** we used "nearest string line" by perpendicular distance. **Problem**: with angled strings and occlusion (e.g. hand), string 1 could "steal" edges from string 2 - each string took some from the one below. **Changed** to band-based assignment: at each x, compute the y-position of each string line; boundaries are midpoints between adjacent strings; each edge falls into exactly one band. Non-overlapping, respects angle.

![Band boundaries at sample x positions](images/string_tracking/07_band_boundaries.png)

![Colored edges by string](images/string_tracking/08_colored_edges.png)

#### Final Overlay

Frame with colored edges (in ROI) and string legend (bottom right).

![Final overlay](images/string_tracking/09_final_overlay.png)

---

### Suspect Detection (String Intensity)

For each frame we compute **edge intensity** per string: mean vertical Sobel gradient in a band around the string line. A rolling median over the previous 25 frames gives the baseline. If `current < baseline * (1 - 0.18)` (18% drop), we mark that string as suspected.

![Edge intensity over time - drops indicate possible play](images/string_tracking/test_edges.png)

---

## Part 2: Hands Region Detection

The hands (fretting and strumming) occlude parts of the strings. Edge intensity computed over the full frame width is diluted by static hand regions. We restrict analysis to the **relevant zone** between the two hands, where strings are visible and motion blur from playing is detectable.

### Approach: Two Signals

1. **Skin detection (HSV)** - Hands have high skin density. Columns with low skin fraction = between hands.
2. **String visibility** - Columns where all 6 strings have strong edges = between hands (strings visible, not occluded).

We combine both: when both succeed, use their intersection; otherwise use the longer run. Then stretch left and right to better match hand positions.

### Debug: Skin and String Signals

![Skin overlay, skin density plot, string visibility plot, and detected bbox](images/hands_region/01_skin_string_debug.png)

Top to bottom: original frame; frame with skin mask overlay (red = skin); skin density per column (low = between hands); string visibility per column (high = between hands). Yellow box = algorithm-detected used region.

### Algorithm Output

![Algorithm-detected used region with colored edges](images/hands_region/02_algorithm_region.png)

Per-frame detection with temporal smoothing. No per-video calibration required.

### Manual Calibration (Alternative)

For tuning or when the algorithm struggles, we support manual calibration: select the ROI between hands on 10 sample frames. Interpolation gives a bbox for any frame.

![Calibration preview at sample frame](images/hands_region/03_calibration_preview.png)

### Calibration Over Frames

![10x10 grid of frames with interpolated used region](images/hands_region/04_used_region_grid.png)

Shows how the calibration bbox changes across 100 consecutive frames.

### Why Filter: Filtered vs Full Width

Computing edge intensity only in the relevant zone (filtered) vs full frame width (unfiltered):

![Edge intensity: filtered (algorithm) vs full width](images/hands_region/05_filtered_vs_full_comparison.png)

Filtered traces show higher dynamic range and clearer peaks/dips. Full-width traces are smoother and diluted by background. Filtering improves suspect detection.

### Config

Parameters in `config/hands_region.json`:

- `skinFracThresh` (0.30) - Skin fraction threshold; columns below = between hands
- `minRunFrac` (0.08) - Minimum run length as fraction of width
- `leftStretchFrac` (0.30), `rightStretchFrac` (0.10) - Extend detected range
- `smoothingAlpha` (0.70) - Temporal smoothing
- `stringMinFrac` (0.40) - String visibility threshold

### Iterations

| Change | Reason |
|--------|--------|
| Full width -> Filter to relevant zone | Hands dilute edge signal; focus on visible strings |
| Calibration-only -> Algorithm option | No manual calibration needed; `useAlgorithm=True` |
| Skin only -> Skin + string visibility | String visibility improves robustness |
| Intersection when both succeed | More conservative, avoids false expansion |
| Temporal smoothing | Reduces jitter across frames |
| Config file | Tune params without code changes |

---

## Annotation Tool

Single-window annotator with:

- **N/P** - Next/previous suspect frame
- **Space** - Play/pause (with audio when available)
- **Arrows** - Frame step
- **V** - Mark suspect as correct (on suspect frames)
- **1-6** - Annotate string played (on suspect: correct wrong detection; on any frame: manual tag for algorithm misses)
- **Q** - Quit
- **-a** - Use algorithm for hands region (no calibration)

### Iterations

| Change | Reason |
|--------|--------|
| Frame-by-frame -> Suspect-only | Too many frames; focus on algorithm detections |
| Two windows -> Single window | Simpler; one view with sound and overlay |
| Rectangles -> Colored edges + legend | Rectangles cluttered; colored edges show string assignment clearly |
| C/N/1-6 -> V/1-6 | V = correct, 1-6 = wrong or manual tag |
| A/D -> N/P | N/P for next/prev suspect |
| Play only with audio -> Play without audio | Space works even when ffmpeg unavailable |
| Suspect-only navigation -> Arrows for any frame | Can manually tag frames algorithm missed |
| Calibration-only -> Algorithm option | `-a` flag uses hands region algorithm |

---

## Audio

**Originally** we used `ffmpeg` from system PATH. **Problem**: PATH not always visible to Python (e.g. IDE terminal, user env vars not propagated). **Changed** to `imageio-ffmpeg` which bundles ffmpeg - no PATH needed. Run `pip install imageio-ffmpeg`.

---

## Output

- `output/annotations.json` - All annotations (correct, wrong, manual)
- `output/debug/` - String tracking debug images
- `output/debug_hands/` - Hands region debug images
- `output/algorithm_100_frames/` - Algorithm-detected region per frame
- `output/calibration_preview/` - Calibration preview
- `output/compare_edge_intensity.png` - Filtered vs full width comparison

Regenerate images:

```
python run_string_tracking.py data/video.mp4 -o out --no-show
python -m scripts.inference.generateDebugImages
python run_hands_region.py debug
python -m scripts.inference.outputAlgorithmResults
python run_compare_intensity
```

---

## Next Steps

- Compare algorithm suspects to human annotations (precision/recall)
- Tune drop threshold and baseline window
- Consider temporal smoothing or multi-frame confirmation
