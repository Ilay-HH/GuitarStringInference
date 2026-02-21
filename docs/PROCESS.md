# Guitar Tab Inference - Process Documentation

Work-in-progress documentation of our approach to infer which guitar strings are played from video using edge detection and motion blur.

---

## Hypothesis

When a guitar string is played, it vibrates at high speed. This motion causes **blur** in the video frame: the string "smears" across multiple pixels. A stationary string has sharp edges; a vibrating string has softer, spread-out edges. Therefore: **a drop in edge intensity** around a string indicates it was likely played.

---

## Pipeline Overview

1. **Hands Region** - Find the relevant zone between the two hands (filter out fretting/strumming hands). Used first to define the ROI.
2. **String Tracking** - Detect the 6 string positions within the hands region ROI; track edge intensity per string
3. **Suspect Detection** - Scan video for frames where edge intensity drops (within the relevant zone)
4. **Annotation** - Human validates/corrects suspects and tags missed frames
5. **Evaluation** (future) - Compare algorithm output to annotations

### Hands-First Flow

We use hands_region to define the processing ROI *before* string detection. Flow: (1) Get bbox from skin detection in config-defined vertical search area; (2) Crop to that bbox; (3) Run Canny + Hough to detect strings; (4) When strings are found, refine ROI height to the actual string span. This ensures we only process the relevant fretboard area from the start.

---

## Part 1: String Tracking

### Calibration: Finding the Strings

We use the first frame to locate the 6 strings. The process:

#### Step 1: Original Frame

![Original frame](images/string_tracking/01_original.png)

#### Step 2: Region of Interest (Hands-First)

We use hands_region to define the ROI *before* string detection. `getProcessingRoi` returns the bbox: when `roi_width` is `"auto"`, skin detection finds the x-range (low skin = between hands); when `roi_height` is `"auto"`, we use config fractions for the search area. After string detection, we refine the height to the actual string span (string 1 to 6 at mid-x with padding). Config: `config/hands_region.json` (`roi_height`, `roi_width`). Use `"auto"` or `[minFrac, maxFrac]` for fixed fractions.

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

#### Iteration: Hands-First ROI

**Originally** we used a fixed 20%-80% vertical ROI for string detection, then applied hands_region to filter the x-range for intensity tracking. **Changed** to use hands_region first: `detectStringLinesInHandsRegion` calls `getProcessingRoi` (skin-based bbox) to crop the frame, runs Canny and Hough on that crop, then refines the ROI height from the detected strings. This reduces noise and focuses string detection on the relevant zone from the start.

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

### Algorithm

**Skin detection (width)** - Hands have high skin density. Columns with low skin fraction = between hands. We find the longest contiguous run of low-skin columns, stretch left and right (`leftStretchFrac`, `rightStretchFrac`) to better match hand positions, and apply per-frame temporal smoothing (exponential moving average). No per-video calibration required.

**Height refinement (when strings available)** - The vertical bounds come from config (`roi_height_fixed` when `"auto"`). When string lines are detected, we refine y1/y2 to the span of string 1 and 6 at mid-x, with `stringHeightPadding` for margin. This tightens the ROI to the actual fretboard.

![Algorithm-detected used region with colored edges](images/hands_region/02_algorithm_region.png)

### Skin Detection (Detail)

We convert the frame to HSV and threshold for skin tones: Hue 0-20 (red/orange range), Saturation >= 48, Value >= 80. A small morphological open (erode then dilate) removes noise. For each column in the ROI, we compute the fraction of skin pixels in a 9-pixel-wide band. Columns below `skinFracThresh` are considered "between hands" (low skin = fretboard visible). We find the longest contiguous run of such columns and require it to span at least `minRunFrac` of the frame width.

![Skin overlay, skin density plot, string visibility plot, and detected bbox](images/hands_region/01_skin_string_debug.png)

Top to bottom: original frame; skin mask overlay (red = skin); skin density per column (low = between hands); string visibility per column (high = between hands). Yellow box = algorithm-detected used region.

### Fine Tuning

Parameters in `config/hands_region.json`:

| Param | Default | Description |
|-------|---------|-------------|
| `roi_height` | "auto" | `"auto"` = use hands_region; `[minFrac, maxFrac]` = fixed vertical bounds |
| `roi_width` | "auto" | `"auto"` = use hands_region; `[minFrac, maxFrac]` = fixed horizontal bounds |
| `roi_height_fixed` | [0.2, 0.8] | Fallback when roi_height is "auto" (search area for skin) |
| `roi_width_fixed` | [0.2, 0.8] | Fallback when roi_width is "auto" |
| `skinFracThresh` | 0.30 | Skin fraction threshold; columns below = between hands |
| `minRunFrac` | 0.08 | Minimum run length as fraction of frame width |
| `leftStretchFrac` | 0.30 | Extend detected left edge outward by this fraction of width |
| `rightStretchFrac` | 0.10 | Extend detected right edge outward |
| `smoothingAlpha` | 0.70 | Temporal smoothing (higher = less smoothing) |
| `stringHeightPadding` | 0.15 | Padding around string span when refining height |

### Iterations

| Change | Reason |
|--------|--------|
| Skin + string visibility -> Skin-only width | String visibility often failed; skin detection alone gave good x-range |
| Kept string-based height refinement | Height from config fractions (20%-80%) was too tall; strings tighten to actual fretboard |
| Fixed ROI first -> Hands-first flow | Use hands_region bbox before string detection; crop then detect; refine height from strings |

Filtering to the relevant zone improves suspect detection:

![Edge intensity: filtered (algorithm) vs full width](images/hands_region/05_filtered_vs_full_comparison.png)

Filtered traces show higher dynamic range and clearer peaks/dips. Full-width traces are smoother and diluted by background.

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
