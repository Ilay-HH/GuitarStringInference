import cv2
import numpy as np
import matplotlib.pyplot as plt


FRET_CONFIG = {
    "sobel_ksize": 3,
    "canny_low": 30,
    "canny_high": 30,
    "hough_threshold": 30,
    "min_height_ratio": 0.15, 
    "max_gap": 10,
    "min_fret_dist": 15,      # Minimum pixels between frets to avoid double-counting
    "max_fret_tilt": 50        # Max degrees a fret can tilt from vertical
}


def get_rotated_string_crop(frame, string_params, padding=0):
    """
    Rotates the frame to make strings horizontal and returns a tight crop.
    """
    if not string_params:
        return frame

    h, w = frame.shape[:2]
    
    # 1. Calculate the average slope (m) to find the rotation angle
    avg_m = np.mean([p[0] for p in string_params])
    # angle in radians: theta = arctan(m)
    angle_deg = np.degrees(np.arctan(avg_m))

    # 2. Get the center of the string area for rotation
    # We use the midpoint of the strings' Y-positions at the center of the frame
    mid_x = w / 2
    avg_y_at_mid = np.mean([m * mid_x + b for m, b in string_params])
    center = (mid_x, avg_y_at_mid)

    # 3. Create the rotation matrix
    # We rotate by -angle_deg to "level" the strings
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # 4. Rotate the entire frame
    rotated_frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

    # 5. Calculate the new vertical boundaries in the rotated space
    # After rotation, all strings are (theoretically) horizontal at their y_mid height
    rotated_y_coords = []
    for m, b in string_params:
        # The Y-coordinate of a leveled line is its height at the rotation center
        y_rotated = m * mid_x + b
        rotated_y_coords.append(y_rotated)
    
    y_min = max(0, int(min(rotated_y_coords) - padding))
    y_max = min(h, int(max(rotated_y_coords) + padding))

    # 6. Final crop
    return rotated_frame[y_min:y_max, 0:w], M, y_min, y_max



def get_original_segment(x_coord, y_min, y_max, M):
    """
    Converts a vertical line from the cropped image into a 
    tilted segment in the original image.
    """
    # 1. Get the inverse rotation matrix
    M_inv = cv2.invertAffineTransform(M)
    
    # 2. Define the segment endpoints in the cropped space
    # The top of the crop is 0, the bottom is (y_max - y_min)
    crop_height = y_max - y_min
    points_in_crop = np.array([
        [x_coord, 0 + y_min],           # Top point (adjusted for crop offset)
        [x_coord, crop_height + y_min]  # Bottom point (adjusted for crop offset)
    ], dtype='float32').reshape(-1, 1, 2)

    # 3. Transform points back to original coordinates
    points_original = cv2.transform(points_in_crop, M_inv)
    
    # Reshape to a simple list of two (x, y) tuples
    p1, p2 = points_original.squeeze().tolist()
    
    return {
        "start": (int(p1[0]), int(p1[1])),
        "end": (int(p2[0]), int(p2[1]))
    }


def get_vertical_edges(image, config):
    """Highlights vertical lines using Sobel X."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel X (1, 0) detects vertical changes in intensity
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=config["sobel_ksize"])
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    
    # Normalize and apply Canny
    normalized = cv2.normalize(abs_grad_x, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(normalized, config["canny_low"], config["canny_high"])
    
    return edges

def clean_fret_extremes(frets_x, max_frets=20, anomaly_ratio=1.5):
        """
        Recursively cleans leading outliers and trims frets beyond a theoretical maximum.
        Assumes frets_x is a sorted list of X-coordinates.
        """
        # Base case: we need at least 3 frets to compare two gaps
        if len(frets_x) < 3:
            return frets_x

        # --- STEP 1: Recursive Leading Gap Cleaning ---
        gap0 = frets_x[1] - frets_x[0]
        gap1 = frets_x[2] - frets_x[1]

        # If the first gap is massively larger than the second gap, 
        # the first point is an outlier (e.g., soundhole or body edge).
        if gap0 > anomaly_ratio * gap1:
            # Remove the first element and recurse
            return clean_fret_extremes(frets_x[1:], max_frets, anomaly_ratio)

        # --- STEP 2: Estimate Maximum Fretboard Span ---
        # First, figure out which direction the neck is facing based on the gaps.
        # If gaps shrink, we are going Nut -> Bridge. If they grow, Bridge -> Nut.
        if gap1 < gap0:
            theoretical_r = 1.0 / (2 ** (1.0 / 12))  # ~0.943
        else:
            theoretical_r = 2 ** (1.0 / 12)          # ~1.059

        # Project the position of the Nth fret using a geometric series
        projected_max_x = frets_x[0]
        current_gap = gap0

        for _ in range(max_frets):
            projected_max_x += current_gap
            current_gap *= theoretical_r

        # Add a small tolerance buffer (half a gap) to account for slight detection noise
        tolerance = current_gap / 2
        absolute_max_x = projected_max_x + tolerance

        # Keep only the frets that fall within our mathematically projected guitar neck
        cleaned_frets = [x for x in frets_x if x <= absolute_max_x]

        return cleaned_frets

def interpolate_missing_frets(detected_x_coords, min_gap_px=10):
    """
    Generates full fretboard hypotheses from each adjacent pair,
    scores them using MSE, and merges the best hypothesis with the 
    original detections, dropping interpolated frets that collide.
    """
    x_coords = sorted(list(set(detected_x_coords)))

    if len(x_coords) < 2:
        return x_coords
    ratio = 2 ** (1.0 / 12)
    min_x = x_coords[0]
    max_x = x_coords[-1]
    
    best_series = []
    best_score = float('inf')

    # --- 1. Generate & Score Hypotheses ---
    for i in range(min(len(x_coords)-1,8)):
        x1 = x_coords[i]
        x2 = x_coords[i+1]
        initial_gap = x2 - x1
        
        # Generate Right
        right_series = [x1, x2]
        curr_x = x2
        curr_gap = initial_gap
        loop_count = 0
        while curr_x < max_x and loop_count < 100:
            curr_gap = max(curr_gap * ratio, min_gap_px)
            next_x = curr_x + curr_gap
            if next_x >= max_x:
                if abs(next_x - max_x) < abs(curr_x - max_x):
                    right_series.append(next_x)
                break
            else:
                right_series.append(next_x)
                curr_x = next_x
            loop_count += 1
            
        # Generate Left
        left_series = []
        curr_x = x1
        curr_gap = initial_gap
        loop_count = 0
        while curr_x > min_x and loop_count < 100:
            curr_gap = max(curr_gap / ratio, min_gap_px)
            prev_x = curr_x - curr_gap
            if prev_x <= min_x:
                if abs(prev_x - min_x) < abs(curr_x - min_x):
                    left_series.append(prev_x)
                break
            else:
                left_series.append(prev_x)
                curr_x = prev_x
            loop_count += 1
            
        left_series.reverse()
        full_series = left_series + right_series
        
        # Score Hypothesis (MSE)
        score = 0
        for actual in x_coords:
            closest = min(full_series, key=lambda x: abs(x - actual))
            score += (actual - closest) ** 2
            
        if score < best_score:
            best_score = score
            best_series = full_series

    # --- 2. Merge and Resolve Collisions ---
    # Guarantee all original detected frets are kept
    final_frets = set(x_coords)
    
    for i, t_x in enumerate(best_series):
        # Determine the local gap to create a dynamic collision threshold
        if i < len(best_series) - 1:
            local_gap = best_series[i+1] - t_x
        elif i > 0:
            local_gap = t_x - best_series[i-1]
        else:
            local_gap = min_gap_px
            
        # We define a collision if the theoretical fret is within 30% 
        # of the local gap distance to an actual detected fret.
        collision_threshold = local_gap * 0.3 
        closest_actual = min(x_coords, key=lambda x: abs(x - t_x))
        
        if abs(t_x - closest_actual) > collision_threshold:
            # No collision occurred; this is a genuinely missing fret
            final_frets.add(t_x)

    # Clean up, convert to ints, and sort
    return sorted(list(int(round(x)) for x in final_frets))



def _plot_frets(img, edges, frets, interpolated_frets):
    """
    Debug plotting for frets.
    Original frets are drawn in Blue.
    Interpolated (missing) frets are drawn in Green.
    """
    vis = img.copy()
    
    # 1. Isolate the newly added frets
    original_frets_set = set(frets)
    new_frets = [x for x in interpolated_frets if x not in original_frets_set]
    
    # 2. Draw original detected frets (OpenCV uses BGR, so Blue is (255, 0, 0))
    for x in frets:
        cv2.line(vis, (x, 0), (x, img.shape[0]), (255, 0, 0), 2)
        
    # 3. Draw interpolated frets (Green is (0, 255, 0))
    for x in new_frets:
        cv2.line(vis, (x, 0), (x, img.shape[0]), (0, 255, 0), 2)
    
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot edge map
    ax1.imshow(edges, cmap='gray')
    ax1.set_title("Fret Edge Detection (Sobel X + Canny)")
    ax1.axis('off')
    
    # Plot final image with both sets of lines
    # Convert BGR to RGB so matplotlib displays the colors correctly
    ax2.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    
    # Update title to show the breakdown of the counts
    title_str = (f"Frets - Detected: {len(frets)} | "
                 f"Interpolated: {len(new_frets)} | "
                 f"Total: {len(interpolated_frets)}")
    ax2.set_title(title_str)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()



def find_frets(frame, strings, debug=False):
    """
    Identifies fret X-coordinates based on height span and verticality.
    """
    config = FRET_CONFIG
    cropped_frame, M, y_min, y_max = get_rotated_string_crop(frame, strings)
    if cropped_frame is None: return []
    h, w = cropped_frame.shape[:2]
    
    edges = get_vertical_edges(cropped_frame, config)
    
    # Calculate min length based on your 0.9 requirement
    min_len = int(h * config["min_height_ratio"])
    
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 
        threshold=config["hough_threshold"], 
        minLineLength=min_len, 
        maxLineGap=config["max_gap"]
    )

    if lines is None: return []

    # Filter for verticality and extract X-positions
    fret_x_points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Check if the line is vertical-ish using arctan
        angle = np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1)))
        if angle < config["max_fret_tilt"]:
            fret_x_points.append(int((x1 + x2) / 2))

    # Cluster X-positions to handle thick fret wires
    fret_x_points.sort()
    unique_frets = []
    if fret_x_points:
        unique_frets.append(fret_x_points[0])
        for i in range(1, len(fret_x_points)):
            if fret_x_points[i] - unique_frets[-1] > config["min_fret_dist"]:
                unique_frets.append(fret_x_points[i])
            interpolated_frets = interpolate_missing_frets(unique_frets)
    
    cleaned_frets = clean_fret_extremes(unique_frets)
    interpolated_frets = interpolate_missing_frets(cleaned_frets)
    if debug:
        _plot_frets(cropped_frame, edges, cleaned_frets, interpolated_frets)

    return [get_original_segment(x, y_min, y_max, M) for x in interpolated_frets]