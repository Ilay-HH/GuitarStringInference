import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# HYPERPARAMETERS
# ==========================================
DETECTION_CONFIG = {
    # Edge Detection
    "sobel_ksize": 3,
    "canny_low": 150,    # Lower this if the top string is missing
    "canny_high": 200,
    
    # Hough Transform
    "hough_threshold": 300,
    "hough_min_line_len_ratio": 0.25, # Fraction of frame width
    "hough_max_gap": 100,             # Higher jumps over fingers/frets
    
    # Clustering & Filtering
    "cluster_dist_px": 10,            # Max Y-distance to group segments
    "max_slope_threshold": 0.2,       # Filters out vertical lines (frets)
    "num_strings": 15                  # Default target count
}

def draw_mathematical_strings(frame, lines, color=(0, 255, 0), thickness=2):
    """
    Guarantees 6 single, straight lines from edge to edge.
    """
    # Create a clean copy so we don't draw over old lines (prevents the piecewise look)
    display_frame = frame.copy()
    h, w = frame.shape[:2]

    for m, b in lines:
        # 1. Define the x-coordinates at the extreme boundaries
        x0 = 0
        x1 = w
        
        # 2. Calculate the corresponding y-coordinates using y = mx + b
        y0 = int(m * x0 + b)
        y1 = int(m * x1 + b)

        # 3. Draw ONE single line from the left edge to the right edge
        # cv2.LINE_AA makes the line smooth and professional
        cv2.line(display_frame, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)

    cv2.imshow("Detected Strings", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return display_frame

class GuitarStringDetector:
    def __init__(self, config=DETECTION_CONFIG, debug=False):
        self.config = config
        self.debug = debug

    def _show_debug_image(self, title, image):
        if self.debug:
            plt.figure(figsize=(10, 5))
            is_gray = len(image.shape) == 2
            plt.imshow(image, cmap='gray' if is_gray else None)
            plt.title(title)
            plt.axis('off')
            plt.show()

    def get_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Gradient detection
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=self.config["sobel_ksize"])
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # 2. Normalize to boost faint strings
        normalized = cv2.normalize(abs_grad_y, None, 0, 255, cv2.NORM_MINMAX)
        
        # 3. Canny Hysteresis
        edges = cv2.Canny(normalized, self.config["canny_low"], self.config["canny_high"])
        
        self._show_debug_image("Step 1: Edge Detection", edges)
        return edges

    def extract_line_segments(self, edges, frame_width):
        min_len = int(frame_width * self.config["hough_min_line_len_ratio"])
        
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=self.config["hough_threshold"], 
            minLineLength=min_len, 
            maxLineGap=self.config["hough_max_gap"]
        )
        
        if self.debug and lines is not None:
            debug_img = np.zeros((edges.shape[0], frame_width, 3), dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self._show_debug_image("Step 2: Hough Segments", debug_img)
            
        return lines

    def cluster_and_average(self, lines, frame_width, target_count=None):
        if lines is None: return []
        
        target_count = target_count or self.config["num_strings"]
        mid_x = frame_width / 2
        line_data = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = (x2 - x1)
            if dx == 0: continue 
            
            m = (y2 - y1) / dx
            
            # FILTER: Ignore lines that are too steep (likely frets or background)
            if abs(m) > self.config["max_slope_threshold"]:
                continue
                
            b = y1 - m * x1
            y_at_mid = m * mid_x + b
            line_data.append({'m': m, 'b': b, 'y_mid': y_at_mid})

        if not line_data: return []
        line_data.sort(key=lambda x: x['y_mid'])
        
        # Group segments vertically
        clusters = []
        current_cluster = [line_data[0]]
        
        for i in range(1, len(line_data)):
            dist = line_data[i]['y_mid'] - current_cluster[-1]['y_mid']
            if dist < self.config["cluster_dist_px"]:
                current_cluster.append(line_data[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [line_data[i]]
        clusters.append(current_cluster)

        # Average math for each cluster
        final_lines = []
        for cluster in clusters[:target_count]:
            avg_m = np.mean([item['m'] for item in cluster])
            avg_b = np.mean([item['b'] for item in cluster])
            final_lines.append((avg_m, avg_b))
            
        return final_lines

    def run(self, frame, num_strings=None):
        edges = self.get_edges(frame)
        raw_lines = self.extract_line_segments(edges, frame.shape[1])
        results = self.cluster_and_average(raw_lines, frame.shape[1], num_strings)
        if self.debug:
            draw_mathematical_strings(frame=frame, lines=results)
        
        return results