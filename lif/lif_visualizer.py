import cv2

def draw_combined_overlay(frame, strings, segments, string_color=(0, 255, 0), segment_color=(255,0,0), thickness=2):
    """
    Plots the infinite mathematical string lines and the mapped vertical segments.
    
    strings: List of (m, b) tuples.
    segments: List of dicts with 'start': (x,y) and 'end': (x,y).
    """
    display_frame = frame.copy()
    h, w = frame.shape[:2]

    # 1. Draw the Mathematical Strings (Edge-to-Edge)
    for m, b in strings:
        x0, x1 = 0, w
        y0 = int(m * x0 + b)
        y1 = int(m * x1 + b)
        cv2.line(display_frame, (x0, y0), (x1, y1), string_color, thickness, lineType=cv2.LINE_AA)

    # 2. Draw the Mapped Segments (The tilted vertical lines)
    for seg in segments:
        p1 = seg['start']
        p2 = seg['end']
        # We use a different color (red by default) to distinguish them from strings
        cv2.line(display_frame, p1, p2, segment_color, thickness + 1, lineType=cv2.LINE_AA)

    # 3. Display results
    cv2.imshow("Overlaid Strings and Segments", display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return display_frame