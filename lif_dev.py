import cv2
import numpy as np
import matplotlib.pyplot as plt
from lif.StringDetector import GuitarStringDetector
from lif.FretDetectoer import FRET_CONFIG, find_frets, get_rotated_string_crop
from lif.lif_visualizer import draw_combined_overlay

DEBUG = False
image_path = 'lif\\guitar.png'
image_path = 'docs\\images\\01_original.png'

# Initialize with debug=True to see the plots
detector = GuitarStringDetector(debug=DEBUG)

# Run the pipeline
frame = cv2.imread(image_path)
strings = detector.run(frame)
frets = find_frets(frame, strings, debug=DEBUG)

draw_combined_overlay(frame, strings, frets)


