"""Run hands region calibrator (standalone). Usage: python run_hands_calibrator.py video.mp4 [--sequence]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.hands_region.handsCalibrator import main

if __name__ == "__main__":
    main()
