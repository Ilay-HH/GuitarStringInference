"""Run the main frame annotator. Usage: python run_annotator.py [-v video] [-a]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.inference.frameAnnotator import main

if __name__ == "__main__":
    main()
