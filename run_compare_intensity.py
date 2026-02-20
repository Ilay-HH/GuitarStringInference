"""Run edge intensity comparison. Usage: python run_compare_intensity.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.inference.compareEdgeIntensity import main

if __name__ == "__main__":
    main()
