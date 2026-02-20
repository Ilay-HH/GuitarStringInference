"""Run string tracking (standalone). Usage: python run_string_tracking.py video.mp4 [-o output]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.string_tracking.stringEdgeTracker import main

if __name__ == "__main__":
    main()
