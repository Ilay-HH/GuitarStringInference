"""Run hands region tools (standalone). Usage: python run_hands_region.py [calibrate|debug|tune] video.mp4"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_hands_region.py <calibrate|debug|tune> [video] [args...]")
        print("  calibrate: python run_hands_region.py calibrate video.mp4 [--sequence]")
        print("  debug:    python run_hands_region.py debug [video]")
        print("  tune:     python run_hands_region.py tune [video]")
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "calibrate":
        from scripts.hands_region.handsCalibrator import main as run
        sys.argv = ["handsCalibrator"] + sys.argv[2:]
        run()
    elif cmd == "debug":
        from scripts.hands_region.debugHandsRegion import main as run
        sys.argv = ["debugHandsRegion"] + sys.argv[2:]
        run()
    elif cmd == "tune":
        from scripts.hands_region.tuneHandsParams import main as run
        sys.argv = ["tuneHandsParams"] + sys.argv[2:]
        run()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    main()
