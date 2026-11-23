"""Check which pretrained YOLO weights load with the installed Ultralytics version.

Run from the project root (needs internet the first time, weights are downloaded):
    python -m scripts.check_models
"""
import time

import ultralytics
from ultralytics import YOLO

CANDIDATES = ["yolo11n.pt", "yolo26n.pt"]


def main() -> None:
    print(f"ultralytics {ultralytics.__version__}")
    for weights in CANDIDATES:
        start = time.perf_counter()
        try:
            model = YOLO(weights)
        except Exception as exc:  # report every failure, do not stop at the first
            print(f"FAIL {weights}: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - start
        print(f"OK   {weights}: loaded in {elapsed:.1f}s, {len(model.names)} classes")


if __name__ == "__main__":
    main()
