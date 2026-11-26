"""Cut a short, smaller copy of a video so it is easy to share.

Example:
    python scripts/shrink_clip.py data/clips/site_walk_01.mp4 --seconds 10 --width 640

Writes data/clips/small/<name>.mp4 next to the original (or use --out). Audio is dropped.
The original is never modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def shrink(source: Path, out: Path, *, start: float, seconds: float, width: int) -> dict:
    if width < 64:
        raise ValueError("width must be at least 64")
    if seconds <= 0 or start < 0:
        raise ValueError("seconds must be positive and start must not be negative")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {source}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0 or fps > 240:
            raise ValueError(f"unusable frame rate ({fps}) in {source}")
        src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"unusable frame size in {source}")
        new_w = min(width, src_w)
        new_w -= new_w % 2
        new_h = int(round(src_h * new_w / src_w)) // 2 * 2
        first = int(round(start * fps))
        last = first + int(round(seconds * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))
        if not writer.isOpened():
            raise RuntimeError("could not create the output video")
        written = 0
        try:
            for _ in range(first, last):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA))
                written += 1
        finally:
            writer.release()
    finally:
        cap.release()
    if written == 0:
        out.unlink(missing_ok=True)
        raise ValueError("no frames were read (is --start beyond the end of the video?)")
    return {"fps": fps, "frames": written, "size": (new_w, new_h), "bytes": out.stat().st_size}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--start", type=float, default=0.0, help="start second (default 0)")
    ap.add_argument("--seconds", type=float, default=10.0, help="length to keep (default 10)")
    ap.add_argument("--width", type=int, default=640, help="output width in pixels (default 640)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or args.video.parent / "small" / f"{args.video.stem}.mp4"
    try:
        info = shrink(args.video, out, start=args.start, seconds=args.seconds, width=args.width)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}: {info['frames']} frames, {info['size'][0]}x{info['size'][1]}, "
          f"{info['fps']:.1f} fps, {info['bytes'] / 1_000_000:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
