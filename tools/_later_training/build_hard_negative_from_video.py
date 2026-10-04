# build_hard_negative_from_video.py
# -*- coding: utf-8 -*-
"""
Hard-negative frame extractor (YOLO format: empty labels).

What it does:
- Extracts frames from a video at a target FPS (sampling).
- Writes images to:   <out_dir>/images/
- Writes empty txt to:<out_dir>/labels/   (same stem as image)
- Writes list file:   <out_dir>/train.txt (paths to images)

Typical use (project root):
  py -3.10 .\tools\build_hard_negative_from_video.py --video .\youtube_videos\empty_scene.mp4 --out_dir .\hard_negative --fps 2

Then you can include hard_negative/train.txt into training (either merge lists or build a combined list).
"""
import argparse
from pathlib import Path
import cv2


def cap_fps(cap) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6 or fps > 240:
        return 25.0
    return fps


def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(p.resolve()).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video file path (mp4/avi/mkv...).")
    ap.add_argument("--out_dir", default="hard_negative", help="Output folder (will create images/ labels/ train.txt inside).")
    ap.add_argument("--fps", type=float, default=2.0, help="Target sampling FPS (e.g., 2 or 5).")
    ap.add_argument("--start_sec", type=float, default=0.0, help="Start time (seconds).")
    ap.add_argument("--end_sec", type=float, default=-1.0, help="End time (seconds). -1 means until end.")
    ap.add_argument("--max_frames", type=int, default=0, help="Max frames to save (0 = no limit).")
    ap.add_argument("--prefix", default="neg", help="Filename prefix, e.g., neg_000001.jpg")
    ap.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality 1-100.")
    ap.add_argument("--list_root", default=".", help="Paths in train.txt will be relative to this root if possible.")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"ERROR: video not found: {video}")

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open video: {video}")

    src_fps = cap_fps(cap)
    target_fps = float(args.fps)
    if target_fps <= 0:
        target_fps = src_fps  # no sampling, take all frames

    step = max(1, int(round(src_fps / target_fps)))

    # Jump to start time
    if args.start_sec and args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(args.start_sec) * 1000.0)

    # End time handling
    end_sec = float(args.end_sec)
    end_msec = end_sec * 1000.0 if end_sec and end_sec > 0 else None

    saved = 0
    read_i = 0

    train_list_path = out_dir / "train.txt"
    train_list_path.write_text("", encoding="utf-8")

    list_root = Path(args.list_root)

    print("VIDEO:", video)
    print("SRC_FPS:", f"{src_fps:.2f}", "| TARGET_FPS:", f"{target_fps:.2f}", "| STEP:", step)
    print("OUT:", out_dir.resolve())
    print("START_SEC:", args.start_sec, "| END_SEC:", args.end_sec)
    print("NOTE: labels are empty txt files (hard negative).")

    while True:
        if end_msec is not None:
            cur_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if cur_msec >= end_msec:
                break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if read_i % step == 0:
            name = f"{args.prefix}_{saved:06d}"
            img_path = img_dir / f"{name}.jpg"
            lbl_path = lbl_dir / f"{name}.txt"

            okw = cv2.imwrite(
                str(img_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, args.jpg_quality)))],
            )
            if not okw:
                print("WARN: failed to write", img_path)
            else:
                lbl_path.write_text("", encoding="utf-8")

                rel = safe_relpath(img_path, list_root)
                with train_list_path.open("a", encoding="utf-8") as f:
                    f.write(rel + "\n")

                saved += 1
                if args.max_frames and saved >= int(args.max_frames):
                    break

        read_i += 1

        if saved and saved % 200 == 0 and read_i % step == 0:
            print("saved:", saved)

    cap.release()

    print("\nDONE")
    print("images :", img_dir.resolve())
    print("labels :", lbl_dir.resolve())
    print("list   :", train_list_path.resolve())
    print("saved  :", saved)


if __name__ == "__main__":
    main()
