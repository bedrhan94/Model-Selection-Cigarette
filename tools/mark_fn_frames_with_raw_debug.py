from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from ultralytics import YOLO

Box = Tuple[float, float, float, float, float]  # x1,y1,x2,y2,conf


def load_final_preds(pred_jsonl: Path) -> Dict[int, List[Box]]:
    out: Dict[int, List[Box]] = {}
    with pred_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fi = int(rec.get("frame_idx", -1))
            preds = rec.get("pred", []) or []
            boxes = []
            for p in preds:
                try:
                    boxes.append((
                        float(p["x1"]),
                        float(p["y1"]),
                        float(p["x2"]),
                        float(p["y2"]),
                        float(p.get("conf", 0.0)),
                    ))
                except Exception:
                    continue
            out[fi] = boxes
    return out


def load_pred_presence(pred_jsonl: Path, conf_thr: float) -> Dict[int, bool]:
    frame_has_pred: Dict[int, bool] = {}
    with pred_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fi = int(rec.get("frame_idx", -1))
            preds = rec.get("pred", []) or []
            ok = False
            for p in preds:
                try:
                    conf = float(p.get("conf", 0.0))
                except Exception:
                    conf = 0.0
                if conf >= conf_thr:
                    ok = True
                    break
            if fi >= 0:
                frame_has_pred[fi] = ok
    return frame_has_pred


def load_gt_segments(gt_csv: Path) -> List[Tuple[float, float]]:
    segs = []
    with gt_csv.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            segs.append((float(r["start_s"]), float(r["end_s"])))
    return segs


def is_gt_positive(t_s: float, segs: List[Tuple[float, float]]) -> bool:
    for s, e in segs:
        if s <= t_s <= e:
            return True
    return False


def draw_box(img, box: Box, color, label, thickness=2):
    x1, y1, x2, y2, conf = box
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty1 = max(0, y1 - th - 8)
    cv2.rectangle(img, (x1, ty1), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, text, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--fn_frames_dir", default="")
    ap.add_argument("--gt_segments", default="")
    ap.add_argument("--final_conf_thr", type=float, default=0.10)
    ap.add_argument("--raw_conf", type=float, default=0.03)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max_det", type=int, default=50)
    ap.add_argument("--save_only_fn_jpg", type=int, default=1)
    args = ap.parse_args()

    video_path = Path(args.video)
    weights_path = Path(args.weights)
    pred_path = Path(args.pred)
    out_mp4 = Path(args.out_mp4)
    out_csv = Path(args.out_csv)
    fn_dir = Path(args.fn_frames_dir) if args.fn_frames_dir else None

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if fn_dir:
        fn_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    frame_has_pred = load_pred_presence(pred_path, args.final_conf_thr)
    final_preds = load_final_preds(pred_path)

    segs = None
    if args.gt_segments:
        segs = load_gt_segments(Path(args.gt_segments))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    fn_rows = []
    frame_idx = 0
    fn_count = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        t_s = frame_idx / fps
        gt_pos = True if segs is None else is_gt_positive(t_s, segs)
        pred_pos = frame_has_pred.get(frame_idx, False)
        final_boxes = final_preds.get(frame_idx, [])

        vis = frame.copy()

        is_fn = gt_pos and not pred_pos

        # raw model prediction
        raw_boxes: List[Box] = []
        if is_fn:
            res = model.predict(
                source=frame,
                conf=args.raw_conf,
                imgsz=args.imgsz,
                max_det=args.max_det,
                verbose=False
            )[0]

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), c in zip(xyxy, confs):
                    raw_boxes.append((float(x1), float(y1), float(x2), float(y2), float(c)))

        if is_fn:
            fn_count += 1

            # red border
            cv2.rectangle(vis, (0, 0), (W - 1, H - 1), (0, 0, 255), 12)

            # raw boxes = yellow
            for rb in raw_boxes:
                draw_box(vis, rb, (0, 255, 255), "RAW", thickness=2)

            # final boxes = green (usually none, but draw them anyway)
            for fb in final_boxes:
                draw_box(vis, fb, (0, 255, 0), "FINAL", thickness=2)

            cv2.rectangle(vis, (0, 0), (W, 52), (0, 0, 180), -1)
            txt = f"FN FRAME | frame={frame_idx} | time={t_s:.2f}s | raw={len(raw_boxes)} | final={len(final_boxes)}"
            cv2.putText(vis, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

            # legend
            cv2.rectangle(vis, (12, H - 110), (350, H - 10), (30, 30, 30), -1)
            cv2.putText(vis, "RED = FN frame", (22, H - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, "YELLOW = RAW detections", (22, H - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, "GREEN = FINAL detections", (22, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            fn_rows.append({
                "frame_idx": frame_idx,
                "time_s": round(t_s, 3),
                "raw_count": len(raw_boxes),
                "final_count": len(final_boxes),
            })

            if fn_dir is not None and args.save_only_fn_jpg == 1:
                cv2.imwrite(str(fn_dir / f"fn_{frame_idx:06d}.jpg"), vis)

        else:
            cv2.rectangle(vis, (0, 0), (W, 40), (30, 30, 30), -1)
            status = "TP/COVERED" if gt_pos and pred_pos else "NON-GT"
            txt = f"{status} | frame={frame_idx} | time={t_s:.2f}s"
            cv2.putText(vis, txt, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx", "time_s", "raw_count", "final_count"])
        w.writeheader()
        w.writerows(fn_rows)

    print("Done.")
    print("FN count:", fn_count)
    print("Video   :", out_mp4)
    print("CSV     :", out_csv)
    if fn_dir:
        print("FN JPG  :", fn_dir)


if __name__ == "__main__":
    main()