# tools/eval_video_metrics.py
# -*- coding: utf-8 -*-
"""
Video Metrics Toolkit (cigarette detection) - purely NUMERICAL evaluation
=========================================================================
Reports metrics only:

FRAME-LEVEL:
- How many minutes of smoking are in a 10 min video? (GT duration)
- How many frames contain a cigarette? (GT frame count)
- How many frames did we catch? (TP / FN)
- How many frames are false alarms? (FP)
- Why were 500 frames missed? (FN breakdown)
  - FN_NO_PRED   : no bbox at all
  - FN_LOW_CONF  : bbox exists but below conf_thr
  - FN_LOW_IOU   : bbox exists, conf is fine but IoU is low (if GT bbox is available)
  - FN_FILTERED_OUT : (optional) raw exists, filtered does not (if logged)

BOX-LEVEL:
- IoU overall/median/mean
- IoU trend (per_bucket.csv: 10 s windows)

AP (single class):
- AP@0.30 and AP@0.50 (if GT bbox is available)

EVENT (database style):
- Extracts events from the predicted frame sequence (confirm + stop_miss)
- Writes them to SQLite: events + evidence
- This lets us check on video, before going live, whether events reach the DB.

Commands:
1) Generate PRED:
   - YOLO:  python tools/eval_video_metrics.py predict --source yolo --video ... --weights models/best.pt --out outputs/pred.jsonl
   - API :  python tools/eval_video_metrics.py predict --source api  --video ... --api http://127.0.0.1:5000 --out outputs/pred.jsonl

2) Generate a GT template (frame export + csv template):
   python tools/eval_video_metrics.py make-gt --video ... --frames_out outputs/gt_frames --sample_fps 5 --csv_out outputs/gt_template.csv

3) EVAL (metrics + optional DB events):
   python tools/eval_video_metrics.py eval --video ... --pred outputs/pred.jsonl --gt_csv outputs/gt_filled.csv --out_dir outputs/metrics_run1 \
      --conf_thr 0.10 --iou_thr 0.30 --bucket_sec 10 \
      --event_db outputs/metrics_run1/metrics_events.db --event_confirm 3 --event_stop_miss 8 --event_min_dur_s 0.8 --event_evidence 1

GT formats:
A) BBOX GT (recommended):
- CSV columns: frame_idx,x1,y1,x2,y2   (add a class column if needed)
B) SEGMENT GT (no bbox, presence only):
- CSV columns: start_s,end_s
  (No IoU/AP in this mode; frame-level metrics only.)

Note:
- Cigarettes are small objects -> IoU=0.30 is the more meaningful report. Report 0.50 as well (standard).
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import requests  # type: ignore
except Exception:
    requests = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


# ------------------ geometry ------------------
def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, float(x1)))
    y1 = max(0, min(H - 1, float(y1)))
    x2 = max(0, min(W, float(x2)))
    y2 = max(0, min(H, float(y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 1:
        x2 = min(W, x1 + 1)
    if y2 - y1 < 1:
        y2 = min(H, y1 + 1)
    return (x1, y1, x2, y2)


# ------------------ IO utils ------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, total, W, H


def load_gt_csv(gt_csv: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    """
    GT bbox CSV: frame_idx,x1,y1,x2,y2
    Returns dict: frame_idx -> list of boxes
    """
    gt: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if pd is not None:
        df = pd.read_csv(gt_csv)
        required = {"frame_idx", "x1", "y1", "x2", "y2"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"GT CSV missing column(s): {sorted(missing)}")
        for _, r in df.iterrows():
            # skip empty rows
            if pd.isna(r["x1"]) or pd.isna(r["y1"]) or pd.isna(r["x2"]) or pd.isna(r["y2"]):
                continue
            fi = int(r["frame_idx"])
            box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            gt.setdefault(fi, []).append(box)
    else:
        import csv
        with open(gt_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("x1", "") == "" or row.get("y1", "") == "" or row.get("x2", "") == "" or row.get("y2", "") == "":
                    continue
                fi = int(row["frame_idx"])
                box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
                gt.setdefault(fi, []).append(box)
    return gt


def load_segments_csv(seg_csv: str, fps: float, total_frames: int) -> Dict[int, int]:
    """
    segments CSV: start_s,end_s
    returns frame_idx -> gt_present (0/1)
    """
    present = np.zeros((total_frames,), dtype=np.uint8)
    if pd is not None:
        df = pd.read_csv(seg_csv)
        if "start_s" not in df.columns or "end_s" not in df.columns:
            raise SystemExit("Segments CSV columns must be: start_s,end_s.")
        for _, r in df.iterrows():
            s = float(r["start_s"])
            e = float(r["end_s"])
            a = max(0, int(math.floor(s * fps)))
            b = min(total_frames, int(math.ceil(e * fps)))
            if b > a:
                present[a:b] = 1
    else:
        import csv
        with open(seg_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = float(row["start_s"])
                e = float(row["end_s"])
                a = max(0, int(math.floor(s * fps)))
                b = min(total_frames, int(math.ceil(e * fps)))
                if b > a:
                    present[a:b] = 1
    return {i: int(present[i]) for i in range(total_frames)}


def read_pred_jsonl(pred_jsonl: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["frame_idx"])] = rec
    return out


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ------------------ prediction ------------------
def predict_yolo(
    video: str,
    weights: str,
    out_jsonl: str,
    imgsz: int,
    conf: float,
    iou: float,
    sample_fps: float,
    max_det: int,
):
    if YOLO is None:
        raise SystemExit("ultralytics is not installed. Install: pip install ultralytics")
    model = YOLO(weights)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 1e-6 or fps > 240:
        fps = 25.0
    step = 1 if not sample_fps or sample_fps <= 0 else max(1, int(round(fps / float(sample_fps))))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    records: List[Dict[str, Any]] = []
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1
        if fi % step != 0:
            continue
        t_s = fi / fps

        res = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )[0]

        pred_list = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                pred_list.append(
                    {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(c)}
                )

        records.append({"frame_idx": fi, "time_s": float(t_s), "fps": float(fps), "pred": pred_list})

        if total and fi % (step * 200) == 0:
            pct = (fi / total) * 100.0
            print(f"[YOLO] {fi}/{total} ({pct:.1f}%) pred={len(pred_list)}")

    cap.release()
    write_jsonl(Path(out_jsonl), records)
    print("PRED JSONL ->", out_jsonl)


def predict_api(
    video: str,
    api: str,
    out_jsonl: str,
    jpeg_quality: int,
    sample_fps: float,
    camera_id: str,
):
    if requests is None:
        raise SystemExit("requests is not installed. Install: pip install requests")

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 1e-6 or fps > 240:
        fps = 25.0
    step = 1 if not sample_fps or sample_fps <= 0 else max(1, int(round(fps / float(sample_fps))))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    records: List[Dict[str, Any]] = []
    fi = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1
        if fi % step != 0:
            continue
        t_s = fi / fps

        ok2, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok2:
            continue
        files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
        data = {"camera_id": str(camera_id), "source": "video", "return_frame": "0"}
        r = requests.post(f"{api.rstrip('/')}/process_frame", files=files, data=data, timeout=120)
        r.raise_for_status()
        j = r.json()

        pred_boxes = j.get("smoking_boxes", []) or j.get("smokes", []) or []
        pred_list = []
        for sb in pred_boxes:
            box = None
            conf_val = 1.0
            if isinstance(sb, (list, tuple)) and len(sb) >= 4:
                box = sb[:4]
            elif isinstance(sb, dict):
                conf_val = float(sb.get("conf", 1.0))
                for k in ("bbox", "xyxy", "box"):
                    if k in sb and isinstance(sb[k], (list, tuple)) and len(sb[k]) >= 4:
                        box = sb[k][:4]
                        break
            if box is None:
                continue
            x1, y1, x2, y2 = map(float, box)
            pred_list.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf_val})

        rec = {"frame_idx": fi, "time_s": float(t_s), "fps": float(fps), "pred": pred_list}

        # optional raw/filtered if API logs them
        for k in ("raw_smoking_boxes", "raw_boxes", "raw_smokes"):
            if k in j:
                rec["raw"] = j[k]
        for k in ("filtered_smoking_boxes", "filtered_boxes", "filtered_smokes"):
            if k in j:
                rec["filtered"] = j[k]

        records.append(rec)

        if total and fi % (step * 200) == 0:
            pct = (fi / total) * 100.0
            print(f"[API] {fi}/{total} ({pct:.1f}%) pred={len(pred_list)}")

    cap.release()
    write_jsonl(Path(out_jsonl), records)
    print("PRED JSONL ->", out_jsonl)


# ------------------ evaluation (AP) ------------------
def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_ap_single_class(all_preds, gt_by_frame, iou_thr: float) -> float:
    n_gt = sum(len(v) for v in gt_by_frame.values())
    if n_gt == 0:
        return float("nan")

    preds = sorted(all_preds, key=lambda x: -x["conf"])
    tp = np.zeros(len(preds), dtype=np.float32)
    fp = np.zeros(len(preds), dtype=np.float32)

    matched: Dict[int, List[bool]] = {fi: [False] * len(gt_by_frame[fi]) for fi in gt_by_frame}

    for i, p in enumerate(preds):
        fi = p["frame_idx"]
        box_p = p["box"]
        gts = gt_by_frame.get(fi, [])
        if not gts:
            fp[i] = 1
            continue

        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gts):
            if matched[fi][j]:
                continue
            v = iou_xyxy(box_p, g)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            tp[i] = 1
            matched[fi][best_j] = True
        else:
            fp[i] = 1

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec = tp_c / float(n_gt)
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    return voc_ap(rec, prec)


# ------------------ events (DB) ------------------
@dataclass
class Event:
    source: str          # "PRED" or "GT"
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    meta: Dict[str, Any]


def build_events_from_binary(
    present: np.ndarray,
    fps: float,
    confirm_frames: int,
    stop_miss_frames: int,
    min_dur_s: float,
) -> List[Tuple[int, int]]:
    """
    present: 0/1 array
    start: after confirm_frames consecutive 1
    end: after stop_miss_frames consecutive 0
    Returns list of (start_frame, end_frame) inclusive.
    """
    confirm_frames = max(1, int(confirm_frames))
    stop_miss_frames = max(1, int(stop_miss_frames))
    min_frames = max(1, int(round(min_dur_s * fps)))

    events: List[Tuple[int, int]] = []
    in_ev = False
    on_count = 0
    off_count = 0
    start_frame = 0

    for i in range(len(present)):
        if present[i] == 1:
            on_count += 1
            off_count = 0
            if (not in_ev) and (on_count >= confirm_frames):
                in_ev = True
                start_frame = i - confirm_frames + 1
        else:
            off_count += 1
            on_count = 0
            if in_ev and (off_count >= stop_miss_frames):
                end_frame = i - stop_miss_frames
                if end_frame >= start_frame and (end_frame - start_frame + 1) >= min_frames:
                    events.append((start_frame, end_frame))
                in_ev = False
                off_count = 0

    # close if still open
    if in_ev:
        end_frame = len(present) - 1
        if end_frame >= start_frame and (end_frame - start_frame + 1) >= min_frames:
            events.append((start_frame, end_frame))

    return events


def init_event_db(db_path: Path):
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            video TEXT NOT NULL,
            model_tag TEXT,
            conf_thr REAL,
            iou_thr REAL,
            start_frame INTEGER NOT NULL,
            end_frame INTEGER NOT NULL,
            start_time_s REAL NOT NULL,
            end_time_s REAL NOT NULL,
            duration_s REAL NOT NULL,
            created_at_utc TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS event_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            frame_idx INTEGER NOT NULL,
            time_s REAL NOT NULL,
            image_path TEXT NOT NULL,
            best_conf REAL,
            best_iou REAL,
            note TEXT,
            created_at_utc TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    return conn


def save_event_evidence_frames(
    video: str,
    out_dir: Path,
    fps: float,
    events: List[Event],
    per_frame_rows: List[Dict[str, Any]],
    max_per_event: int,
):
    """
    Saves evidence frames for each event:
    - picks top frames by best_conf within event window (if available)
    - falls back to evenly spaced frames
    Writes JPEGs, returns updated events with evidence records.
    """
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print("WARN: could not open video for evidence export:", video)
        return []

    # Build quick index: frame_idx -> row
    row_by_fi = {int(r["frame_idx"]): r for r in per_frame_rows}

    # We'll seek by reading sequentially (fast enough for small #frames).
    # Precompute which frames to export
    export_map: Dict[int, List[Tuple[int, str, float, float]]] = {}  # event_index -> list[(fi, note, conf, iou)]
    for ei, ev in enumerate(events):
        s, e = ev.start_frame, ev.end_frame
        candidates = []
        for fi in range(s, e + 1):
            rr = row_by_fi.get(fi)
            if rr is None:
                continue
            bc = rr.get("best_conf", float("nan"))
            bi = rr.get("best_iou", float("nan"))
            if bc is None:
                continue
            try:
                bc_f = float(bc)
            except Exception:
                bc_f = float("nan")
            candidates.append((fi, bc_f, float(bi) if bi is not None else float("nan")))

        chosen: List[Tuple[int, str, float, float]] = []
        # pick top by conf
        cand2 = [c for c in candidates if not math.isnan(c[1])]
        cand2.sort(key=lambda x: x[1], reverse=True)
        for fi, bc, bi in cand2[:max_per_event]:
            chosen.append((fi, "top_conf", bc, bi))

        # if none, pick evenly spaced
        if not chosen:
            step = max(1, (e - s + 1) // max(1, max_per_event))
            for fi in range(s, e + 1, step):
                rr = row_by_fi.get(fi, {})
                bc = float(rr.get("best_conf", float("nan"))) if rr else float("nan")
                bi = float(rr.get("best_iou", float("nan"))) if rr else float("nan")
                chosen.append((fi, "uniform", bc, bi))
                if len(chosen) >= max_per_event:
                    break

        export_map[ei] = chosen

    # Read frames sequentially and export those needed
    needed = set()
    for lst in export_map.values():
        for fi, _, _, _ in lst:
            needed.add(fi)

    fi = -1
    saved_paths: Dict[int, Path] = {}
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1
        if fi not in needed:
            continue
        outp = out_dir / f"evidence_f_{fi:06d}.jpg"
        # annotate frame with minimal header
        txt = f"fi={fi} t={fi/fps:.2f}s"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(outp), frame)
        saved_paths[fi] = outp

    cap.release()

    # build evidence list per event (return for DB insert)
    evidence_records: List[Dict[str, Any]] = []
    for ei, lst in export_map.items():
        for fi, note, bc, bi in lst:
            if fi in saved_paths:
                evidence_records.append({
                    "event_index": ei,
                    "frame_idx": fi,
                    "time_s": fi / fps,
                    "image_path": str(saved_paths[fi]),
                    "best_conf": None if math.isnan(bc) else float(bc),
                    "best_iou": None if math.isnan(bi) else float(bi),
                    "note": note
                })
    return evidence_records


def write_events_to_db(
    db_path: Path,
    video: str,
    model_tag: str,
    conf_thr: float,
    iou_thr: float,
    fps: float,
    events_pred: List[Event],
    events_gt: List[Event],
    per_frame_rows: List[Dict[str, Any]],
    save_evidence: bool,
    evidence_dir: Path,
    evidence_max_per_event: int,
):
    conn = init_event_db(db_path)
    cur = conn.cursor()

    all_events = events_pred + events_gt
    event_ids: List[int] = []

    for ev in all_events:
        cur.execute(
            """
            INSERT INTO events (source, video, model_tag, conf_thr, iou_thr, start_frame, end_frame, start_time_s, end_time_s, duration_s, created_at_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ev.source,
                str(video),
                model_tag,
                float(conf_thr),
                float(iou_thr),
                int(ev.start_frame),
                int(ev.end_frame),
                float(ev.start_time_s),
                float(ev.end_time_s),
                float(ev.duration_s),
                utc_now_str(),
            ),
        )
        event_ids.append(int(cur.lastrowid))

    conn.commit()

    if save_evidence and all_events:
        # save frames once and map to event indices
        evidence_records = save_event_evidence_frames(
            video=video,
            out_dir=evidence_dir,
            fps=fps,
            events=all_events,
            per_frame_rows=per_frame_rows,
            max_per_event=int(evidence_max_per_event),
        )

        # insert evidence
        for rec in evidence_records:
            ev_idx = int(rec["event_index"])
            if ev_idx < 0 or ev_idx >= len(event_ids):
                continue
            ev_id = event_ids[ev_idx]
            cur.execute(
                """
                INSERT INTO event_evidence (event_id, frame_idx, time_s, image_path, best_conf, best_iou, note, created_at_utc)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    ev_id,
                    int(rec["frame_idx"]),
                    float(rec["time_s"]),
                    str(rec["image_path"]),
                    rec.get("best_conf", None),
                    rec.get("best_iou", None),
                    rec.get("note", ""),
                    utc_now_str(),
                ),
            )
        conn.commit()

    conn.close()


# ------------------ evaluation main ------------------
def eval_metrics(
    video: str,
    pred_jsonl: str,
    out_dir: str,
    gt_csv: Optional[str],
    gt_segments: Optional[str],
    conf_thr: float,
    iou_thr: float,
    bucket_sec: float,
    export_misses: int,
    max_export: int,
    # event db
    event_db: Optional[str],
    model_tag: str,
    event_confirm: int,
    event_stop_miss: int,
    event_min_dur_s: float,
    event_evidence: int,
    event_evidence_max: int,
):
    fps, total, W, H = read_video_info(video)
    preds = read_pred_jsonl(pred_jsonl)

    gt_boxes_by_frame: Dict[int, List[Tuple[float, float, float, float]]] = {}
    gt_present = np.zeros((total,), dtype=np.uint8)

    if gt_csv:
        gt_boxes_by_frame = load_gt_csv(gt_csv)
        for fi, boxes in gt_boxes_by_frame.items():
            if 0 <= fi < total and boxes:
                gt_present[fi] = 1
    elif gt_segments:
        seg_map = load_segments_csv(gt_segments, fps=fps, total_frames=total)
        for fi in range(total):
            gt_present[fi] = seg_map.get(fi, 0)
    else:
        raise SystemExit("Evaluation needs GT. Pass --gt_csv or --gt_segments.")

    rows: List[Dict[str, Any]] = []
    TP = FP = TN = FN = 0
    iou_list = []
    conf_list = []

    fn_no_pred = 0
    fn_low_conf = 0
    fn_low_iou = 0
    fn_filtered_out = 0

    all_preds_for_ap = []

    bucket_frames = max(1, int(round(bucket_sec * fps)))
    bucket_stats: List[Dict[str, Any]] = []

    # also keep pred_present series for event extraction
    pred_present_series = np.zeros((total,), dtype=np.uint8)

    for fi in range(total):
        rec = preds.get(fi, None)
        pred_boxes = []
        raw_boxes = None
        filtered_boxes = None
        if rec is not None:
            pred_boxes = rec.get("pred", []) or []
            raw_boxes = rec.get("raw", None)
            filtered_boxes = rec.get("filtered", None)

        pred_norm = []
        for b in pred_boxes:
            try:
                box = clip_xyxy(b["x1"], b["y1"], b["x2"], b["y2"], W, H)
                c = float(b.get("conf", 1.0))
                pred_norm.append((box, c))
            except Exception:
                continue

        pred_present = 1 if any(c >= conf_thr for _, c in pred_norm) else 0
        pred_present_series[fi] = pred_present
        gt_p = int(gt_present[fi])

        if gt_p == 1 and pred_present == 1:
            TP += 1
        elif gt_p == 1 and pred_present == 0:
            FN += 1
        elif gt_p == 0 and pred_present == 1:
            FP += 1
        else:
            TN += 1

        best_iou = float("nan")
        best_conf = float("nan")
        n_gt = len(gt_boxes_by_frame.get(fi, [])) if gt_csv else (1 if gt_p == 1 else 0)
        n_pred = sum(1 for _, c in pred_norm if c >= conf_thr)

        if gt_csv and gt_p == 1:
            gts = [clip_xyxy(*g, W, H) for g in gt_boxes_by_frame.get(fi, [])]
            cands = [(box, c) for box, c in pred_norm if c >= conf_thr]
            if cands and gts:
                bi = 0.0
                bc = 0.0
                for gb in gts:
                    for pb, pc in cands:
                        v = iou_xyxy(pb, gb)
                        if v > bi:
                            bi = v
                            bc = pc
                best_iou = float(bi)
                best_conf = float(bc)
                if best_iou >= iou_thr:
                    iou_list.append(best_iou)
                    conf_list.append(best_conf)
            else:
                best_iou = 0.0
                best_conf = max([c for _, c in pred_norm], default=0.0)

            # FN breakdown on GT-positive frames
            if gt_p == 1:
                if len(pred_norm) == 0:
                    fn_no_pred += 1
                else:
                    max_c = max([c for _, c in pred_norm], default=0.0)
                    if max_c < conf_thr:
                        fn_low_conf += 1
                    else:
                        if best_iou < iou_thr:
                            fn_low_iou += 1

                # FILTERED_OUT if raw exists but filtered empty
                if raw_boxes is not None and filtered_boxes is not None:
                    def parse_boxes(arr):
                        out = []
                        if not isinstance(arr, list):
                            return out
                        for sb in arr:
                            box = None
                            if isinstance(sb, (list, tuple)) and len(sb) >= 4:
                                box = sb[:4]
                            elif isinstance(sb, dict):
                                for k in ("bbox", "xyxy", "box"):
                                    if k in sb and isinstance(sb[k], (list, tuple)) and len(sb[k]) >= 4:
                                        box = sb[k][:4]
                                        break
                            if box is None:
                                continue
                            out.append(clip_xyxy(box[0], box[1], box[2], box[3], W, H))
                        return out

                    raw_xyxy = parse_boxes(raw_boxes)
                    fil_xyxy = parse_boxes(filtered_boxes)
                    if raw_xyxy and (not fil_xyxy):
                        fn_filtered_out += 1

        # collect preds for AP
        if gt_csv:
            for box, c in pred_norm:
                all_preds_for_ap.append({"frame_idx": fi, "conf": float(c), "box": box})

        # reason label (for per_frame)
        if gt_p == 1 and pred_present == 0:
            if len(pred_norm) == 0:
                reason = "FN_NO_PRED"
            else:
                max_c = max([c for _, c in pred_norm], default=0.0)
                if max_c < conf_thr:
                    reason = "FN_LOW_CONF"
                else:
                    reason = "FN_LOW_IOU" if (gt_csv and best_iou < iou_thr) else "FN_OTHER"
        elif gt_p == 0 and pred_present == 1:
            reason = "FP"
        elif gt_p == 1 and pred_present == 1:
            reason = "TP"
        else:
            reason = "TN"

        rows.append(
            {
                "frame_idx": fi,
                "time_s": fi / fps,
                "gt_present": gt_p,
                "pred_present": pred_present,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "best_iou": best_iou,
                "best_conf": best_conf,
                "reason": reason,
            }
        )

    def safe_div(a, b):
        return float(a / b) if b else 0.0

    prec_f = safe_div(TP, TP + FP)
    rec_f = safe_div(TP, TP + FN)
    f1_f = safe_div(2 * prec_f * rec_f, (prec_f + rec_f)) if (prec_f + rec_f) > 0 else 0.0

    gt_smoke_frames = int(gt_present.sum())
    pred_smoke_frames = int(pred_present_series.sum())

    gt_minutes = gt_smoke_frames / fps / 60.0
    pred_minutes = pred_smoke_frames / fps / 60.0
    covered_minutes = TP / fps / 60.0
    missed_minutes = FN / fps / 60.0

    mean_iou = float(np.mean(iou_list)) if iou_list else float("nan")
    med_iou = float(np.median(iou_list)) if iou_list else float("nan")

    ap03 = ap05 = float("nan")
    if gt_csv:
        ap03 = compute_ap_single_class(all_preds_for_ap, gt_boxes_by_frame, iou_thr=0.30)
        ap05 = compute_ap_single_class(all_preds_for_ap, gt_boxes_by_frame, iou_thr=0.50)

    fn_total = max(gt_smoke_frames, 1)
    bd = {
        "FN_NO_PRED": fn_no_pred,
        "FN_LOW_CONF": fn_low_conf,
        "FN_LOW_IOU": fn_low_iou,
        "FN_FILTERED_OUT": fn_filtered_out,
    }
    bd_pct = {k: (100.0 * v / fn_total) for k, v in bd.items()}

    # bucket trend
    for start in range(0, total, bucket_frames):
        end = min(total, start + bucket_frames)
        part = rows[start:end]
        tp_b = sum(1 for r in part if r["gt_present"] == 1 and r["pred_present"] == 1)
        fn_b = sum(1 for r in part if r["gt_present"] == 1 and r["pred_present"] == 0)
        fp_b = sum(1 for r in part if r["gt_present"] == 0 and r["pred_present"] == 1)
        gt_b = sum(1 for r in part if r["gt_present"] == 1)
        pr_b = safe_div(tp_b, tp_b + fp_b)
        rc_b = safe_div(tp_b, tp_b + fn_b)
        ious_b = [
            r["best_iou"]
            for r in part
            if (not math.isnan(float(r["best_iou"])))
            and r["gt_present"] == 1
            and r["pred_present"] == 1
        ]
        miou_b = float(np.mean(ious_b)) if ious_b else float("nan")
        bucket_stats.append(
            {
                "t_start_s": start / fps,
                "t_end_s": end / fps,
                "gt_frames": gt_b,
                "tp_frames": tp_b,
                "fp_frames": fp_b,
                "fn_frames": fn_b,
                "precision": pr_b,
                "recall": rc_b,
                "mean_iou": miou_b,
            }
        )

    outp = Path(out_dir)
    ensure_dir(outp)

    # save CSVs
    if pd is not None:
        pd.DataFrame(rows).to_csv(outp / "per_frame.csv", index=False)
        pd.DataFrame(bucket_stats).to_csv(outp / "per_bucket.csv", index=False)
    else:
        import csv

        with open(outp / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(outp / "per_bucket.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bucket_stats[0].keys()))
            w.writeheader()
            w.writerows(bucket_stats)

    summary = {
        "video": str(video),
        "fps": float(fps),
        "total_frames": int(total),
        "resolution": {"W": int(W), "H": int(H)},
        "gt_smoke_frames": int(gt_smoke_frames),
        "pred_smoke_frames": int(pred_smoke_frames),
        "gt_smoke_minutes": float(gt_minutes),
        "pred_smoke_minutes": float(pred_minutes),
        "TP_frames": int(TP),
        "FP_frames": int(FP),
        "FN_frames": int(FN),
        "TN_frames": int(TN),
        "frame_precision": float(prec_f),
        "frame_recall": float(rec_f),
        "frame_f1": float(f1_f),
        "conf_thr": float(conf_thr),
        "iou_thr": float(iou_thr),
        "mean_iou_matched": mean_iou,
        "median_iou_matched": med_iou,
        "AP@0.30": ap03,
        "AP@0.50": ap05,
        "FN_breakdown_counts": bd,
        "FN_breakdown_pct": bd_pct,
        "covered_minutes": float(covered_minutes),
        "missed_minutes": float(missed_minutes),
    }

    with open(outp / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if pd is not None:
        pd.DataFrame([summary]).to_csv(outp / "summary.csv", index=False)

    # export example frames for misses/FP
    if export_misses:
        exp_dir = outp / "export_frames"
        ensure_dir(exp_dir)
        for k in ("FN_NO_PRED", "FN_LOW_CONF", "FN_LOW_IOU", "FP"):
            ensure_dir(exp_dir / k)

        cap = cv2.VideoCapture(video)
        exports = {"FN_NO_PRED": 0, "FN_LOW_CONF": 0, "FN_LOW_IOU": 0, "FP": 0}
        fi = -1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            fi += 1
            r = rows[fi]
            reason = r["reason"]
            if reason in exports and exports[reason] < max_export:
                txt = f"fi={fi} t={r['time_s']:.2f}s {reason} iou={r['best_iou']} conf={r['best_conf']}"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
                cv2.putText(
                    frame, txt, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.imwrite(str(exp_dir / reason / f"f_{fi:06d}.jpg"), frame)
                exports[reason] += 1
        cap.release()

    # ---------- EVENT DB (pred + optionally GT) ----------
    if event_db:
        # build predicted events
        pred_events_idx = build_events_from_binary(
            present=pred_present_series,
            fps=fps,
            confirm_frames=event_confirm,
            stop_miss_frames=event_stop_miss,
            min_dur_s=event_min_dur_s,
        )
        events_pred: List[Event] = []
        for s, e in pred_events_idx:
            events_pred.append(
                Event(
                    source="PRED",
                    start_frame=int(s),
                    end_frame=int(e),
                    start_time_s=float(s / fps),
                    end_time_s=float(e / fps),
                    duration_s=float((e - s + 1) / fps),
                    meta={},
                )
            )

        # build GT events (for reference) from gt_present
        gt_events_idx = build_events_from_binary(
            present=gt_present.astype(np.uint8),
            fps=fps,
            confirm_frames=1,
            stop_miss_frames=1,
            min_dur_s=0.0,
        )
        events_gt: List[Event] = []
        for s, e in gt_events_idx:
            events_gt.append(
                Event(
                    source="GT",
                    start_frame=int(s),
                    end_frame=int(e),
                    start_time_s=float(s / fps),
                    end_time_s=float(e / fps),
                    duration_s=float((e - s + 1) / fps),
                    meta={},
                )
            )

        db_path = Path(event_db)
        evidence_dir = outp / "event_evidence"
        write_events_to_db(
            db_path=db_path,
            video=video,
            model_tag=model_tag,
            conf_thr=conf_thr,
            iou_thr=iou_thr,
            fps=fps,
            events_pred=events_pred,
            events_gt=events_gt,
            per_frame_rows=rows,
            save_evidence=bool(event_evidence),
            evidence_dir=evidence_dir,
            evidence_max_per_event=int(event_evidence_max),
        )

        # also drop an event_summary.json
        evsum = {
            "pred_event_count": len(events_pred),
            "gt_event_count": len(events_gt),
            "pred_total_duration_s": float(sum(e.duration_s for e in events_pred)),
            "gt_total_duration_s": float(sum(e.duration_s for e in events_gt)),
            "event_confirm_frames": int(event_confirm),
            "event_stop_miss_frames": int(event_stop_miss),
            "event_min_dur_s": float(event_min_dur_s),
            "event_db": str(db_path),
        }
        with open(outp / "event_summary.json", "w", encoding="utf-8") as f:
            json.dump(evsum, f, ensure_ascii=False, indent=2)

    print("EVAL OK ->", str(outp))
    print("SUMMARY:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ------------------ GT helper ------------------
def make_gt(video: str, frames_out: str, sample_fps: float, csv_out: str):
    fps, total, W, H = read_video_info(video)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    out_dir = Path(frames_out)
    ensure_dir(out_dir)

    step = 1 if not sample_fps or sample_fps <= 0 else max(1, int(round(fps / float(sample_fps))))

    rows = []
    fi = -1
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        fi += 1
        if fi % step != 0:
            continue
        fn = f"frame_{fi:06d}.jpg"
        fp = out_dir / fn
        cv2.imwrite(str(fp), frame)
        saved += 1
        rows.append({"frame_idx": fi, "image": str(fp), "x1": "", "y1": "", "x2": "", "y2": "", "class": "cigarette"})
        if saved % 200 == 0:
            print(f"[GT] saved={saved} last_frame={fi}")

    cap.release()

    if pd is not None:
        pd.DataFrame(rows).to_csv(csv_out, index=False)
    else:
        import csv
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("FRAMES ->", frames_out)
    print("GT TEMPLATE CSV ->", csv_out)
    print(f"Video fps={fps:.2f} total_frames={total} sample_fps={sample_fps} step={step} exported_frames={saved}")


# ------------------ CLI ------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="video -> predictions jsonl")
    p_pred.add_argument("--source", choices=["yolo", "api"], required=True)
    p_pred.add_argument("--video", required=True)
    p_pred.add_argument("--out", required=True)
    p_pred.add_argument("--sample_fps", type=float, default=0.0, help="0 => all frames; 5 => sample at 5 fps")

    # YOLO args
    p_pred.add_argument("--weights", default="models/best.pt")
    p_pred.add_argument("--imgsz", type=int, default=1280)
    p_pred.add_argument("--conf", type=float, default=0.05)
    p_pred.add_argument("--iou", type=float, default=0.45)
    p_pred.add_argument("--max_det", type=int, default=50)

    # API args
    p_pred.add_argument("--api", default="http://127.0.0.1:5000")
    p_pred.add_argument("--camera_id", default="1")
    p_pred.add_argument("--jpeg_quality", type=int, default=80)

    p_gt = sub.add_parser("make-gt", help="video -> frame export + gt template csv")
    p_gt.add_argument("--video", required=True)
    p_gt.add_argument("--frames_out", required=True)
    p_gt.add_argument("--sample_fps", type=float, default=5.0)
    p_gt.add_argument("--csv_out", required=True)

    p_eval = sub.add_parser("eval", help="gt + pred -> metrics (+ optional event db)")
    p_eval.add_argument("--video", required=True)
    p_eval.add_argument("--pred", required=True)
    p_eval.add_argument("--out_dir", required=True)

    p_eval.add_argument("--gt_csv", default="", help="GT bbox CSV (frame_idx,x1,y1,x2,y2)")
    p_eval.add_argument("--gt_segments", default="", help="GT segments CSV (start_s,end_s)")

    p_eval.add_argument("--conf_thr", type=float, default=0.10, help="frame-level present threshold")
    p_eval.add_argument("--iou_thr", type=float, default=0.30, help="IoU match threshold (if bbox GT is available)")
    p_eval.add_argument("--bucket_sec", type=float, default=10.0, help="trend window in seconds")

    p_eval.add_argument("--export_misses", type=int, default=1, help="1 => export example FN/FP frames")
    p_eval.add_argument("--max_export", type=int, default=200)

    # event DB options
    p_eval.add_argument("--event_db", default="", help="SQLite DB path (e.g. outputs/metrics/events.db). Nothing is written if empty.")
    p_eval.add_argument("--model_tag", default="best.pt", help="Model tag written to the DB (e.g. trial_39)")
    p_eval.add_argument("--event_confirm", type=int, default=3, help="Consecutive positive frames needed to open an event")
    p_eval.add_argument("--event_stop_miss", type=int, default=8, help="Consecutive negative frames needed to close an event")
    p_eval.add_argument("--event_min_dur_s", type=float, default=0.8, help="Minimum event duration (s)")
    p_eval.add_argument("--event_evidence", type=int, default=1, help="1 => save event evidence frames")
    p_eval.add_argument("--event_evidence_max", type=int, default=8, help="max evidence frames per event")

    args = ap.parse_args()

    if args.cmd == "predict":
        if args.source == "yolo":
            predict_yolo(
                video=args.video,
                weights=args.weights,
                out_jsonl=args.out,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                sample_fps=args.sample_fps,
                max_det=args.max_det,
            )
        else:
            predict_api(
                video=args.video,
                api=args.api,
                out_jsonl=args.out,
                jpeg_quality=args.jpeg_quality,
                sample_fps=args.sample_fps,
                camera_id=args.camera_id,
            )

    elif args.cmd == "make-gt":
        make_gt(video=args.video, frames_out=args.frames_out, sample_fps=args.sample_fps, csv_out=args.csv_out)

    elif args.cmd == "eval":
        gt_csv = args.gt_csv.strip() or None
        gt_seg = args.gt_segments.strip() or None
        event_db = args.event_db.strip() or None

        eval_metrics(
            video=args.video,
            pred_jsonl=args.pred,
            out_dir=args.out_dir,
            gt_csv=gt_csv,
            gt_segments=gt_seg,
            conf_thr=args.conf_thr,
            iou_thr=args.iou_thr,
            bucket_sec=args.bucket_sec,
            export_misses=args.export_misses,
            max_export=args.max_export,
            event_db=event_db,
            model_tag=args.model_tag,
            event_confirm=args.event_confirm,
            event_stop_miss=args.event_stop_miss,
            event_min_dur_s=args.event_min_dur_s,
            event_evidence=args.event_evidence,
            event_evidence_max=args.event_evidence_max,
        )


if __name__ == "__main__":
    main()