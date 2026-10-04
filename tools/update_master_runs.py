# tools/update_master_runs.py
# -*- coding: utf-8 -*-

import argparse
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import csv


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=None):
    try:
        return int(float(x))
    except Exception:
        return default


def read_summary_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def read_event_stats(db_path: Path):
    out = {
        "pred_event_count": None,
        "pred_event_total_s": None,
        "gt_event_count": None,
        "gt_event_total_s": None,
    }
    if not db_path.exists():
        return out
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("SELECT source, COUNT(*), COALESCE(SUM(duration_s),0) FROM events GROUP BY source;")
        rows = cur.fetchall()
        con.close()
        for src, cnt, tot in rows:
            srcu = str(src).upper()
            if srcu == "PRED":
                out["pred_event_count"] = int(cnt)
                out["pred_event_total_s"] = float(tot)
            elif srcu == "GT":
                out["gt_event_count"] = int(cnt)
                out["gt_event_total_s"] = float(tot)
        return out
    except Exception:
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", default="outputs/04_metrics_active")
    ap.add_argument("--out_csv", default="outputs/06_summary_tables/MASTER_runs_summary.csv")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    runs = sorted(
        [d for d in outputs_dir.rglob("metrics_*") if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    rows = []
    for run_dir in runs:
        sj = run_dir / "summary.json"
        if not sj.exists():
            continue

        s = read_summary_json(sj)
        fps = safe_float(s.get("fps"), 0.0) or 0.0

        fp_frames = safe_int(s.get("FP_frames"), 0) or 0
        fn_frames = safe_int(s.get("FN_frames"), 0) or 0
        fp_s = (fp_frames / fps) if fps > 0 else None
        fn_s = (fn_frames / fps) if fps > 0 else None

        ev = read_event_stats(run_dir / "metrics_events.db")

        mtime = datetime.fromtimestamp(sj.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        res = s.get("resolution") or {}
        W = safe_int(res.get("W"), None)
        H = safe_int(res.get("H"), None)

        row = {
            "run_dir": str(run_dir.relative_to(outputs_dir)).replace("\\", "/"),
            "run_mtime": mtime,
            "video": s.get("video"),
            "fps": safe_float(s.get("fps")),
            "total_frames": safe_int(s.get("total_frames")),
            "W": W,
            "H": H,
            "conf_thr": safe_float(s.get("conf_thr")),
            "iou_thr": safe_float(s.get("iou_thr")),
            "TP_frames": safe_int(s.get("TP_frames")),
            "FP_frames": fp_frames,
            "FN_frames": fn_frames,
            "TN_frames": safe_int(s.get("TN_frames")),
            "frame_precision": safe_float(s.get("frame_precision")),
            "frame_recall": safe_float(s.get("frame_recall")),
            "frame_f1": safe_float(s.get("frame_f1")),
            "FP_s": fp_s,
            "FN_s": fn_s,
            "gt_smoke_minutes": safe_float(s.get("gt_smoke_minutes")),
            "pred_smoke_minutes": safe_float(s.get("pred_smoke_minutes")),
            "covered_minutes": safe_float(s.get("covered_minutes")),
            "missed_minutes": safe_float(s.get("missed_minutes")),
            "AP@0.30": safe_float(s.get("AP@0.30")),
            "AP@0.50": safe_float(s.get("AP@0.50")),
            "mean_iou_matched": safe_float(s.get("mean_iou_matched")),
            "median_iou_matched": safe_float(s.get("median_iou_matched")),
            "pred_event_count": ev["pred_event_count"],
            "pred_event_total_s": ev["pred_event_total_s"],
            "gt_event_count": ev["gt_event_count"],
            "gt_event_total_s": ev["gt_event_total_s"],
        }
        rows.append(row)

    if not rows:
        print(f"No metrics_* runs with summary.json found under {outputs_dir}/")
        return

    cols = [
        "run_dir","run_mtime","video","fps","total_frames","W","H",
        "conf_thr","iou_thr",
        "TP_frames","FP_frames","FN_frames","TN_frames",
        "frame_precision","frame_recall","frame_f1",
        "FP_s","FN_s",
        "gt_smoke_minutes","pred_smoke_minutes","covered_minutes","missed_minutes",
        "AP@0.30","AP@0.50","mean_iou_matched","median_iou_matched",
        "pred_event_count","pred_event_total_s","gt_event_count","gt_event_total_s",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

    print("OK ->", str(out_csv))
    print("rows =", len(rows))


if __name__ == "__main__":
    main()
