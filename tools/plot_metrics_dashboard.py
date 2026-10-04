# tools/plot_metrics_dashboard.py
# -*- coding: utf-8 -*-
"""
Metrics Dashboard Generator (PNG + optional PDF)
================================================
Reads a run folder (summary/per_frame/per_bucket + event DB) and plots it.

Expected files (eval output):
- summary.json  (or summary.csv)
- per_bucket.csv
- per_frame.csv
- metrics_events.db (optional)
- event_summary.json (optional)

Outputs (written into run_dir):
- dashboard_01_pr_rc_iou.png      (per-bucket precision/recall/mean_iou)
- dashboard_02_fp_fn.png          (per-bucket FP/FN/GT)
- dashboard_03_timeline.png       (frame timeline: GT vs Pred + reason heat)
- dashboard_04_distributions.png  (conf/n_pred histograms if available)
- dashboard_05_events.png         (event timeline + duration histogram if a DB exists)
- dashboard_compare.png           (compare mode)

Usage:
  py -3.10 tools/plot_metrics_dashboard.py --run_dir outputs/metrics_mini_audit_night_run1
  py -3.10 tools/plot_metrics_dashboard.py --compare outputs/runA outputs/runB outputs/runC

Note:
- no seaborn, matplotlib only
"""

import argparse
import json
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_summary(run_dir: Path):
    js = run_dir / "summary.json"
    cs = run_dir / "summary.csv"
    if js.exists():
        return json.loads(js.read_text(encoding="utf-8"))
    if cs.exists():
        df = pd.read_csv(cs)
        return df.iloc[0].to_dict()
    raise FileNotFoundError(f"summary.json/csv not found: {run_dir}")


def _safe_float(x):
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bucket_pr_rc_iou(run_dir: Path, out_dir: Path):
    pb = run_dir / "per_bucket.csv"
    if not pb.exists():
        return None
    df = pd.read_csv(pb)
    # normalize columns
    for col in ["t_start_s", "t_end_s", "precision", "recall", "mean_iou", "fp_frames", "fn_frames", "gt_frames", "tp_frames"]:
        if col not in df.columns:
            df[col] = np.nan

    tmid = (df["t_start_s"].values + df["t_end_s"].values) / 2.0

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.plot(tmid, df["precision"].values, label="precision (bucket)")
    ax.plot(tmid, df["recall"].values, label="recall (bucket)")

    # mean_iou only if not NaN
    if np.isfinite(df["mean_iou"].values).any():
        ax.plot(tmid, df["mean_iou"].values, label="mean_iou (bucket)")

    ax.set_title("Bucket Trend: Precision / Recall / Mean IoU")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # annotate summary headline
    s = _read_summary(run_dir)
    text = f"TP={int(float(s.get('TP_frames',0)))} FP={int(float(s.get('FP_frames',0)))} FN={int(float(s.get('FN_frames',0)))} | P={float(s.get('frame_precision',0)):.3f} R={float(s.get('frame_recall',0)):.3f}"
    ax.text(0.01, 0.02, text, transform=ax.transAxes)

    ax.legend(loc="lower right")
    out = out_dir / "dashboard_01_pr_rc_iou.png"
    _save_fig(fig, out)
    return out


def plot_bucket_fp_fn(run_dir: Path, out_dir: Path):
    pb = run_dir / "per_bucket.csv"
    if not pb.exists():
        return None
    df = pd.read_csv(pb)
    for col in ["t_start_s", "t_end_s", "fp_frames", "fn_frames", "gt_frames", "tp_frames"]:
        if col not in df.columns:
            df[col] = 0

    tmid = (df["t_start_s"].values + df["t_end_s"].values) / 2.0

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.plot(tmid, df["gt_frames"].values, label="GT frames in bucket")
    ax.plot(tmid, df["tp_frames"].values, label="TP frames in bucket")
    ax.plot(tmid, df["fp_frames"].values, label="FP frames in bucket")
    ax.plot(tmid, df["fn_frames"].values, label="FN frames in bucket")

    ax.set_title("Bucket Trend: GT / TP / FP / FN (frames)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frames per bucket")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    out = out_dir / "dashboard_02_fp_fn.png"
    _save_fig(fig, out)
    return out


def plot_timeline_gt_pred(run_dir: Path, out_dir: Path, max_points=6000):
    pf = run_dir / "per_frame.csv"
    if not pf.exists():
        return None
    df = pd.read_csv(pf)
    # downsample if too large for plot
    n = len(df)
    step = max(1, n // max_points)
    df2 = df.iloc[::step].copy()

    t = df2["time_s"].values if "time_s" in df2.columns else df2["frame_idx"].values
    gt = df2["gt_present"].values if "gt_present" in df2.columns else np.zeros(len(df2))
    pr = df2["pred_present"].values if "pred_present" in df2.columns else np.zeros(len(df2))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # plot as step lines
    ax.step(t, gt, where="post", label="GT present (0/1)")
    ax.step(t, pr, where="post", label="Pred present (0/1)")

    # reason heat as scatter (optional)
    if "reason" in df2.columns:
        reason = df2["reason"].astype(str).values
        # map reason to y offset
        y = np.full_like(t, 0.0, dtype=float)
        # show only FP/FN points
        mask_fp = reason == "FP"
        mask_fn = np.char.startswith(reason.astype(str), "FN")
        ax.scatter(t[mask_fp], (pr[mask_fp] + 0.05), s=10, label="FP points")
        ax.scatter(t[mask_fn], (gt[mask_fn] + 0.10), s=10, label="FN points")

    ax.set_title("Timeline: GT vs Pred (downsampled)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("present")
    ax.set_ylim(-0.1, 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    out = out_dir / "dashboard_03_timeline.png"
    _save_fig(fig, out)
    return out


def plot_distributions(run_dir: Path, out_dir: Path):
    pf = run_dir / "per_frame.csv"
    if not pf.exists():
        return None
    df = pd.read_csv(pf)

    has_conf = "best_conf" in df.columns and df["best_conf"].notna().any()
    has_npred = "n_pred" in df.columns and df["n_pred"].notna().any()

    if not has_conf and not has_npred:
        return None

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    if has_conf:
        x = df["best_conf"].dropna().values
        ax1.hist(x, bins=40)
        ax1.set_title("best_conf distribution (non-NaN)")
        ax1.set_xlabel("conf")
        ax1.set_ylabel("count")
        ax1.grid(True, alpha=0.2)
    else:
        ax1.text(0.5, 0.5, "no best_conf (GT may be segments)", ha="center", va="center")
        ax1.axis("off")

    if has_npred:
        x = df["n_pred"].dropna().values.astype(int)
        # integer bins
        bins = np.arange(x.min(), x.max() + 2) - 0.5
        ax2.hist(x, bins=bins)
        ax2.set_title("n_pred distribution")
        ax2.set_xlabel("n_pred")
        ax2.set_ylabel("count")
        ax2.grid(True, alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "no n_pred", ha="center", va="center")
        ax2.axis("off")

    out = out_dir / "dashboard_04_distributions.png"
    _save_fig(fig, out)
    return out


def plot_events_from_db(run_dir: Path, out_dir: Path):
    db = run_dir / "metrics_events.db"
    if not db.exists():
        return None

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    # fetch events
    cur.execute("SELECT id, source, start_time_s, end_time_s, duration_s FROM events ORDER BY id ASC;")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return None

    # separate pred vs gt
    pred = [r for r in rows if str(r[1]).upper() == "PRED"]
    gt = [r for r in rows if str(r[1]).upper() == "GT"]

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    def _plot_timeline(ax, events, label):
        if not events:
            ax.text(0.5, 0.5, f"{label}: no events", ha="center", va="center")
            ax.set_axis_off()
            return
        for i, (_, _, s, e, dur) in enumerate(events):
            ax.plot([s, e], [i, i], linewidth=4)
        ax.set_title(f"Event timeline ({label})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("event index")
        ax.grid(True, alpha=0.2)

    _plot_timeline(ax1, pred, "PRED")
    _plot_timeline(ax2, gt, "GT")

    out = out_dir / "dashboard_05_events.png"
    _save_fig(fig, out)

    # durations histogram
    durs = [float(r[4]) for r in pred] if pred else []
    if durs:
        fig2 = plt.figure(figsize=(10, 5))
        ax = fig2.add_subplot(111)
        ax.hist(durs, bins=30)
        ax.set_title("PRED event duration distribution (s)")
        ax.set_xlabel("duration (s)")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        out2 = out_dir / "dashboard_05b_event_durations.png"
        _save_fig(fig2, out2)

    return out


def compare_runs(run_dirs, out_path: Path):
    # Compare summary points: precision/recall and FP/FN minutes
    rows = []
    for rd in run_dirs:
        s = _read_summary(rd)
        rows.append({
            "run": rd.name,
            "precision": float(s.get("frame_precision", 0.0)),
            "recall": float(s.get("frame_recall", 0.0)),
            "FP_frames": float(s.get("FP_frames", 0.0)),
            "FN_frames": float(s.get("FN_frames", 0.0)),
            "missed_minutes": float(s.get("missed_minutes", 0.0)),
            "covered_minutes": float(s.get("covered_minutes", 0.0)),
            "pred_smoke_minutes": float(s.get("pred_smoke_minutes", 0.0)),
        })
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    ax.scatter(df["recall"], df["precision"])
    for _, r in df.iterrows():
        ax.annotate(r["run"], (r["recall"], r["precision"]), textcoords="offset points", xytext=(6, 4))

    ax.set_title("Compare Runs: Precision vs Recall (frame-level)")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    _save_fig(fig, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="outputs/04_metrics_active/.../metrics_*")
    ap.add_argument("--compare", nargs="*", default=[], help="run dirs for compare plot")
    args = ap.parse_args()

    if args.compare:
        rds = [Path(p) for p in args.compare]
        out = Path("outputs/07_master_charts/dashboard_compare.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        compare_runs(rds, out)
        print("OK ->", out)
        return

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    out_dir = run_dir  # write into run folder

    made = []
    x = plot_bucket_pr_rc_iou(run_dir, out_dir)
    if x: made.append(x)
    x = plot_bucket_fp_fn(run_dir, out_dir)
    if x: made.append(x)
    x = plot_timeline_gt_pred(run_dir, out_dir)
    if x: made.append(x)
    x = plot_distributions(run_dir, out_dir)
    if x: made.append(x)
    x = plot_events_from_db(run_dir, out_dir)
    if x: made.append(x)

    if made:
        print("Dashboard PNGs written:")
        for p in made:
            print(" -", p)
    else:
        print("No charts produced (files may be missing).")


if __name__ == "__main__":
    main()
