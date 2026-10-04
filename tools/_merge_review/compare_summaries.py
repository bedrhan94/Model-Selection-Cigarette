# tools/compare_summaries.py
# -*- coding: utf-8 -*-
import json
from pathlib import Path

RUNS = [
    "metrics_mini_audit_night_run1",
    "metrics_mini_audit_night_thr012",
    "metrics_mini_audit_night_thr015",
    "metrics_mini_audit_night_thr018",
]

base = Path("outputs")

def read(run):
    p = base / run / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    fps = float(d["fps"])
    return {
        "run": run,
        "thr": float(d.get("conf_thr", 0.0)),
        "P": float(d.get("frame_precision", 0.0)),
        "R": float(d.get("frame_recall", 0.0)),
        "FP_s": float(d.get("FP_frames", 0.0)) / fps,
        "FN_s": float(d.get("FN_frames", 0.0)) / fps,
        "pred_smoke_s": float(d.get("pred_smoke_minutes", 0.0)) * 60.0,
    }

rows = [read(r) for r in RUNS]
rows = [r for r in rows if r is not None]
rows.sort(key=lambda x: x["thr"])

print("RUN\tthr\tP\tR\tFP_s\tFN_s\tpred_smoke_s")
for x in rows:
    print(f"{x['run']}\t{x['thr']:.2f}\t{x['P']:.3f}\t{x['R']:.3f}\t{x['FP_s']:.2f}\t{x['FN_s']:.2f}\t{x['pred_smoke_s']:.1f}")
