import pandas as pd, numpy as np, matplotlib.pyplot as plt, re
from pathlib import Path

MASTER = Path("outputs/06_summary_tables/MASTER_runs_summary.csv")
OUTDIR = Path("outputs/07_master_charts")
OUTDIR.mkdir(parents=True, exist_ok=True)

def safe_name(s):
    s = str(s) if s is not None else "unknown"
    s = s.replace("\\","_").replace("/","_").replace(":","_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+","_", s)
    return s[:120]

df = pd.read_csv(MASTER)

num_cols = ["fps","conf_thr","frame_precision","frame_recall","FP_s","FN_s","FP_frames","FN_frames"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if ("FP_s" not in df.columns or df["FP_s"].isna().all()) and "FP_frames" in df.columns and "fps" in df.columns:
    df["FP_s"] = df["FP_frames"] / df["fps"]
if ("FN_s" not in df.columns or df["FN_s"].isna().all()) and "FN_frames" in df.columns and "fps" in df.columns:
    df["FN_s"] = df["FN_frames"] / df["fps"]

# 1) Precision vs Recall
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.scatter(df["frame_recall"], df["frame_precision"])
ax.set_title("All Runs: Precision vs Recall (frame-level)")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xlim(0,1.02); ax.set_ylim(0,1.02)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR/"01_precision_vs_recall.png", dpi=160)
plt.close(fig)

# 2) FP_s vs FN_s
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.scatter(df["FN_s"], df["FP_s"])
ax.set_title("All Runs: FP seconds vs FN seconds")
ax.set_xlabel("FN_s (seconds)"); ax.set_ylabel("FP_s (seconds)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUTDIR/"02_fp_s_vs_fn_s.png", dpi=160)
plt.close(fig)

# Per-video: recall/precision/FP_s/FN_s vs conf_thr
for v in df["video"].dropna().unique():
    g = df[df["video"]==v].copy()
    if g.empty or g["conf_thr"].isna().all():
        continue
    g = g.sort_values("conf_thr")
    vname = safe_name(v)

    def save_line(ycol, title, fname, ylim01=False):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        ax.plot(g["conf_thr"], g[ycol], marker="o")
        ax.set_title(f"{title}\n{v}")
        ax.set_xlabel("conf_thr"); ax.set_ylabel(ycol)
        if ylim01:
            ax.set_ylim(0,1.02)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTDIR/fname, dpi=160)
        plt.close(fig)

    save_line("frame_recall","Recall vs conf_thr", f"video_{vname}_recall_vs_thr.png", True)
    save_line("frame_precision","Precision vs conf_thr", f"video_{vname}_precision_vs_thr.png", True)
    save_line("FP_s","FP_s vs conf_thr", f"video_{vname}_FPs_vs_thr.png", False)
    save_line("FN_s","FN_s vs conf_thr", f"video_{vname}_FNs_vs_thr.png", False)

print("OK ->", OUTDIR)
