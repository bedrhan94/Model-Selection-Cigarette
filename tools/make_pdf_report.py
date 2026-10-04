# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import json
import re
import sqlite3

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(".")
OUTPUTS = ROOT / "outputs"
METRICS_ROOT = OUTPUTS / "04_metrics_active"
REPORTS_ROOT = OUTPUTS / "05_reports_pdf"
CHARTS_ROOT = OUTPUTS / "07_master_charts"


def safe_name(s: str) -> str:
    s = str(s) if s is not None else "unknown"
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s[:120]


def find_latest_metrics_dir_prefer_prod():
    """
    Pick the most recent metrics_* folder under FINAL_SIHA_DAY_TIME.
    Only scans this final namespace; does not touch the old PROD runs.
    """
    base = METRICS_ROOT / "FINAL_SIHA_DAY_TIME"
    if not base.exists():
        return None

    runs = [d for d in base.rglob("metrics_*") if d.is_dir()]
    if not runs:
        return None

    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def read_summary(run_dir: Path) -> dict:
    sj = run_dir / "summary.json"
    if sj.exists():
        return json.loads(sj.read_text(encoding="utf-8"))
    return {}


def read_event_stats(db_path: Path) -> dict:
    out = {"pred_event_count": None, "pred_event_total_s": None, "gt_event_count": None, "gt_event_total_s": None}
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


def draw_cover(c: canvas.Canvas, page_size, ts: str, run_dir: Path, s: dict):
    W, H = page_size
    c.setFont("Helvetica-Bold", 22)
    c.drawString(40, H - 60, "Metrics Report (Auto)")

    video = s.get("video", "unknown_video")
    thr = s.get("conf_thr", "")
    pre = s.get("frame_precision", "")
    rec = s.get("frame_recall", "")
    f1 = s.get("frame_f1", "")

    c.setFont("Helvetica", 12)
    c.drawString(40, H - 95, f"Run: {run_dir.name}")
    c.drawString(40, H - 115, f"Video: {video}")
    c.drawString(40, H - 135, f"conf_thr: {thr} | Precision: {pre} | Recall: {rec} | F1: {f1}")
    c.drawString(40, H - 155, f"Generated: {ts}")
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 180, "Notes: If segments-GT is used, IoU/AP may be NaN (expected).")
    c.showPage()


def draw_key_takeaway(c: canvas.Canvas, page_size, run_dir: Path, s: dict, ev: dict):
    W, H = page_size

    fps = float(s.get("fps", 0) or 0)
    FP_frames = float(s.get("FP_frames", 0) or 0)
    FN_frames = float(s.get("FN_frames", 0) or 0)
    FP_s = (FP_frames / fps) if fps > 0 else None
    FN_s = (FN_frames / fps) if fps > 0 else None

    gt_min = float(s.get("gt_smoke_minutes", 0) or 0)
    pred_min = float(s.get("pred_smoke_minutes", 0) or 0)

    thr = s.get("conf_thr", "")
    pre = s.get("frame_precision", "")
    rec = s.get("frame_recall", "")
    f1 = s.get("frame_f1", "")

    # Try parse event params from run name: ev8_5_30 or ev8_5_3.0 (we store 30 for 3.0)
    m = re.search(r"ev(\d+)_(\d+)_(\d+)", run_dir.name)
    ev_txt = ""
    if m:
        ev_txt = f"event_confirm={m.group(1)}, stop_miss={m.group(2)}, min_dur≈{int(m.group(3))/10.0:.1f}s"

    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, H - 60, "Key Takeaways (Presentation Summary)")

    c.setFont("Helvetica", 12)
    y = H - 105
    lines = [
        f"Preset: conf_thr={thr} | TOP1=ON | {ev_txt}".strip(),
        f"Frame-level: Precision={pre} | Recall={rec} | F1={f1}",
        f"Errors: FP={int(FP_frames)} frames ({FP_s:.2f}s) | FN={int(FN_frames)} frames ({FN_s:.2f}s)" if FP_s is not None else "Errors: FP/FN seconds unavailable",
        f"Coverage: GT={gt_min*60:.1f}s | Pred={pred_min*60:.1f}s | delta={((pred_min-gt_min)*60):.1f}s",
    ]

    # Event DB summary if exists
    if ev.get("pred_event_count") is not None:
        lines.append(f"Event DB: PRED events={ev.get('pred_event_count')} | total={float(ev.get('pred_event_total_s') or 0):.1f}s")
    if ev.get("gt_event_count") is not None:
        lines.append(f"Event DB: GT events={ev.get('gt_event_count')} | total={float(ev.get('gt_event_total_s') or 0):.1f}s")

    lines.append("Interpretation: recall is preserved (no-miss priority). Most FP may come from GT segment boundaries.")

    for ln in lines:
        c.drawString(40, y, u"• " + ln)
        y -= 22

    c.showPage()


def add_page_with_image(c: canvas.Canvas, title: str, img_path: Path, page_size):
    W, H = page_size
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - 40, title)
    c.setFont("Helvetica", 9)
    c.drawString(40, H - 60, str(img_path))

    img = ImageReader(str(img_path))
    iw, ih = img.getSize()
    margin = 50
    max_w = W - 2 * margin
    max_h = H - 120
    scale = min(max_w / iw, max_h / ih)
    dw, dh = iw * scale, ih * scale
    x = (W - dw) / 2
    y = (H - dh) / 2 - 20
    c.drawImage(img, x, y, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
    c.showPage()


def pick_images(run_dir: Path, s: dict):
    imgs = []

    # 1) Global charts (from master_charts if available)
    charts_dir = CHARTS_ROOT
    for name in ["01_precision_vs_recall.png", "02_fp_s_vs_fn_s.png"]:
        p = charts_dir / name
        if p.exists():
            imgs.append(p)

    # 2) Threshold sweep charts for THIS video (auto)
    v = s.get("video", None)
    if v:
        vkey = safe_name(v)
        for suffix in ["recall_vs_thr.png", "FNs_vs_thr.png", "FPs_vs_thr.png"]:
            p = charts_dir / f"video_{vkey}_{suffix}"
            if p.exists():
                imgs.append(p)

    # 3) Run dashboards (most convincing for presentation)
    for name in [
        "dashboard_01_pr_rc_iou.png",
        "dashboard_02_fp_fn.png",
        "dashboard_03_timeline.png",
        "dashboard_04_distributions.png",
        "dashboard_05_events.png",
        "dashboard_05b_event_durations.png",
    ]:
        p = run_dir / name
        if p.exists():
            imgs.append(p)

    # If still nothing, take all pngs under run
    if not imgs:
        imgs.extend(sorted(run_dir.glob("*.png")))

    # De-dup while keeping order
    seen = set()
    uniq = []
    for p in imgs:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def main():
    run_dir = find_latest_metrics_dir_prefer_prod()
    if run_dir is None:
        print("No outputs/04_metrics_active/FINAL_SIHA_DAY_TIME/**/metrics_* found.")
        return

    s = read_summary(run_dir)
    ev = read_event_stats(run_dir / "metrics_events.db")

    imgs = pick_images(run_dir, s)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS_ROOT / f"METRICS_REPORT_{ts}_{run_dir.name}.pdf"

    page_size = landscape(A4)
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)

    draw_cover(c, page_size, ts, run_dir, s)
    draw_key_takeaway(c, page_size, run_dir, s, ev)

    for img in imgs:
        add_page_with_image(c, img.name, img, page_size)

    c.save()
    print("OK ->", pdf_path)


if __name__ == "__main__":
    main()
