import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

GT_CSV   = ROOT / r"outputs\02_gt_csv\FINAL_SIHA_DAY_TIME\11\gt_11_final.csv"
PRED_IN  = ROOT / r"outputs\03_predictions_jsonl\FINAL_SIHA_DAY_TIME\11\pred_11.jsonl"
OUT_JSON = ROOT / r"outputs\04_metrics_active\FINAL_SIHA_DAY_TIME\11\metrics_BBOX_GT_11\summary_sample_only.json"

CONF_THR = 0.10
IOU_THR  = 0.30

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# -------------------------
# read GT
# -------------------------
labeled_frames = set()
gt_by_frame = defaultdict(list)

with GT_CSV.open("r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for r in reader:
        fi = int(r["frame_idx"])
        labeled_frames.add(fi)

        x1 = str(r.get("x1", "")).strip()
        y1 = str(r.get("y1", "")).strip()
        x2 = str(r.get("x2", "")).strip()
        y2 = str(r.get("y2", "")).strip()

        # add bbox if present; otherwise the frame counts as negative within the sample set
        if x1 and y1 and x2 and y2:
            gt_by_frame[fi].append((float(x1), float(y1), float(x2), float(y2)))

# -------------------------
# read predictions (labeled frames only)
# -------------------------
pred_by_frame = defaultdict(list)

with PRED_IN.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        fi = int(rec["frame_idx"])
        if fi not in labeled_frames:
            continue

        preds = rec.get("pred", []) or []
        for p in preds:
            try:
                conf = float(p.get("conf", 1.0))
                if conf < CONF_THR:
                    continue
                pred_by_frame[fi].append((
                    float(p["x1"]), float(p["y1"]),
                    float(p["x2"]), float(p["y2"]),
                    conf
                ))
            except Exception:
                continue

# -------------------------
# Eval
# -------------------------
tp = 0
fp = 0
fn = 0
matched_ious = []

positive_frame_tp = 0
positive_frame_fn = 0

for fi in sorted(labeled_frames):
    gts = list(gt_by_frame.get(fi, []))
    preds = list(pred_by_frame.get(fi, []))

    # greedy match
    used_gt = set()
    used_pred = set()

    while True:
        best_iou = -1.0
        best_g = -1
        best_p = -1

        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            for pi, p in enumerate(preds):
                if pi in used_pred:
                    continue
                iou = iou_xyxy(g, p[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_g = gi
                    best_p = pi

        if best_g == -1 or best_p == -1 or best_iou < IOU_THR:
            break

        used_gt.add(best_g)
        used_pred.add(best_p)
        tp += 1
        matched_ious.append(best_iou)

    fp += (len(preds) - len(used_pred))
    fn += (len(gts) - len(used_gt))

    # frame-level positive recall
    if len(gts) > 0:
        if len(used_gt) > 0:
            positive_frame_tp += 1
        else:
            positive_frame_fn += 1

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

pos_frame_recall = positive_frame_tp / (positive_frame_tp + positive_frame_fn) if (positive_frame_tp + positive_frame_fn) > 0 else 0.0

summary = {
    "mode": "sample_only_bbox_eval",
    "gt_csv": str(GT_CSV),
    "pred_jsonl": str(PRED_IN),
    "conf_thr": CONF_THR,
    "iou_thr": IOU_THR,

    "labeled_frames": len(labeled_frames),
    "labeled_positive_frames": sum(1 for fi in labeled_frames if len(gt_by_frame.get(fi, [])) > 0),
    "labeled_negative_frames": sum(1 for fi in labeled_frames if len(gt_by_frame.get(fi, [])) == 0),

    "TP_boxes": tp,
    "FP_boxes": fp,
    "FN_boxes": fn,
    "box_precision": precision,
    "box_recall": recall,
    "box_f1": f1,

    "matched_iou_mean": (sum(matched_ious) / len(matched_ious)) if matched_ious else None,
    "matched_iou_median": statistics.median(matched_ious) if matched_ious else None,

    "positive_frame_tp": positive_frame_tp,
    "positive_frame_fn": positive_frame_fn,
    "positive_frame_recall": pos_frame_recall,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with OUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("OK ->", OUT_JSON)
print(json.dumps(summary, ensure_ascii=False, indent=2))
