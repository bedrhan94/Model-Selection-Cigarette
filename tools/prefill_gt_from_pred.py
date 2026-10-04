# tools/prefill_gt_from_pred.py
# -*- coding: utf-8 -*-
import json, csv, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="outputs/03_predictions_jsonl/pred_*.jsonl")
    ap.add_argument("--out", required=True, help="outputs/02_gt_csv/gt_*_filled.csv")
    ap.add_argument("--conf_thr", type=float, default=0.10)
    ap.add_argument("--topk", type=int, default=1, help="1=single bbox (highest conf), 999=all")
    ap.add_argument("--class_name", default="cigarette")
    args = ap.parse_args()

    pred_path = Path(args.pred)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fi = int(rec["frame_idx"])
            pred = rec.get("pred", []) or []

            # filter + sort
            cand = []
            for b in pred:
                try:
                    c = float(b.get("conf", 1.0))
                    if c < args.conf_thr:
                        continue
                    x1 = float(b["x1"]); y1 = float(b["y1"]); x2 = float(b["x2"]); y2 = float(b["y2"])
                    cand.append((c, x1, y1, x2, y2))
                except Exception:
                    continue
            cand.sort(key=lambda x: -x[0])

            if args.topk <= 1:
                # single bbox
                if cand:
                    c, x1, y1, x2, y2 = cand[0]
                    rows.append({"frame_idx": fi, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": args.class_name, "prefill_conf": c})
                else:
                    rows.append({"frame_idx": fi, "x1": "", "y1": "", "x2": "", "y2": "", "class": args.class_name, "prefill_conf": ""})
            else:
                # multiple bboxes
                if not cand:
                    rows.append({"frame_idx": fi, "x1": "", "y1": "", "x2": "", "y2": "", "class": args.class_name, "prefill_conf": ""})
                else:
                    for (c, x1, y1, x2, y2) in cand[:args.topk]:
                        rows.append({"frame_idx": fi, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": args.class_name, "prefill_conf": c})

    # write
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_idx","x1","y1","x2","y2","class","prefill_conf"])
        w.writeheader()
        w.writerows(rows)

    print("OK ->", str(out_path), "rows=", len(rows))

if __name__ == "__main__":
    main()