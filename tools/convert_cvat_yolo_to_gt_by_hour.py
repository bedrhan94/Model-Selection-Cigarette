from pathlib import Path
import argparse
import csv
import cv2

ROOT = Path(__file__).resolve().parents[1]

def yolo_to_xyxy(line, W, H):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = (cx - bw / 2.0) * W
    y1 = (cy - bh / 2.0) * H
    x2 = (cx + bw / 2.0) * W
    y2 = (cy + bh / 2.0) * H
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W, x2))
    y2 = max(0.0, min(H, y2))
    return x1, y1, x2, y2

def convert(hour: str):
    template_csv = ROOT / fr"outputs\02_gt_csv\FINAL_SIHA_DAY_TIME\{hour}\gt_{hour}_template.csv"
    frames_dir   = ROOT / fr"outputs\02_gt_csv\FINAL_SIHA_DAY_TIME\{hour}\gt_frames_{hour}"
    export_dir   = ROOT / fr"outputs\02_gt_csv\FINAL_SIHA_DAY_TIME\{hour}\cvat_export_{hour}"
    out_csv      = ROOT / fr"outputs\02_gt_csv\FINAL_SIHA_DAY_TIME\{hour}\gt_{hour}_final.csv"

    label_map = {}
    for p in export_dir.rglob("*.txt"):
        if p.name.lower() in {"train.txt", "val.txt", "test.txt"}:
            continue
        label_map[p.stem] = p

    rows_out = []

    with open(template_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        template_rows = list(reader)

    for r in template_rows:
        frame_idx = int(r["frame_idx"])
        img_path = Path(r["image"])
        if not img_path.exists():
            img_path = frames_dir / img_path.name

        img = cv2.imread(str(img_path))
        if img is None:
            rows_out.append({
                "frame_idx": frame_idx,
                "image": str(img_path),
                "x1": "", "y1": "", "x2": "", "y2": "",
                "class": "cigarette"
            })
            continue

        H, W = img.shape[:2]
        stem = img_path.stem
        lab = label_map.get(stem)

        if lab is None or lab.stat().st_size == 0:
            rows_out.append({
                "frame_idx": frame_idx,
                "image": str(img_path),
                "x1": "", "y1": "", "x2": "", "y2": "",
                "class": "cigarette"
            })
            continue

        lines = [ln.strip() for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            rows_out.append({
                "frame_idx": frame_idx,
                "image": str(img_path),
                "x1": "", "y1": "", "x2": "", "y2": "",
                "class": "cigarette"
            })
            continue

        for ln in lines:
            conv = yolo_to_xyxy(ln, W, H)
            if conv is None:
                continue
            x1, y1, x2, y2 = conv
            rows_out.append({
                "frame_idx": frame_idx,
                "image": str(img_path),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "class": "cigarette"
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "image", "x1", "y1", "x2", "y2", "class"])
        writer.writeheader()
        writer.writerows(rows_out)

    print("OK ->", out_csv)
    print("rows =", len(rows_out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hour", required=True)
    args = ap.parse_args()
    convert(str(args.hour))
