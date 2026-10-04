from pathlib import Path
import math
import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / r"runs\final\01_yolo11m_siha_optuna_final_e80\01_train_run\weights\best.pt"
INPUT_DIR  = ROOT / r"preview_inputs"
OUTPUT_DIR = ROOT / r"preview_outputs"

CONF_THR = 0.10          # start low; the script sorts by confidence afterwards
IMGSZ = 640
MAX_IMAGES = 16          # at most 16 images in the montage
COLS = 4
TILE_W = 420
TILE_H = 260
ONLY_WITH_DET = True     # only images with a smoking detection go into the montage

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(str(MODEL_PATH))

# -----------------------------
# COLLECT IMAGES
# -----------------------------
exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = sorted([p for p in INPUT_DIR.rglob("*") if p.suffix.lower() in exts])

if not image_paths:
    raise SystemExit(f"No images found: {INPUT_DIR}")

ranked = []

# -----------------------------
# PREDICTION + ANNOTATION
# -----------------------------
for img_path in image_paths:
    results = model.predict(
        source=str(img_path),
        conf=CONF_THR,
        imgsz=IMGSZ,
        verbose=False,
        save=False
    )

    r = results[0]
    boxes = r.boxes

    max_conf = 0.0
    det_count = 0

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy().tolist()
        det_count = len(confs)
        max_conf = max(confs) if confs else 0.0

    if ONLY_WITH_DET and det_count == 0:
        continue

    plotted = r.plot()

    # top info banner
    cv2.rectangle(plotted, (0, 0), (plotted.shape[1], 32), (40, 40, 40), -1)
    txt = f"{img_path.name} | max_conf={max_conf:.3f} | det={det_count}"
    cv2.putText(
        plotted,
        txt,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    out_path = OUTPUT_DIR / img_path.name
    cv2.imwrite(str(out_path), plotted)

    ranked.append({
        "path": out_path,
        "name": img_path.name,
        "max_conf": max_conf,
        "det_count": det_count
    })

if not ranked:
    raise SystemExit("No image with a smoking detection. Try lowering CONF_THR.")

# -----------------------------
# SORT BY CONF
# -----------------------------
ranked.sort(key=lambda x: x["max_conf"], reverse=True)

# save txt summary
summary_txt = OUTPUT_DIR / "ranked_summary.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    for i, item in enumerate(ranked, 1):
        f.write(f"{i:02d}. {item['name']} | max_conf={item['max_conf']:.3f} | det={item['det_count']}\n")

# -----------------------------
# BUILD MONTAGE
# -----------------------------
top_items = ranked[:MAX_IMAGES]
rows = math.ceil(len(top_items) / COLS)

canvas = np.full((rows * TILE_H, COLS * TILE_W, 3), 235, dtype=np.uint8)

for i, item in enumerate(top_items):
    img = cv2.imread(str(item["path"]))
    if img is None:
        continue

    img = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)

    r = i // COLS
    c = i % COLS
    y1 = r * TILE_H
    x1 = c * TILE_W

    canvas[y1:y1 + TILE_H, x1:x1 + TILE_W] = img
    cv2.rectangle(canvas, (x1, y1), (x1 + TILE_W, y1 + TILE_H), (180, 180, 180), 2)

montage_path = OUTPUT_DIR / "montage_top_conf.jpg"
cv2.imwrite(str(montage_path), canvas)

print("Done.")
print("Input folder :", INPUT_DIR)
print("Output folder:", OUTPUT_DIR)
print("Montage       :", montage_path)
print("Summary list  :", summary_txt)