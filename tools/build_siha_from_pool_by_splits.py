from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: Path, img) -> bool:
    try:
        ext = path.suffix.lower()
        if ext == ".jpg":
            ext = ".jpeg"
        if ext not in {".jpeg", ".png", ".bmp", ".webp"}:
            ext = ".png"

        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False

        buf.tofile(str(path))
        return True
    except Exception:
        return False


def read_lines(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        return []
    return [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_file_index(root: Path, allowed_exts: set[str]) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not root.exists():
        return idx

    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed_exts:
            idx[p.name.lower()] = p
    return idx


def resolve_image_path(
    line: str,
    project_root: Path,
    pool_image_index: Dict[str, Path],
) -> Optional[Path]:
    p = Path(line)

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(project_root / line)
        candidates.append(project_root / p)
        candidates.append(project_root / "data_mine_pool" / "images" / p.name)

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    return pool_image_index.get(p.name.lower())


def find_label_for_image(
    image_path: Path,
    pool_labels_dir: Path,
    pool_label_index: Dict[str, Path],
) -> Optional[Path]:
    target_name = f"{image_path.stem}.txt".lower()

    # 1) train/images -> train/labels
    #    val/images   -> val/labels
    #    test/images  -> test/labels
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            sibling_parts = parts.copy()
            sibling_parts[i] = "labels"
            sibling_path = Path(*sibling_parts).with_suffix(".txt")
            if sibling_path.exists():
                return sibling_path.resolve()

    # 2) txt in the same folder
    same_dir = image_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir.resolve()

    # 3) same stem directly in the pool label folder
    direct = pool_labels_dir / f"{image_path.stem}.txt"
    if direct.exists():
        return direct.resolve()

    # 4) look up via the index
    return pool_label_index.get(target_name)


def load_yolo_boxes(
    label_path: Optional[Path],
    img_w: int,
    img_h: int,
) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []

    if label_path is None or not label_path.exists():
        return boxes

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        boxes.append((cls_id, x1, y1, x2, y2))

    return boxes


def sliding_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]

    positions = [0]
    pos = 0

    while True:
        nxt = pos + stride
        if nxt + tile >= length:
            last = length - tile
            if last not in positions:
                positions.append(last)
            break
        positions.append(nxt)
        pos = nxt

    return positions


def clip_box_to_tile(
    box: Tuple[int, float, float, float, float],
    left: int,
    top: int,
    tile_w: int,
    tile_h: int,
    min_visible_ratio: float,
    min_box_px: int,
) -> Optional[Tuple[int, float, float, float, float]]:
    cls_id, x1, y1, x2, y2 = box

    inter_x1 = max(x1, left)
    inter_y1 = max(y1, top)
    inter_x2 = min(x2, left + tile_w)
    inter_y2 = min(y2, top + tile_h)

    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1

    if inter_w < min_box_px or inter_h < min_box_px:
        return None

    orig_area = max((x2 - x1) * (y2 - y1), 1.0)
    inter_area = inter_w * inter_h
    visible_ratio = inter_area / orig_area

    if visible_ratio < min_visible_ratio:
        return None

    new_x1 = inter_x1 - left
    new_y1 = inter_y1 - top
    new_x2 = inter_x2 - left
    new_y2 = inter_y2 - top

    cx = ((new_x1 + new_x2) / 2.0) / tile_w
    cy = ((new_y1 + new_y2) / 2.0) / tile_h
    bw = (new_x2 - new_x1) / tile_w
    bh = (new_y2 - new_y1) / tile_h

    return (cls_id, cx, cy, bw, bh)


def save_tile_and_label(
    tile_img,
    tile_labels: List[Tuple[int, float, float, float, float]],
    out_img_path: Path,
    out_lbl_path: Path,
) -> None:
    ensure_dir(out_img_path.parent)
    ensure_dir(out_lbl_path.parent)

    ok = imwrite_unicode(out_img_path, tile_img)
    if not ok:
        raise RuntimeError(f"Could not write image: {out_img_path}")

    with out_lbl_path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, bw, bh in tile_labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def process_one_image(
    image_path: Path,
    label_path: Optional[Path],
    out_images_dir: Path,
    out_labels_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    save_empty_tiles: bool,
    min_visible_ratio: float,
    min_box_px: int,
) -> List[Path]:
    img = imread_unicode(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    boxes = load_yolo_boxes(label_path, img_w, img_h)

    stride_x = max(1, int(round(tile_w * (1.0 - overlap))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap))))

    xs = sliding_positions(img_w, tile_w, stride_x)
    ys = sliding_positions(img_h, tile_h, stride_y)

    written_images: List[Path] = []
    out_ext = image_path.suffix.lower() if image_path.suffix.lower() in IMG_EXTS else ".png"

    for top in ys:
        for left in xs:
            tile = img[top:top + tile_h, left:left + tile_w]

            tile_labels: List[Tuple[int, float, float, float, float]] = []
            for box in boxes:
                clipped = clip_box_to_tile(
                    box=box,
                    left=left,
                    top=top,
                    tile_w=tile.shape[1],
                    tile_h=tile.shape[0],
                    min_visible_ratio=min_visible_ratio,
                    min_box_px=min_box_px,
                )
                if clipped is not None:
                    tile_labels.append(clipped)

            if not tile_labels and not save_empty_tiles:
                continue

            tile_name = f"{image_path.stem}__x{left:04d}_y{top:04d}{out_ext}"
            lbl_name = f"{image_path.stem}__x{left:04d}_y{top:04d}.txt"

            out_img_path = out_images_dir / tile_name
            out_lbl_path = out_labels_dir / lbl_name

            save_tile_and_label(tile, tile_labels, out_img_path, out_lbl_path)
            written_images.append(out_img_path.resolve())

    return written_images


def write_split_txt(split_txt_path: Path, image_paths: List[Path]) -> None:
    ensure_dir(split_txt_path.parent)
    with split_txt_path.open("w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(str(p) + "\n")


def write_data_yaml(dst_yaml: Path, scenario_out_root: Path) -> None:
    yaml_text = (
        f"path: {scenario_out_root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        f"  0: cigarette\n"
    )
    dst_yaml.write_text(yaml_text, encoding="utf-8")


def process_scenario(
    project_root: Path,
    scenario: str,
    tile_w: int,
    tile_h: int,
    overlap: float,
    save_empty_tiles: bool,
    min_visible_ratio: float,
    min_box_px: int,
) -> None:
    src_split_dir = project_root / "SIHA_senario_splits" / scenario
    if not src_split_dir.exists():
        raise FileNotFoundError(f"Scenario folder not found: {src_split_dir}")

    pool_images_dir = project_root / "data_mine_pool" / "images"
    pool_labels_dir = project_root / "data_mine_pool" / "labels"

    pool_image_index = build_file_index(pool_images_dir, IMG_EXTS)
    pool_label_index = build_file_index(pool_labels_dir, {".txt"})

    out_root = project_root / "SIHA_data_mine" / scenario
    out_split_dir = project_root / "SIHA_senario_splits" / scenario

    split_names = ["train", "val", "test"]

    for split_name in split_names:
        src_list_path = src_split_dir / f"{split_name}.txt"
        lines = read_lines(src_list_path)

        out_images_dir = out_root / "images" / split_name
        out_labels_dir = out_root / "labels" / split_name
        ensure_dir(out_images_dir)
        ensure_dir(out_labels_dir)

        written_images: List[Path] = []

        for i, line in enumerate(lines, start=1):
            img_path = resolve_image_path(line, project_root, pool_image_index)
            if img_path is None:
                print(f"[WARN] Source image not found: {line}")
                continue

            lbl_path = find_label_for_image(img_path, pool_labels_dir, pool_label_index)
            if lbl_path is None:
                print(f"[WARN] Label not found: {img_path}")
                continue

            try:
                generated = process_one_image(
                    image_path=img_path,
                    label_path=lbl_path,
                    out_images_dir=out_images_dir,
                    out_labels_dir=out_labels_dir,
                    tile_w=tile_w,
                    tile_h=tile_h,
                    overlap=overlap,
                    save_empty_tiles=save_empty_tiles,
                    min_visible_ratio=min_visible_ratio,
                    min_box_px=min_box_px,
                )
                written_images.extend(generated)
                print(f"[{scenario}/{split_name}] {i}/{len(lines)} -> {img_path.name} -> {len(generated)} tile")
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

        write_split_txt(out_split_dir / f"{split_name}.txt", written_images)

    write_data_yaml(out_split_dir / "data.yaml", out_root)
    print(f"[OK] Scenario done: {scenario}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True, type=str)
    parser.add_argument(
        "--scenario",
        required=True,
        type=str,
        choices=["all_time", "day_time", "mid_time", "night_time", "all"],
    )
    parser.add_argument("--tile_w", type=int, default=512)
    parser.add_argument("--tile_h", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--save_empty_tiles", type=int, default=1)
    parser.add_argument("--min_visible_ratio", type=float, default=0.30)
    parser.add_argument("--min_box_px", type=int, default=2)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    scenarios = ["all_time", "day_time", "mid_time", "night_time"] if args.scenario == "all" else [args.scenario]

    for scenario in scenarios:
        process_scenario(
            project_root=project_root,
            scenario=scenario,
            tile_w=args.tile_w,
            tile_h=args.tile_h,
            overlap=args.overlap,
            save_empty_tiles=bool(args.save_empty_tiles),
            min_visible_ratio=args.min_visible_ratio,
            min_box_px=args.min_box_px,
        )


if __name__ == "__main__":
    main()