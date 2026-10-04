# tools/video_smoking_review.py
# -*- coding: utf-8 -*-
"""
VIDEO -> Render Overlay (VLM dashboard + Zoom Grid)
==================================================
Runs a video file through the API server (core/apiserver.py) and writes a
"render"-style output: bboxes land on the same frame they belong to (no lag).

Features:
- --ui_vlm : adds a top header + bottom footer band (name + SMOKING/NO + LAT)
- --zoom_panel : draws a ZOOM GRID (2x2 default) in a right-hand panel
  (tiled so it stays readable when several people smoke at once)
- --dedup_iou : merges multiple bboxes of the same cigarette (fewer duplicates)

Run (2 windows):
  # 1) API server:
  py -3.10 .\\core\\apiserver.py

  # 2) Render:
  py -3.10 .\\tools\\video_smoking_review.py --api http://127.0.0.1:5000 --video "youtube_videos/video1.mp4" ^
    --sync_api --analyze_every 1 --jpeg_quality 85 --no_show --save_mp4 "outputs\\video1_overlay_vlm.mp4" ^
    --ui_vlm --zoom_panel --zoom_tiles 4 --zoom_cols 2 --zoom_tile 280 --zoom_pad 2.0 --zoom_hold 1.2 --dedup_iou 0.70

Notes:
- Rendering is not realtime: speed = per-frame API latency.
- --no_show does not open a window, it only writes the file.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import requests  # type: ignore
except Exception:
    requests = None


# ----------------- helpers -----------------
def _cap_fps(cap) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6 or fps > 240:
        return 25.0
    return fps


def _put(img, text, org, scale=0.7, fg=(255, 255, 255), bg=(0, 0, 0), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def _clip_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W, int(x2)))
    y2 = max(0, min(H, int(y2)))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def _encode_jpg(frame_bgr: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter + 1e-9)


def _dedup_iou(boxes: List[Tuple[int, int, int, int]], thr: float) -> List[Tuple[int, int, int, int]]:
    kept: List[Tuple[int, int, int, int]] = []
    for b in boxes:
        if all(_iou_xyxy(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def _as_box_any(obj) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        try:
            return int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        except Exception:
            return None
    if isinstance(obj, dict):
        for k in ("bbox", "xyxy", "box"):
            v = obj.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                try:
                    return int(v[0]), int(v[1]), int(v[2]), int(v[3])
                except Exception:
                    return None
    return None


def api_process_frame(api_url: str, frame_bgr: np.ndarray, camera_id: str, jpeg_quality: int) -> Dict:
    if requests is None:
        raise RuntimeError("requests is not installed. Install: pip install requests")
    jpg = _encode_jpg(frame_bgr, jpeg_quality)
    files = {"frame": ("frame.jpg", jpg, "image/jpeg")}
    data = {"camera_id": str(camera_id), "source": "video", "return_frame": "0"}
    r = requests.post(f"{api_url.rstrip('/')}/process_frame", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def smoke_matches_face(
    smoke_xyxy: Tuple[int, int, int, int],
    face_xyxy: Tuple[int, int, int, int],
    *,
    mouth_dist: float,
    max_area_ratio: float,
    max_side_ratio: float,
    x_margin: float,
    y_up: float,
    y_down: float,
    lower_y: float,
) -> bool:
    fx1, fy1, fx2, fy2 = face_xyxy
    sx1, sy1, sx2, sy2 = smoke_xyxy
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    sw = max(1, sx2 - sx1)
    sh = max(1, sy2 - sy1)

    # drop very large boxes
    if sw > fw * max_side_ratio or sh > fh * max_side_ratio:
        return False
    if (sw * sh) > (fw * fh) * max_area_ratio:
        return False

    scx = (sx1 + sx2) * 0.5
    scy = (sy1 + sy2) * 0.5

    ex1 = fx1 - fw * x_margin
    ex2 = fx2 + fw * x_margin
    ey1 = fy1 - fh * y_up
    ey2 = fy2 + fh * y_down

    if not (ex1 <= scx <= ex2 and ey1 <= scy <= ey2):
        return False

    mcx = (fx1 + fx2) * 0.5
    mcy = fy1 + fh * 0.65

    if scy < fy1 + fh * lower_y:
        return False

    d = ((scx - mcx) ** 2 + (scy - mcy) ** 2) ** 0.5 / float(fh)
    return d <= mouth_dist


@dataclass
class FaceDet:
    name: str
    sim: float
    bbox: Tuple[int, int, int, int]


@dataclass
class Track:
    tid: int
    name: str
    sim: float
    bbox: Tuple[int, int, int, int]
    last_seen: float
    misses: int = 0
    smoking: bool = False


class SimpleTracker:
    def __init__(self, iou_match: float = 0.25, max_misses: int = 15):
        self.iou_match = float(iou_match)
        self.max_misses = int(max_misses)
        self._next = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[FaceDet], now: float) -> List[Track]:
        for t in self.tracks.values():
            t.misses += 1

        unmatched = dets[:]
        for tid, t in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for i, d in enumerate(unmatched):
                iou = _iou_xyxy(t.bbox, d.bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0 and best_iou >= self.iou_match:
                d = unmatched.pop(best_idx)
                t.bbox = d.bbox
                t.name = d.name
                t.sim = d.sim
                t.last_seen = now
                t.misses = 0

        for d in unmatched:
            tid = self._next
            self._next += 1
            self.tracks[tid] = Track(tid=tid, name=d.name, sim=d.sim, bbox=d.bbox, last_seen=now)

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].misses > self.max_misses:
                del self.tracks[tid]

        return list(self.tracks.values())


def draw_header_footer(canvas: np.ndarray, header_h: int, footer_h: int, header_text: str, footer_lines: List[str]):
    H, W = canvas.shape[:2]
    if header_h > 0:
        cv2.rectangle(canvas, (0, 0), (W, header_h), (0, 0, 0), -1)
        _put(canvas, header_text, (10, int(header_h * 0.70)), scale=0.75)
    if footer_h > 0:
        y0 = H - footer_h
        cv2.rectangle(canvas, (0, y0), (W, H), (0, 0, 0), -1)
        y = y0 + 24
        for line in footer_lines[:5]:
            _put(canvas, line, (10, y), scale=0.7)
            y += 22


def crop_with_pad(frame: np.ndarray, box: Tuple[int, int, int, int], pad: float) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    px = int(bw * pad)
    py = int(bh * pad)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
    return frame[cy1:cy2, cx1:cx2].copy()


def build_zoom_grid(
    tiles: List[Tuple[np.ndarray, str]],
    tile_size: int,
    grid_cols: int,
    total_tiles: int,
) -> np.ndarray:
    grid_rows = int(np.ceil(total_tiles / float(grid_cols)))
    out_h = grid_rows * tile_size
    out_w = grid_cols * tile_size
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for idx in range(total_tiles):
        r = idx // grid_cols
        c = idx % grid_cols
        y1 = r * tile_size
        x1 = c * tile_size
        if idx < len(tiles):
            img, label = tiles[idx]
            if img is None:
                crop = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            else:
                ih, iw = img.shape[:2]
                interp = cv2.INTER_CUBIC if (iw < tile_size or ih < tile_size) else cv2.INTER_AREA
                crop = cv2.resize(img, (tile_size, tile_size), interpolation=interp)
            canvas[y1 : y1 + tile_size, x1 : x1 + tile_size] = crop
            cv2.rectangle(canvas, (x1, y1), (x1 + tile_size, y1 + 26), (0, 0, 0), -1)
            _put(canvas, label, (x1 + 6, y1 + 20), scale=0.6)
        else:
            cv2.rectangle(canvas, (x1, y1), (x1 + tile_size, y1 + tile_size), (25, 25, 25), -1)
            _put(canvas, "EMPTY", (x1 + 6, y1 + 20), scale=0.6)
    return canvas


def _make_writer(out_path: Path, fps: float, size_wh: Tuple[int, int]) -> cv2.VideoWriter:
    ext = out_path.suffix.lower()
    if ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size_wh
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")
    return writer


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:5000")
    ap.add_argument("--camera_id", default="1")
    ap.add_argument("--video", default="youtube_videos/video1.mp4")
    ap.add_argument("--fps_limit", type=float, default=0.0, help="0 => no render throttle (recommended with no_show)")
    ap.add_argument("--analyze_every", type=int, default=5)
    ap.add_argument("--sync_api", action="store_true")
    ap.add_argument("--jpeg_quality", type=int, default=85)
    ap.add_argument("--draw_raw", action="store_true")

    ap.add_argument("--no_fp_filter", action="store_true")
    ap.add_argument("--dedup_iou", type=float, default=0.70)

    ap.add_argument("--mouth_dist", type=float, default=1.35)
    ap.add_argument("--max_area_ratio", type=float, default=0.25)
    ap.add_argument("--max_side_ratio", type=float, default=1.20)
    ap.add_argument("--x_margin", type=float, default=1.00)
    ap.add_argument("--y_up", type=float, default=0.30)
    ap.add_argument("--y_down", type=float, default=2.00)
    ap.add_argument("--lower_y", type=float, default=0.35)

    ap.add_argument("--save_mp4", default="", help="outputs/annotated.mp4 or .avi")
    ap.add_argument("--no_show", action="store_true")

    # UI / VLM
    ap.add_argument("--ui_vlm", action="store_true", help="header/footer band")
    ap.add_argument("--header_h", type=int, default=44)
    ap.add_argument("--footer_h", type=int, default=72)

    ap.add_argument("--zoom_panel", action="store_true", help="zoom grid in the right panel")
    ap.add_argument("--zoom_tiles", type=int, default=4)
    ap.add_argument("--zoom_cols", type=int, default=2)
    ap.add_argument("--zoom_tile", type=int, default=280)
    ap.add_argument("--zoom_pad", type=float, default=2.0)
    ap.add_argument("--zoom_hold", type=float, default=1.2)
    ap.add_argument("--panel_w", type=int, default=0, help="0 => zoom_cols*zoom_tile")

    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    vpath = (root / args.video).resolve()
    if not vpath.exists():
        raise SystemExit(f"ERROR: video not found: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise SystemExit("ERROR: could not open VideoCapture.")

    fps_native = _cap_fps(cap)

    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("ERROR: could not read the first frame.")
    H, W = frame.shape[:2]

    # output canvas sizing
    panel_w = int(args.panel_w) if args.panel_w > 0 else int(args.zoom_cols * args.zoom_tile)
    use_panel = bool(args.zoom_panel)
    out_W = W + (panel_w if use_panel else 0)
    out_H = H + (int(args.header_h) if args.ui_vlm else 0) + (int(args.footer_h) if args.ui_vlm else 0)
    header_h = int(args.header_h) if args.ui_vlm else 0
    footer_h = int(args.footer_h) if args.ui_vlm else 0

    writer = None
    if args.save_mp4:
        out_path = (root / args.save_mp4).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = _make_writer(out_path, fps_native, (out_W, out_H))
        print("SAVE ->", out_path)

    tracker = SimpleTracker(iou_match=0.25, max_misses=15)
    zoom_cache: Dict[int, Tuple[np.ndarray, float]] = {}

    # warmup on first frame
    last_api_json = {"results": [], "smoking_boxes": []}
    last_latency = 0.0

    frame_idx = 0
    paused = False
    last_api_t = time.time()

    screenshot_dir = (root / "outputs" / "screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1

            do_analyze = (frame_idx % max(1, int(args.analyze_every)) == 0)

            if do_analyze:
                t0 = time.time()
                # sync_api: render-like mode already waits; no difference, flag kept for compatibility
                last_api_json = api_process_frame(args.api, frame, args.camera_id, int(args.jpeg_quality))
                last_latency = time.time() - t0
                last_api_t = time.time()

            # parse faces
            faces_raw = last_api_json.get("results", []) or []
            face_dets: List[FaceDet] = []
            for r in faces_raw[:12]:
                bb = r.get("bbox", None)
                if not bb or len(bb) < 4:
                    continue
                fx1, fy1, fx2, fy2 = map(int, bb[:4])
                fx1, fy1, fx2, fy2 = _clip_xyxy(fx1, fy1, fx2, fy2, W, H)
                name = str(r.get("name", "Unknown"))
                try:
                    sim = float(r.get("sim", -1.0))
                except Exception:
                    sim = -1.0
                face_dets.append(FaceDet(name=name, sim=sim, bbox=(fx1, fy1, fx2, fy2)))

            now = time.time()
            tracks = tracker.update(face_dets, now)

            # parse smokes
            raw_smokes: List[Tuple[int, int, int, int]] = []
            for sb in (last_api_json.get("smoking_boxes", []) or [])[:40]:
                b = _as_box_any(sb)
                if b is None:
                    continue
                x1, y1, x2, y2 = _clip_xyxy(*b, W, H)
                raw_smokes.append((x1, y1, x2, y2))
            if args.dedup_iou and args.dedup_iou > 0:
                raw_smokes = _dedup_iou(raw_smokes, float(args.dedup_iou))

            filtered_smokes: List[Tuple[int, int, int, int]] = []
            if args.no_fp_filter or len(tracks) == 0:
                filtered_smokes = list(raw_smokes)
            else:
                for sb in raw_smokes:
                    if any(
                        smoke_matches_face(
                            sb,
                            tr.bbox,
                            mouth_dist=float(args.mouth_dist),
                            max_area_ratio=float(args.max_area_ratio),
                            max_side_ratio=float(args.max_side_ratio),
                            x_margin=float(args.x_margin),
                            y_up=float(args.y_up),
                            y_down=float(args.y_down),
                            lower_y=float(args.lower_y),
                        )
                        for tr in tracks
                    ):
                        filtered_smokes.append(sb)

            if args.dedup_iou and args.dedup_iou > 0:
                filtered_smokes = _dedup_iou(filtered_smokes, float(args.dedup_iou))

            # update per-track smoking + zoom cache
            for tr in tracks:
                tr.smoking = any(
                    smoke_matches_face(
                        sb,
                        tr.bbox,
                        mouth_dist=float(args.mouth_dist),
                        max_area_ratio=float(args.max_area_ratio),
                        max_side_ratio=float(args.max_side_ratio),
                        x_margin=float(args.x_margin),
                        y_up=float(args.y_up),
                        y_down=float(args.y_down),
                        lower_y=float(args.lower_y),
                    )
                    for sb in filtered_smokes
                )
                if tr.smoking and filtered_smokes:
                    # choose best smoke near mouth
                    fx1, fy1, fx2, fy2 = tr.bbox
                    fh = max(1, fy2 - fy1)
                    mcx = (fx1 + fx2) * 0.5
                    mcy = fy1 + fh * 0.65
                    best = None
                    best_d = 1e18
                    for sb in filtered_smokes:
                        sx1, sy1, sx2, sy2 = sb
                        scx = (sx1 + sx2) * 0.5
                        scy = (sy1 + sy2) * 0.5
                        d = (scx - mcx) ** 2 + (scy - mcy) ** 2
                        if d < best_d:
                            best_d, best = d, sb
                    if best is not None:
                        # FACE + SMOKE UNION (both sharp in the same tile)
                        fx1, fy1, fx2, fy2 = map(int, tr.bbox)     # face bbox
                        sx1, sy1, sx2, sy2 = map(int, best)        # smoke bbox

                        ux1, uy1 = min(fx1, sx1), min(fy1, sy1)
                        ux2, uy2 = max(fx2, sx2), max(fy2, sy2)

                        # cap so the face does not shrink when zoom_pad is large
                        pad_union = min(float(args.zoom_pad), 0.8)

                        crop = crop_with_pad(frame, (ux1, uy1, ux2, uy2), pad=pad_union)
                        zoom_cache[tr.tid] = (crop, now)
            # draw on main frame
            vis = frame.copy()

            if args.draw_raw:
                for (x1, y1, x2, y2) in raw_smokes:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    _put(vis, "RAW", (x1, max(0, y1 - 6)), scale=0.6, fg=(0, 255, 255))

            for (x1, y1, x2, y2) in filtered_smokes:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                _put(vis, "SMOKE", (x1, max(0, y1 - 6)), scale=0.7, fg=(0, 0, 255))

            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                _put(vis, f"{tr.name} ({tr.sim:.2f})", (x1, max(0, y1 - 8)), scale=0.7, fg=(0, 255, 0))
                if tr.smoking:
                    _put(vis, "SMOKING", (x1, min(H - 10, y2 + 22)), scale=0.8, fg=(0, 0, 255))

            aplage = max(0.0, time.time() - last_api_t)
            header_text = f"VIDEO | LAT={last_latency:.2f}s | APLAGE={aplage:.2f}s | every={args.analyze_every} | faces={len(tracks)} | smokes={len(filtered_smokes)}"

            footer_lines = []
            for t in sorted(tracks, key=lambda z: (-int(z.smoking), -z.sim))[:6]:
                footer_lines.append(f"{t.name}: {'SMOKING' if t.smoking else 'NO'}  sim={t.sim:.2f}")

            # compose output canvas if ui_vlm or panel enabled
            if args.ui_vlm or use_panel:
                out = np.zeros((out_H, out_W, 3), dtype=np.uint8)
                # paste main
                out[header_h : header_h + H, 0:W] = vis

                # zoom panel
                if use_panel:
                    # build tiles for smoking tracks (or last cached)
                    tiles: List[Tuple[np.ndarray, str]] = []
                    smoking_tracks = [t for t in tracks if t.smoking]
                    smoking_tracks.sort(key=lambda t: (t.last_seen), reverse=True)

                    for tr in smoking_tracks[: int(args.zoom_tiles)]:
                        img = None
                        if tr.tid in zoom_cache:
                            crop, ts = zoom_cache[tr.tid]
                            if (now - ts) <= float(args.zoom_hold):
                                img = crop
                        label = f"{tr.name} | SMOKING"
                        tiles.append((img if img is not None else np.zeros((int(args.zoom_tile), int(args.zoom_tile), 3), np.uint8), label))

                    grid = build_zoom_grid(
                        tiles,
                        tile_size=int(args.zoom_tile),
                        grid_cols=int(args.zoom_cols),
                        total_tiles=int(args.zoom_tiles),
                    )
                    # center grid vertically in panel
                    ph = H
                    pw = panel_w
                    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
                    gh, gw = grid.shape[:2]
                    y_off = max(0, (ph - gh) // 2)
                    x_off = max(0, (pw - gw) // 2)
                    panel[y_off : y_off + gh, x_off : x_off + gw] = grid
                    out[header_h : header_h + H, W : W + pw] = panel

                if args.ui_vlm:
                    draw_header_footer(out, header_h, footer_h, header_text, footer_lines)

            else:
                out = vis

            if writer is not None:
                writer.write(out)

            if not args.no_show:
                cv2.imshow("VIDEO REVIEW (VLM)", out)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:  # SPACE
                    paused = not paused
                elif key == ord("s"):
                    outp = screenshot_dir / f"shot_{int(time.time())}.jpg"
                    cv2.imwrite(str(outp), out)
                    print("screenshot ->", outp)

            # throttle only when showing
            if (not args.no_show) and args.fps_limit and args.fps_limit > 0:
                time.sleep(max(0.0, (1.0 / float(args.fps_limit)) - 0.0005))

        else:
            if not args.no_show:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:
                    paused = not paused
                elif key == ord("s"):
                    outp = screenshot_dir / f"shot_{int(time.time())}.jpg"
                    cv2.imwrite(str(outp), out)
                    print("screenshot ->", outp)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
