# tools/dual_video_smoking_test.py
# -*- coding: utf-8 -*-
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2

try:
    import requests
except Exception:
    requests = None


def find_video(folder: Path, base_names):
    exts = ["*.mp4", "*.avi", "*.mkv", "*.mov", "*.m4v", "*.webm"]
    for bn in base_names:
        for ext in exts:
            hits = list(folder.glob(f"{bn}{ext[1:]}"))
            if hits:
                return hits[0]
        hits = list(folder.glob(f"{bn}*.*"))
        if hits:
            return hits[0]
    return None


def resize_to_height(img, target_h=540):
    if img is None:
        return None
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def pixelate_roi(img, x1, y1, x2, y2, blocks=20):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return img

    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh <= 0 or rw <= 0:
        return img

    xs = max(1, rw // blocks)
    ys = max(1, rh // blocks)
    small = cv2.resize(roi, (xs, ys), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pix
    return img


def _put_shadow_text(img, text, org, scale=0.9, color=(255, 255, 255), shadow=(0, 0, 0), thickness=3):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, shadow, thickness + 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _get_screen_size():
    # Usually reliable on Windows. Falls back to 1920x1080 on error.
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def _fit_to_screen(img, margin=120):
    if img is None:
        return None
    sw, sh = _get_screen_size()
    sh = max(200, sh - int(margin))  # margin for taskbar/title bar
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return img
    scale = min(sw / float(iw), sh / float(ih))
    if scale >= 0.999:
        return img
    nw, nh = int(iw * scale), int(ih * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def create_tracker(pref="auto"):
    """
    Tracker selection.
    - pref: auto|mosse|kcf|csrt|none
    """
    pref = (pref or "auto").lower().strip()
    if pref == "none":
        return None

    def _mk(name):
        if hasattr(cv2, "legacy"):
            fn = getattr(cv2.legacy, f"Tracker{name}_create", None)
            if fn:
                return fn()
        fn = getattr(cv2, f"Tracker{name}_create", None)
        if fn:
            return fn()
        return None

    if pref == "mosse":
        tr = _mk("MOSSE")
        if tr is not None:
            return tr
    if pref == "kcf":
        tr = _mk("KCF")
        if tr is not None:
            return tr
    if pref == "csrt":
        tr = _mk("CSRT")
        if tr is not None:
            return tr

    for name in ("MOSSE", "KCF", "CSRT"):
        tr = _mk(name)
        if tr is not None:
            return tr
    return None


def post_frame(api_url, frame_bgr, camera_id="1", source="video", jpeg_quality=85):
    if requests is None:
        raise RuntimeError("requests is not installed. Install: pip install requests")

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return {"ok": False, "error": "imencode_failed", "results": [], "smoking_boxes": [], "yolo_used": {}}

    files = {"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    data = {"camera_id": str(camera_id), "source": str(source), "return_frame": "0"}

    r = requests.post(f"{api_url.rstrip('/')}/process_frame", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def _cap_fps(cap):
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6 or fps > 240:
        return 0.0
    return fps


class FaceTrack:
    __slots__ = ("tracker", "bbox", "name", "sim", "smoking")

    def __init__(self, tracker, bbox, name, sim, smoking=False):
        self.tracker = tracker
        self.bbox = bbox
        self.name = name
        self.sim = float(sim)
        self.smoking = bool(smoking)


class SmokeTrack:
    __slots__ = ("tracker", "bbox")

    def __init__(self, tracker, bbox):
        self.tracker = tracker
        self.bbox = bbox


def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = map(int, b)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _xywh_to_xyxy(x, y, w, h):
    return (int(x), int(y), int(x + w), int(y + h))


def _clip_xyxy(b, w, h):
    x1, y1, x2, y2 = b
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return (x1, y1, x2, y2)


def _parse_box_any(obj):
    """
    smoking_boxes may come as either [x1,y1,x2,y2] or {"bbox":[...]}.
    """
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)) and len(obj) >= 4:
        return [obj[0], obj[1], obj[2], obj[3]]
    if isinstance(obj, dict):
        for k in ("bbox", "xyxy", "box"):
            v = obj.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 4:
                return [v[0], v[1], v[2], v[3]]
    return None



def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def _dedup_iou(boxes, iou_thr=0.7):
    """Simple IoU deduplication (the practical option when there is no conf).
    The higher iou_thr, the more aggressive the deduplication.
    """
    if not boxes:
        return []
    kept = []
    for b in boxes:
        if all(_iou_xyxy(b, k) < iou_thr for k in kept):
            kept.append(b)
    return kept


def _smoke_matches_face(
    smoke_xyxy,
    face_xyxy,
    *,
    mouth_dist=1.35,
    max_area_ratio=0.25,
    max_side_ratio=1.20,
    x_margin=1.00,
    y_up=0.30,
    y_down=2.00,
    lower_y=0.35
):
    fx1, fy1, fx2, fy2 = face_xyxy
    sx1, sy1, sx2, sy2 = smoke_xyxy

    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)
    sw = max(1, sx2 - sx1)
    sh = max(1, sy2 - sy1)

    # Drop large boxes (e.g. railings/bars)
    if sw > fw * max_side_ratio or sh > fh * max_side_ratio:
        return False
    face_area = fw * fh
    smoke_area = sw * sh
    if smoke_area > face_area * max_area_ratio:
        return False

    scx = (sx1 + sx2) * 0.5
    scy = (sy1 + sy2) * 0.5

    # Around the face + wider downwards (for a cigarette in hand)
    ex1 = fx1 - fw * x_margin
    ex2 = fx2 + fw * x_margin
    ey1 = fy1 - fh * y_up
    ey2 = fy2 + fh * y_down

    if not (ex1 <= scx <= ex2 and ey1 <= scy <= ey2):
        return False

    # Mouth/chin estimate
    mcx = (fx1 + fx2) * 0.5
    mcy = fy1 + fh * 0.65

    # Must not be too far above
    if scy < fy1 + fh * lower_y:
        return False

    dx = scx - mcx
    dy = scy - mcy
    d = (dx * dx + dy * dy) ** 0.5 / float(fh)
    return d <= mouth_dist


def rebuild_tracks_from_api(
    frame,
    api_json,
    *,
    tracker_pref="auto",
    no_fp_filter=False,
    mouth_dist=1.35,
    max_area_ratio=0.25,
    max_side_ratio=1.20,
    x_margin=1.00,
    y_up=0.30,
    y_down=2.00,
    lower_y=0.35,
    max_faces=8,
    max_smokes=12,
    dedup_iou=0.70
):
    H, W = frame.shape[:2]
    faces = api_json.get("results", []) or []
    smokes = api_json.get("smoking_boxes", []) or []

    parsed_faces = []
    for r in faces[:max_faces]:
        try:
            name = r.get("name", "Unknown")
            sim = float(r.get("sim", -1.0))
            bb = r.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = map(int, bb[:4])
            fb = _clip_xyxy((x1, y1, x2, y2), W, H)
            parsed_faces.append((name, sim, fb))
        except Exception:
            continue

    face_boxes = [fb for (_, _, fb) in parsed_faces]

    raw_smokes = []
    for sb in smokes[:max_smokes]:
        try:
            b = _parse_box_any(sb)
            if b is None:
                continue
            x1, y1, x2, y2 = map(int, b[:4])
            raw_smokes.append(_clip_xyxy((x1, y1, x2, y2), W, H))
        except Exception:
            continue

    # Dedup (merge overlapping bboxes)
    try:
        di = float(dedup_iou)
    except Exception:
        di = 0.0
    if di and di > 0:
        raw_smokes = _dedup_iou(raw_smokes, iou_thr=di)

    # FP filter (face/mouth association)
    filtered_smokes = []
    if no_fp_filter or not face_boxes:
        filtered_smokes = list(raw_smokes)
    elif not face_boxes:
        filtered_smokes = []   # no face -> do NOT count the cigarette
    else:
        for b in raw_smokes:
            ok_assoc = False
            for fb in face_boxes:
                if _smoke_matches_face(
                    b, fb,
                    mouth_dist=mouth_dist,
                    max_area_ratio=max_area_ratio,
                    max_side_ratio=max_side_ratio,
                    x_margin=x_margin,
                    y_up=y_up,
                    y_down=y_down,
                    lower_y=lower_y
                ):
                    ok_assoc = True
                    break
            if ok_assoc:
                filtered_smokes.append(b)

    # Dedup (filtered)
    try:
        di = float(dedup_iou)
    except Exception:
        di = 0.0
    if di and di > 0:
        filtered_smokes = _dedup_iou(filtered_smokes, iou_thr=di)

    smoke_tracks = []
    for b in filtered_smokes:
        try:
            tr = create_tracker(tracker_pref)
            if tr is not None:
                tr.init(frame, _xyxy_to_xywh(b))
            smoke_tracks.append(SmokeTrack(tr, b))
        except Exception:
            continue

    face_tracks = []
    for (name, sim, fb) in parsed_faces:
        try:
            tr = create_tracker(tracker_pref)
            if tr is not None:
                tr.init(frame, _xyxy_to_xywh(fb))

            smoking = False
            for st in smoke_tracks:
                if _smoke_matches_face(
                    st.bbox, fb,
                    mouth_dist=mouth_dist,
                    max_area_ratio=max_area_ratio,
                    max_side_ratio=max_side_ratio,
                    x_margin=x_margin,
                    y_up=y_up,
                    y_down=y_down,
                    lower_y=lower_y
                ):
                    smoking = True
                    break

            face_tracks.append(FaceTrack(tr, fb, name, sim, smoking))
        except Exception:
            continue

    return face_tracks, smoke_tracks, raw_smokes


def update_tracks(
    frame,
    face_tracks,
    smoke_tracks,
    *,
    mouth_dist=1.35,
    max_area_ratio=0.25,
    max_side_ratio=1.20,
    x_margin=1.00,
    y_up=0.30,
    y_down=2.00,
    lower_y=0.35
):
    H, W = frame.shape[:2]

    new_smokes = []
    for st in smoke_tracks:
        try:
            if st.tracker is None:
                new_smokes.append(st)
                continue
            ok, bb = st.tracker.update(frame)
            if not ok:
                continue
            x, y, w, h = bb
            st.bbox = _clip_xyxy(_xywh_to_xyxy(x, y, w, h), W, H)
            new_smokes.append(st)
        except Exception:
            continue
    smoke_tracks[:] = new_smokes

    new_faces = []
    for ft in face_tracks:
        try:
            if ft.tracker is None:
                smoking = False
                for st in smoke_tracks:
                    if _smoke_matches_face(
                        st.bbox, ft.bbox,
                        mouth_dist=mouth_dist,
                        max_area_ratio=max_area_ratio,
                        max_side_ratio=max_side_ratio,
                        x_margin=x_margin,
                        y_up=y_up,
                        y_down=y_down,
                        lower_y=lower_y
                    ):
                        smoking = True
                        break
                ft.smoking = smoking
                new_faces.append(ft)
                continue

            ok, bb = ft.tracker.update(frame)
            if not ok:
                continue
            x, y, w, h = bb
            ft.bbox = _clip_xyxy(_xywh_to_xyxy(x, y, w, h), W, H)

            smoking = False
            for st in smoke_tracks:
                if _smoke_matches_face(
                    st.bbox, ft.bbox,
                    mouth_dist=mouth_dist,
                    max_area_ratio=max_area_ratio,
                    max_side_ratio=max_side_ratio,
                    x_margin=x_margin,
                    y_up=y_up,
                    y_down=y_down,
                    lower_y=lower_y
                ):
                    smoking = True
                    break
            ft.smoking = smoking

            new_faces.append(ft)
        except Exception:
            continue
    face_tracks[:] = new_faces


def draw_with_tracks(
    frame,
    face_tracks,
    smoke_tracks,
    raw_smokes,
    title,
    *,
    api_ok=True,
    api_err=None,
    api_age=0.0,
    api_lat=0.0,
    blur_on=True,
    yolo_used=None,
    draw_raw=True
):
    out = frame.copy()
    _put_shadow_text(out, title, (10, 36), scale=1.0, thickness=3)

    if yolo_used is None:
        yolo_used = {}
    yline = f"YOLO conf={yolo_used.get('conf')} imgsz={yolo_used.get('imgsz')} second={yolo_used.get('second_pass')} up={yolo_used.get('upscale')}"
    _put_shadow_text(out, yline, (10, 78), scale=0.75, thickness=3)

    status = f"API ok={api_ok} raw_smokes={len(raw_smokes)} smokes={len(smoke_tracks)} faces={len(face_tracks)}"
    if api_err:
        status += f" err={str(api_err)[:50]}"
    _put_shadow_text(out, status, (10, 112), scale=0.75, thickness=3)

    _put_shadow_text(out, f"API_AGE={api_age:.2f}s  LAT={api_lat:.2f}s",
                     (10, out.shape[0] - 18), scale=1.05, thickness=4)

    # RAW smoke (yellow)
    if draw_raw:
        for b in raw_smokes:
            x1, y1, x2, y2 = b
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, "RAW", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Filtered smoke (red)
    for st in smoke_tracks:
        x1, y1, x2, y2 = st.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "SMOKE", (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Blur: pixelate non-smoking faces
    if blur_on:
        for ft in face_tracks:
            if not ft.smoking:
                x1, y1, x2, y2 = ft.bbox
                pixelate_roi(out, x1, y1, x2, y2, blocks=20)

    # Faces (green)
    for ft in face_tracks:
        x1, y1, x2, y2 = ft.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = f"{ft.name} ({ft.sim:.2f})"
        cv2.putText(out, lbl, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if ft.smoking:
            cv2.putText(out, "SMOKING!", (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            cv2.putText(out, "NO", (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:5000")
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps_limit", type=float, default=25.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--save_dir", default="outputs/dual_preview")
    ap.add_argument("--analyze_every", type=int, default=5)
    ap.add_argument("--jpeg_quality", type=int, default=85)
    ap.add_argument("--sync_api", action="store_true")
    ap.add_argument("--no_blur", action="store_true")
    ap.add_argument("--tracker", default="auto", choices=["auto", "mosse", "kcf", "csrt", "none"])

    # FP filter controls
    ap.add_argument("--no_fp_filter", action="store_true")
    ap.add_argument("--draw_raw", action="store_true", help="Draw RAW smoke boxes in yellow")
    ap.add_argument("--dedup_iou", type=float, default=0.70, help="Deduplicate overlapping bboxes of the same cigarette by IoU (0=off)")

    # mouth-proximity filter parameters
    ap.add_argument("--mouth_dist", type=float, default=1.35)
    ap.add_argument("--max_area_ratio", type=float, default=0.25)
    ap.add_argument("--max_side_ratio", type=float, default=1.20)
    ap.add_argument("--x_margin", type=float, default=1.00)
    ap.add_argument("--y_up", type=float, default=0.30)
    ap.add_argument("--y_down", type=float, default=2.00)
    ap.add_argument("--lower_y", type=float, default=0.35)

    # fill the screen
    ap.add_argument("--fit_screen", action="store_true", help="Auto-scale to fit the screen")
    ap.add_argument("--screen_margin", type=int, default=120)

    args = ap.parse_args()
    args.analyze_every = max(1, int(args.analyze_every))

    root = Path.cwd()
    vids = root / "youtube_videos"
    if not vids.exists():
        raise SystemExit(f"ERROR: folder not found: {vids}")

    v1 = find_video(vids, ["video1", "video 1", "video_1", "cam1", "cam 1", "cam_1"])
    v2 = find_video(vids, ["video2", "video 2", "video_2", "cam2", "cam 2", "cam_2"])
    if v1 is None or v2 is None:
        raise SystemExit(f"ERROR: videos not found.\n  looked for: {vids}\\video1.* and {vids}\\video2.*")

    print("VIDEO1 =", v1)
    print("VIDEO2 =", v2)
    print("API    =", args.api)
    print("analyze_every =", args.analyze_every)
    print("jpeg_quality  =", args.jpeg_quality)
    print("MODE:", "SYNC" if args.sync_api else "ASYNC")
    print("TRACKER:", args.tracker)
    print("FP_FILTER:", "OFF" if args.no_fp_filter else "ON")
    print("FIT_SCREEN:", "ON" if args.fit_screen else "OFF")
    print("CONTROLS: Q=quit  SPACE=pause  S=screenshot")

    cap1 = cv2.VideoCapture(str(v1))
    cap2 = cv2.VideoCapture(str(v2))
    if not cap1.isOpened() or not cap2.isOpened():
        raise SystemExit("ERROR: could not open VideoCapture.")

    fps1 = _cap_fps(cap1)
    fps2 = _cap_fps(cap2)
    native_fps = min([f for f in [fps1, fps2] if f > 0] or [0.0])
    show_fps = args.fps_limit if args.fps_limit and args.fps_limit > 0 else native_fps
    print(f"VIDEO FPS: cam1={fps1:.2f} cam2={fps2:.2f} show_fps={show_fps:.2f}")

    save_dir = (root / args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paused = False
    frame_idx = 0
    last_tick = time.time()

    last_j1 = {"ok": True, "results": [], "smoking_boxes": [], "yolo_used": {}}
    last_j2 = {"ok": True, "results": [], "smoking_boxes": [], "yolo_used": {}}
    last_update_t1 = 0.0
    last_update_t2 = 0.0
    last_latency1 = 0.0
    last_latency2 = 0.0

    face_tracks1, smoke_tracks1, raw_smokes1 = [], [], []
    face_tracks2, smoke_tracks2, raw_smokes2 = [], [], []

    executor = ThreadPoolExecutor(max_workers=2)
    fut1 = None
    fut2 = None
    sent_t1 = 0.0
    sent_t2 = 0.0
    sent_frame1 = None
    sent_frame2 = None

    ok1, cur1 = cap1.read()
    ok2, cur2 = cap2.read()
    if not ok1 or cur1 is None or not ok2 or cur2 is None:
        raise SystemExit("ERROR: could not read the first frame.")

    win_name = "DUAL VIDEO SMOKING TEST (RAW+FILTER)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    both = None  # for screenshots

    # Warmup
    try:
        t0 = time.time()
        last_j1 = post_frame(args.api, cur1, "1", "video1", args.jpeg_quality)
        last_latency1 = time.time() - t0
        last_update_t1 = time.time()
        face_tracks1, smoke_tracks1, raw_smokes1 = rebuild_tracks_from_api(
            cur1, last_j1,
            tracker_pref=args.tracker,
            no_fp_filter=args.no_fp_filter,
            mouth_dist=args.mouth_dist,
            max_area_ratio=args.max_area_ratio,
            max_side_ratio=args.max_side_ratio,
            x_margin=args.x_margin,
            y_up=args.y_up,
            y_down=args.y_down,
            lower_y=args.lower_y,
            dedup_iou=args.dedup_iou,
        )

        t0 = time.time()
        last_j2 = post_frame(args.api, cur2, "2", "video2", args.jpeg_quality)
        last_latency2 = time.time() - t0
        last_update_t2 = time.time()
        face_tracks2, smoke_tracks2, raw_smokes2 = rebuild_tracks_from_api(
            cur2, last_j2,
            tracker_pref=args.tracker,
            no_fp_filter=args.no_fp_filter,
            mouth_dist=args.mouth_dist,
            max_area_ratio=args.max_area_ratio,
            max_side_ratio=args.max_side_ratio,
            x_margin=args.x_margin,
            y_up=args.y_up,
            y_down=args.y_down,
            lower_y=args.lower_y,
            dedup_iou=args.dedup_iou,
        )
    except Exception as e:
        print("[WARMUP] error:", e)

    try:
        while True:
            if not paused:
                ok1, f1 = cap1.read()
                ok2, f2 = cap2.read()

                if not ok1 or f1 is None:
                    if args.loop:
                        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok1, f1 = cap1.read()
                    else:
                        break
                if not ok2 or f2 is None:
                    if args.loop:
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok2, f2 = cap2.read()
                    else:
                        break

                cur1, cur2 = f1, f2
                do_analyze = (frame_idx % args.analyze_every == 0)

                if do_analyze:
                    if args.sync_api:
                        t0 = time.time()
                        try:
                            last_j1 = post_frame(args.api, cur1, "1", "video1", args.jpeg_quality)
                        except Exception as e:
                            last_j1 = {"ok": False, "error": str(e), "results": [], "smoking_boxes": [], "yolo_used": {}}
                        last_latency1 = time.time() - t0
                        last_update_t1 = time.time()
                        face_tracks1, smoke_tracks1, raw_smokes1 = rebuild_tracks_from_api(
                            cur1, last_j1,
                            tracker_pref=args.tracker,
                            no_fp_filter=args.no_fp_filter,
                            mouth_dist=args.mouth_dist,
                            max_area_ratio=args.max_area_ratio,
                            max_side_ratio=args.max_side_ratio,
                            x_margin=args.x_margin,
                            y_up=args.y_up,
                            y_down=args.y_down,
                            lower_y=args.lower_y,
            dedup_iou=args.dedup_iou,
                        )

                        t0 = time.time()
                        try:
                            last_j2 = post_frame(args.api, cur2, "2", "video2", args.jpeg_quality)
                        except Exception as e:
                            last_j2 = {"ok": False, "error": str(e), "results": [], "smoking_boxes": [], "yolo_used": {}}
                        last_latency2 = time.time() - t0
                        last_update_t2 = time.time()
                        face_tracks2, smoke_tracks2, raw_smokes2 = rebuild_tracks_from_api(
                            cur2, last_j2,
                            tracker_pref=args.tracker,
                            no_fp_filter=args.no_fp_filter,
                            mouth_dist=args.mouth_dist,
                            max_area_ratio=args.max_area_ratio,
                            max_side_ratio=args.max_side_ratio,
                            x_margin=args.x_margin,
                            y_up=args.y_up,
                            y_down=args.y_down,
                            lower_y=args.lower_y,
            dedup_iou=args.dedup_iou,
                        )
                    else:
                        if fut1 is None or fut1.done():
                            sent_t1 = time.time()
                            sent_frame1 = cur1
                            fut1 = executor.submit(post_frame, args.api, cur1, "1", "video1", args.jpeg_quality)
                        if fut2 is None or fut2.done():
                            sent_t2 = time.time()
                            sent_frame2 = cur2
                            fut2 = executor.submit(post_frame, args.api, cur2, "2", "video2", args.jpeg_quality)

                if fut1 is not None and fut1.done():
                    try:
                        last_j1 = fut1.result()
                    except Exception as e:
                        last_j1 = {"ok": False, "error": str(e), "results": [], "smoking_boxes": [], "yolo_used": {}}
                    last_update_t1 = time.time()
                    last_latency1 = max(0.0, last_update_t1 - sent_t1)
                    base_init = sent_frame1 if sent_frame1 is not None else cur1
                    face_tracks1, smoke_tracks1, raw_smokes1 = rebuild_tracks_from_api(
                        base_init, last_j1,
                        tracker_pref=args.tracker,
                        no_fp_filter=args.no_fp_filter,
                        mouth_dist=args.mouth_dist,
                        max_area_ratio=args.max_area_ratio,
                        max_side_ratio=args.max_side_ratio,
                        x_margin=args.x_margin,
                        y_up=args.y_up,
                        y_down=args.y_down,
                        lower_y=args.lower_y,
            dedup_iou=args.dedup_iou,
                    )
                    fut1 = None

                if fut2 is not None and fut2.done():
                    try:
                        last_j2 = fut2.result()
                    except Exception as e:
                        last_j2 = {"ok": False, "error": str(e), "results": [], "smoking_boxes": [], "yolo_used": {}}
                    last_update_t2 = time.time()
                    last_latency2 = max(0.0, last_update_t2 - sent_t2)
                    base_init = sent_frame2 if sent_frame2 is not None else cur2
                    face_tracks2, smoke_tracks2, raw_smokes2 = rebuild_tracks_from_api(
                        base_init, last_j2,
                        tracker_pref=args.tracker,
                        no_fp_filter=args.no_fp_filter,
                        mouth_dist=args.mouth_dist,
                        max_area_ratio=args.max_area_ratio,
                        max_side_ratio=args.max_side_ratio,
                        x_margin=args.x_margin,
                        y_up=args.y_up,
                        y_down=args.y_down,
                        lower_y=args.lower_y,
            dedup_iou=args.dedup_iou,
                    )
                    fut2 = None

                update_tracks(
                    cur1, face_tracks1, smoke_tracks1,
                    mouth_dist=args.mouth_dist,
                    max_area_ratio=args.max_area_ratio,
                    max_side_ratio=args.max_side_ratio,
                    x_margin=args.x_margin,
                    y_up=args.y_up,
                    y_down=args.y_down,
                    lower_y=args.lower_y,
                )
                update_tracks(
                    cur2, face_tracks2, smoke_tracks2,
                    mouth_dist=args.mouth_dist,
                    max_area_ratio=args.max_area_ratio,
                    max_side_ratio=args.max_side_ratio,
                    x_margin=args.x_margin,
                    y_up=args.y_up,
                    y_down=args.y_down,
                    lower_y=args.lower_y,
                )

                age1 = time.time() - last_update_t1 if last_update_t1 > 0 else 999.0
                age2 = time.time() - last_update_t2 if last_update_t2 > 0 else 999.0

                vis1 = draw_with_tracks(
                    cur1, face_tracks1, smoke_tracks1, raw_smokes1,
                    title="CAM-1 (video1)",
                    api_ok=bool(last_j1.get("ok", True)),
                    api_err=last_j1.get("error"),
                    api_age=age1,
                    api_lat=last_latency1,
                    blur_on=(not args.no_blur),
                    yolo_used=last_j1.get("yolo_used", {}),
                    draw_raw=args.draw_raw
                )
                vis2 = draw_with_tracks(
                    cur2, face_tracks2, smoke_tracks2, raw_smokes2,
                    title="CAM-2 (video2)",
                    api_ok=bool(last_j2.get("ok", True)),
                    api_err=last_j2.get("error"),
                    api_age=age2,
                    api_lat=last_latency2,
                    blur_on=(not args.no_blur),
                    yolo_used=last_j2.get("yolo_used", {}),
                    draw_raw=args.draw_raw
                )

                # rough scale by height first (optional)
                if args.height and args.height > 0:
                    vis1 = resize_to_height(vis1, args.height)
                    vis2 = resize_to_height(vis2, args.height)

                # pad to the same height
                h1, _ = vis1.shape[:2]
                h2, _ = vis2.shape[:2]
                H = max(h1, h2)
                if h1 != H:
                    vis1 = cv2.copyMakeBorder(vis1, 0, H - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                if h2 != H:
                    vis2 = cv2.copyMakeBorder(vis2, 0, H - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                both = cv2.hconcat([vis1, vis2])

                # fill the screen
                if args.fit_screen:
                    both = _fit_to_screen(both, margin=args.screen_margin)

                if show_fps and show_fps > 0:
                    elapsed = time.time() - last_tick
                    wait = max(0.0, (1.0 / show_fps) - elapsed)
                    if wait > 0:
                        time.sleep(wait)
                    last_tick = time.time()

                cv2.imshow(win_name, both)
                frame_idx += 1

            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == 32:
                paused = not paused
            if key in (ord("s"), ord("S")):
                if both is None:
                    print("⚠️ Screenshot failed: frame not ready.")
                    continue
                ts = time.strftime("%Y%m%d_%H%M%S")
                fn = save_dir / f"dual_{ts}_frame{frame_idx}.jpg"
                try:
                    cv2.imwrite(str(fn), both)
                    print("SAVED:", fn)
                except Exception as e:
                    print("SAVE ERR:", e)

    finally:
        executor.shutdown(wait=True)
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        print("DONE")


if __name__ == "__main__":
    main()