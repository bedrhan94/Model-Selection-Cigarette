import os
import time
import json
import argparse
from pathlib import Path

import cv2

try:
    import requests
except Exception:
    requests = None


def _cap_fps(cap):
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6 or fps > 240:
        return 0.0
    return fps


def _clip_xyxy(xyxy, W, H):
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(x1), int(y1), int(x2), int(y2)]


def _xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float(x1), float(y1), float(max(1, x2 - x1)), float(max(1, y2 - y1)))


def _parse_box_any(obj):
    # supports: [x1,y1,x2,y2]  or {"bbox":[...]} or {"xyxy":[...]} etc.
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
    return inter / union if union > 0 else 0.0


def dedup_iou(boxes, thr=0.70):
    kept = []
    for b in boxes:
        if all(_iou_xyxy(b, k) < thr for k in kept):
            kept.append(b)
    return kept


def create_tracker(pref="auto"):
    pref = (pref or "auto").lower().strip()

    if pref in ("none", "off", "0"):
        return None

    def _mk(name):
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, f"Tracker{name}_create"):
                return getattr(cv2.legacy, f"Tracker{name}_create")()
            if hasattr(cv2, f"Tracker{name}_create"):
                return getattr(cv2, f"Tracker{name}_create")()
        except Exception:
            return None
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

    # drop large boxes (railings etc.)
    if sw > fw * max_side_ratio or sh > fh * max_side_ratio:
        return False
    face_area = fw * fh
    smoke_area = sw * sh
    if smoke_area > face_area * max_area_ratio:
        return False

    scx = (sx1 + sx2) * 0.5
    scy = (sy1 + sy2) * 0.5

    ex1 = fx1 - fw * x_margin
    ex2 = fx2 + fw * x_margin
    ey1 = fy1 - fh * y_up
    ey2 = fy2 + fh * y_down

    if not (ex1 <= scx <= ex2 and ey1 <= scy <= ey2):
        return False

    # mouth/chin estimate
    mcx = (fx1 + fx2) * 0.5
    mcy = fy1 + fh * 0.65

    # must not be too far above
    if scy < fy1 + fh * lower_y:
        return False

    dx = scx - mcx
    dy = scy - mcy
    d = (dx * dx + dy * dy) ** 0.5 / float(fh)
    return d <= mouth_dist


def parse_api(frame, api_json, *,
              no_fp_filter=False,
              dedup_thr=0.70,
              mouth_dist=1.35,
              max_area_ratio=0.25,
              max_side_ratio=1.20,
              x_margin=1.00,
              y_up=0.30,
              y_down=2.00,
              lower_y=0.35,
              max_faces=8,
              max_smokes=12):
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

    if dedup_thr and dedup_thr > 0:
        raw_smokes = dedup_iou(raw_smokes, thr=float(dedup_thr))

    # FP filter
    filtered_smokes = []
    if no_fp_filter or not face_boxes:
        filtered_smokes = list(raw_smokes)
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

    if dedup_thr and dedup_thr > 0:
        filtered_smokes = dedup_iou(filtered_smokes, thr=float(dedup_thr))

    return parsed_faces, raw_smokes, filtered_smokes


def draw_overlay(frame, parsed_faces, raw_smokes, filtered_smokes, *,
                 draw_raw=False,
                 mouth_dist=1.35,
                 max_area_ratio=0.25,
                 max_side_ratio=1.20,
                 x_margin=1.00,
                 y_up=0.30,
                 y_down=2.00,
                 lower_y=0.35):
    H, W = frame.shape[:2]

    # smoke boxes
    if draw_raw:
        for b in raw_smokes:
            x1, y1, x2, y2 = b
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # yellow
            cv2.putText(frame, "RAW", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    for b in filtered_smokes:
        x1, y1, x2, y2 = b
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
        cv2.putText(frame, "SMOKE", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # faces + smoking flag
    for (name, sim, fb) in parsed_faces:
        fx1, fy1, fx2, fy2 = fb
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        label = f"{name} ({sim:.2f})"
        cv2.putText(frame, label, (fx1, max(0, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        smoking = False
        for sb in filtered_smokes:
            if _smoke_matches_face(
                sb, fb,
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
        if smoking:
            cv2.putText(frame, "SMOKING", (fx1, min(H - 10, fy2 + 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:5000")
    ap.add_argument("--video", default="youtube_videos/video1.mp4")
    ap.add_argument("--fps_limit", type=float, default=25.0)
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

    ap.add_argument("--fit_screen", action="store_true")
    ap.add_argument("--screen_margin", type=int, default=120)

    ap.add_argument("--save_mp4", default="", help="e.g. outputs/01_generated_review_videos/annotated.mp4")
    ap.add_argument("--no_show", action="store_true")

    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    vpath = (root / args.video).resolve()

    if not vpath.exists():
        raise SystemExit(f"ERROR: video not found: {vpath}")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise SystemExit("ERROR: could not open VideoCapture.")

    native_fps = _cap_fps(cap) or 25.0
    show_fps = args.fps_limit if args.fps_limit and args.fps_limit > 0 else native_fps

    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("ERROR: could not read the first frame.")
    H, W = frame.shape[:2]

    writer = None
    if args.save_mp4:
        out_path = (root / args.save_mp4).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, native_fps, (W, H))
        if not writer.isOpened():
            raise SystemExit(f"ERROR: could not open VideoWriter: {out_path}")
        print("SAVE ->", out_path)

    print("VIDEO  =", vpath)
    print("API    =", args.api)
    print("FPS    =", native_fps, "show_fps=", show_fps)
    print("MODE   =", "SYNC" if args.sync_api else "ASYNC")
    print("CONTROLS: Q=quit  SPACE=pause  S=screenshot")

    paused = False
    frame_idx = 0
    last_api_json = {"ok": True, "results": [], "smoking_boxes": []}
    last_api_t = 0.0
    last_latency = 0.0

    # warmup
    t0 = time.time()
    last_api_json = post_frame(args.api, frame, "1", "video", args.jpeg_quality)
    last_latency = time.time() - t0
    last_api_t = time.time()

    screenshot_dir = (root / "outputs" / "01_generated_review_videos" / "screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1

            do_analyze = (frame_idx % max(1, int(args.analyze_every)) == 0)

            if do_analyze:
                if args.sync_api:
                    t0 = time.time()
                    last_api_json = post_frame(args.api, frame, "1", "video", args.jpeg_quality)
                    last_latency = time.time() - t0
                    last_api_t = time.time()
                else:
                    # simple ASYNC approach: still called like sync, load is controlled via fps_limit/analyze_every
                    t0 = time.time()
                    last_api_json = post_frame(args.api, frame, "1", "video", args.jpeg_quality)
                    last_latency = time.time() - t0
                    last_api_t = time.time()

            parsed_faces, raw_smokes, filtered_smokes = parse_api(
                frame, last_api_json,
                no_fp_filter=args.no_fp_filter,
                dedup_thr=args.dedup_iou,
                mouth_dist=args.mouth_dist,
                max_area_ratio=args.max_area_ratio,
                max_side_ratio=args.max_side_ratio,
                x_margin=args.x_margin,
                y_up=args.y_up,
                y_down=args.y_down,
                lower_y=args.lower_y
            )

            show = frame.copy()
            draw_overlay(
                show, parsed_faces, raw_smokes, filtered_smokes,
                draw_raw=args.draw_raw,
                mouth_dist=args.mouth_dist,
                max_area_ratio=args.max_area_ratio,
                max_side_ratio=args.max_side_ratio,
                x_margin=args.x_margin,
                y_up=args.y_up,
                y_down=args.y_down,
                lower_y=args.lower_y
            )

            aplage = max(0.0, time.time() - last_api_t)
            cv2.putText(show, f"APLAGE={aplage:.2f}s  LAT={last_latency:.2f}s  every={args.analyze_every}",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer is not None:
                writer.write(show)

            if not args.no_show:
                disp = show
                if args.fit_screen:
                    # fit to screen (lowers quality, viewing only)
                    max_w = max(320, int(cv2.getWindowImageRect("VIDEO REVIEW")[2] if cv2.getWindowProperty("VIDEO REVIEW", 0) >= 0 else 0))
                    # Simple: the window cannot be measured before it opens, so avoid scaling
                    # with fit_screen, enlarge the window manually.
                    pass

                cv2.imshow("VIDEO REVIEW (FILTER)", show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:  # SPACE
                    paused = not paused
                elif key == ord("s"):
                    out = screenshot_dir / f"shot_{int(time.time())}.jpg"
                    cv2.imwrite(str(out), show)
                    print("screenshot ->", out)

            # fps_limit
            if show_fps and show_fps > 0:
                time.sleep(max(0.0, (1.0 / show_fps) - 0.0005))

        else:
            if not args.no_show:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:
                    paused = not paused
                elif key == ord("s"):
                    out = screenshot_dir / f"shot_{int(time.time())}.jpg"
                    cv2.imwrite(str(out), frame)
                    print("screenshot ->", out)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()