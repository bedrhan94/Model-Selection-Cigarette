from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / r"outputs\03_predictions_jsonl\FALSE_POSITIVE_DAY_TIME"
OUT_CSV = ROOT / r"outputs\06_summary_tables\FALSE_POSITIVE_DAY_TIME\fp_day_time_summary.csv"

FPS = 30.0

# runtime / api event logic
SMOKE_HOLD_SEC = 1.4
STOP_MISS_FRAMES = 15
EVENT_COOLDOWN_SEC = 10.0

def load_pred_binary(p: Path):
    arr = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln)
            arr.append(1 if (d.get("pred") or []) else 0)
    return arr

def runtime_events(pred):
    start_needed = round(SMOKE_HOLD_SEC * FPS)
    cooldown_frames = round(EVENT_COOLDOWN_SEC * FPS)

    events = []
    in_event = False
    run = 0
    miss = 0
    cooldown_until = -1
    start = None

    for i, v in enumerate(pred):
        if i < cooldown_until:
            continue

        if not in_event:
            if v == 1:
                run += 1
                if run >= start_needed:
                    start = i - start_needed + 1
                    in_event = True
                    miss = 0
            else:
                run = 0
        else:
            if v == 1:
                miss = 0
            else:
                miss += 1
                if miss >= STOP_MISS_FRAMES:
                    end = i - STOP_MISS_FRAMES
                    dur = (end - start + 1) / FPS
                    events.append((start, end, dur))
                    in_event = False
                    run = 0
                    miss = 0
                    cooldown_until = i + cooldown_frames
                    start = None

    if in_event and start is not None:
        end = len(pred) - 1
        dur = (end - start + 1) / FPS
        events.append((start, end, dur))

    return events

rows = []

for hour in range(9, 15):
    p = PRED_DIR / f"pred_fp_{hour}.jsonl"
    if not p.exists():
        print(f"Missing file: {p}")
        continue

    pred = load_pred_binary(p)

    total_frames = len(pred)
    fp_frames = sum(pred)
    fp_seconds = fp_frames / FPS
    clean_seconds = (total_frames - fp_frames) / FPS
    fp_ratio = fp_frames / total_frames if total_frames else 0.0

    events = runtime_events(pred)
    false_event_count = len(events)
    longest_false_event_s = max((e[2] for e in events), default=0.0)

    rows.append({
        "Hour": hour,
        "FP_ratio": round(fp_ratio, 3),
        "Total_duration_s": round(total_frames / FPS, 1),
        "FP_duration_s": round(fp_seconds, 1),
        "Clean_duration_s": round(clean_seconds, 1),
        "FP_frame": fp_frames,
        "False_event": false_event_count,
        "Longest_false_event_s": round(longest_false_event_s, 1),
    })

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Hour",
            "FP_ratio",
            "Total_duration_s",
            "FP_duration_s",
            "Clean_duration_s",
            "FP_frame",
            "False_event",
            "Longest_false_event_s",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("OK ->", OUT_CSV)
for r in rows:
    print(r)