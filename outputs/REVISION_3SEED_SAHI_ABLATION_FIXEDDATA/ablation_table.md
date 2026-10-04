### Controlled SAHI ablation (e30, b8, img640, deterministic; n=3 seeds: 42/123/2025)

| Model | Layout | P (%) | R (%) | mAP@50 (%) | mAP@50-95 (%) |
|---|---|---|---|---|---|
| YOLO11m | full-frame | 79.88 ± 1.00 | 73.45 ± 0.91 | 79.88 ± 0.16 | 50.41 ± 0.07 |
| YOLO11m | SAHI sliced | 80.98 ± 0.93 | 81.50 ± 0.59 | 86.35 ± 0.33 | 57.44 ± 0.40 |
| YOLO11s | full-frame | 79.05 ± 1.03 | 72.65 ± 0.66 | 78.99 ± 0.22 | 49.04 ± 0.27 |
| YOLO11s | SAHI sliced | 79.80 ± 0.51 | 80.98 ± 0.63 | 85.42 ± 0.26 | 55.74 ± 0.14 |

**SAHI gain (SAHI − full-frame, seed-paired t-test, df=2):**

| Model | Metric | Delta (points) | t(2) | Result |
|---|---|---|---|---|
| YOLO11m | mAP@50 | +6.46 ± 0.17 | 66.61 | p<0.01 |
| YOLO11m | mAP@50-95 | +7.04 ± 0.33 | 36.73 | p<0.01 |
| YOLO11m | Recall | +8.05 ± 1.30 | 10.71 | p<0.01 |
| YOLO11m | Precision | +1.10 ± 1.91 | 1.00 | not significant |
| YOLO11s | mAP@50 | +6.42 ± 0.05 | 202.82 | p<0.01 |
| YOLO11s | mAP@50-95 | +6.70 ± 0.15 | 77.38 | p<0.01 |
| YOLO11s | Recall | +8.33 ± 0.86 | 16.73 | p<0.01 |
| YOLO11s | Precision | +0.76 ± 0.54 | 2.43 | not significant |
