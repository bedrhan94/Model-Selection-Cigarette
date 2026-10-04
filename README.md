# Model-Selection-Cigarette

Code and training records for a SAHI-assisted YOLO model selection study on cigarette detection in camera footage.

The dataset, labels, videos, model weights and any image showing a person are not included. The recordings contain real people, so they are not shared. The repository holds the scripts, configs, training records (`args.yaml`, `results.csv`) and the plots generated from them.

## Layout

| Folder | Contents |
|---|---|
| `tools/` | Video evaluation, GT conversion, reporting and SAHI slice preparation scripts |
| `tools/_later_training/` | Optuna hyperparameter search, hard-negative frame extraction |
| `configs/` | Data and sweep configs, best Optuna parameters |
| `scenario_splits/`, `SIHA_senario_splits/` | `data.yaml` files for the full-frame and SAHI-sliced splits (image lists not included) |
| `thesis_experiments/P1_architecture_screening/` | Screening of 7 architectures on full-frame data, 30 epochs |
| `runs/siha_day_time/` | Screening of 7 architectures on SAHI-sliced data, 30 epochs |
| `runs/siha_day_time_rerun_80e/` | 80-epoch long runs (stopped manually, see below) |
| `runs/optuna_smoke_siha/` | Optuna search: 11 trials of 15 epochs, `study.db` and `optuna_trials.csv` |
| `runs/final/` | Final YOLO11m training with the Optuna parameters |
| `outputs/REVISION_3SEED_SAHI_ABLATION_FIXEDDATA/` | 3-seed SAHI ablation (YOLO11m/s, full frame vs sliced) |
| `outputs/REVISION_TEST_EVAL/` | Validation and test set results of the final model |
| `outputs/04_metrics_active/` | Per-hour video evaluation and false-alarm audit summaries |
| `exports/` | Early YOLO11n/s runs from January 2026 (public data, dsV6) |

Every training folder has a `TRAINING_STATUS.txt` stating whether the run completed, where it stopped and its best epoch.

"SIHA" in folder names (`SIHA_senario_splits`, `runs/siha_day_time`, `FINAL_SIHA_DAY_TIME`, ...) refers to the SAHI-sliced version of the data.

## Training status

The best epoch is chosen by the Ultralytics fitness score (0.1 x mAP50 + 0.9 x mAP50-95). Metrics are on the validation set, in percent.

**Stopped manually**

- The three runs under `runs/siha_day_time_rerun_80e/` were started for 80 epochs with patience=0 (early stopping off) and were stopped by hand at epochs 52, 41 and 62. `results.csv` covers every epoch up to that point; `results.png` and the final evaluation plots were not produced for these runs.
- Optuna trial t007 was stopped by hand at epoch 4. It is marked FAIL in `study.db` and was not used for parameter selection.

**Stopped by early stopping** (no manual intervention): `runs/final` (48/80, patience=30) and `exports/yolo11s` (71/80, patience=20).

`outputs/REVISION_3SEED_SAHI_ABLATION/` is the first attempt. Because of a data path error none of its 12 runs started; the same runs were redone under `REVISION_3SEED_SAHI_ABLATION_FIXEDDATA` and all of them completed.

| Run | Model | Epochs | patience | Best | mAP50 | mAP50-95 | Status |
|---|---|---|---|---|---|---|---|
| **P1 screening, full frame** | | | | | | | |
| rtdetr_l_day_time | rtdetr-l | 30/30 | 0 | 25 | 77.76 | 47.32 | Completed |
| yolo11m_day_time | yolo11m | 30/30 | 5 | 28 | 80.37 | 50.78 | Completed |
| yolo11n_day_time | yolo11n | 30/30 | 5 | 29 | 75.68 | 46.18 | Completed |
| yolo11s_day_time | yolo11s | 30/30 | 5 | 29 | 78.68 | 49.00 | Completed |
| yolov8m_day_time | yolov8m | 30/30 | 5 | 26 | 79.77 | 50.41 | Completed |
| yolov8n_day_time | yolov8n | 30/30 | 5 | 29 | 75.04 | 45.95 | Completed |
| yolov8s_day_time | yolov8s | 30/30 | 5 | 27 | 77.40 | 48.21 | Completed |
| **SAHI-sliced screening** | | | | | | | |
| 01_yolo11n_siha_day_time_e30 | yolo11n | 30/30 | 100 | 30 | 83.94 | 53.65 | Completed |
| 02_yolo11s_siha_day_time_e30 | yolo11s | 30/30 | 100 | 26 | 85.05 | 55.68 | Completed |
| 03_yolo11m_siha_day_time_e30 | yolo11m | 30/30 | 100 | 25 | 85.89 | 57.31 | Completed |
| 04_yolov8n_siha_day_time_e30 | yolov8n | 30/30 | 100 | 30 | 82.59 | 52.23 | Completed |
| 05_yolov8s_siha_day_time_e30 | yolov8s | 30/30 | 100 | 26 | 83.34 | 53.75 | Completed |
| 06_yolov8m_siha_day_time_e30 | yolov8m | 30/30 | 100 | 24 | 84.73 | 56.00 | Completed |
| 07_rtdetr_l_siha_day_time_e30 | rtdetr-l | 30/30 | 100 | 15 | 84.34 | 53.49 | Completed |
| **Long runs, 80 epochs** | | | | | | | |
| 01_yolo11m_siha_day_time_e80_pat0 | yolo11m | 52/80 | 0 | 34 | 86.65 | 57.88 | Stopped manually |
| 03_yolo11s_siha_day_time_e80_pat0 | yolo11s | 41/80 | 0 | 39 | 85.00 | 55.58 | Stopped manually |
| 04_yolo11n_siha_day_time_e80_pat0 | yolo11n | 62/80 | 0 | 41 | 83.93 | 53.77 | Stopped manually |
| **Optuna** | | | | | | | |
| t000 | yolo11s | 15/15 | 20 | 14 | 84.09 | 52.73 | Completed |
| t001 | yolo11m | 15/15 | 20 | 14 | 87.19 | 57.18 | Completed |
| t002 | yolo11s | 15/15 | 20 | 15 | 82.95 | 51.61 | Completed |
| t003 | yolo11s | 15/15 | 20 | 15 | 86.25 | 53.17 | Completed |
| t004 | yolo11m | 15/15 | 20 | 15 | 76.01 | 42.39 | Completed |
| t005 | yolo11m | 15/15 | 20 | 10 | 87.24 | 54.78 | Completed |
| t006 | yolo11s | 15/15 | 20 | 13 | 85.76 | 54.01 | Completed |
| t007 | yolo11s | 4/15 | 20 | 2 | 73.20 | 42.13 | Stopped manually |
| t008 | yolo11s | 15/15 | 20 | 11 | 84.27 | 52.65 | Completed |
| t009 | yolo11m | 15/15 | 20 | 13 | 87.03 | 57.19 | Completed (best trial) |
| t010 | yolo11s | 15/15 | 20 | 15 | 83.47 | 51.83 | Completed |
| **Final** | | | | | | | |
| 01_yolo11m_siha_optuna_final_e80 | yolo11m | 48/80 | 30 | 18 | 87.53 | 57.94 | Early stopping (patience=30) |
| **3-seed SAHI ablation** | | | | | | | |
| YOLO11m_full_frame seed42 | yolo11m | 30/30 | 100 | 30 | 80.05 | 50.48 | Completed |
| YOLO11m_full_frame seed123 | yolo11m | 30/30 | 100 | 28 | 79.74 | 50.40 | Completed |
| YOLO11m_full_frame seed2025 | yolo11m | 30/30 | 100 | 29 | 79.85 | 50.34 | Completed |
| YOLO11m_sahi_sliced seed42 | yolo11m | 30/30 | 100 | 25 | 86.70 | 57.90 | Completed |
| YOLO11m_sahi_sliced seed123 | yolo11m | 30/30 | 100 | 23 | 86.07 | 57.28 | Completed |
| YOLO11m_sahi_sliced seed2025 | yolo11m | 30/30 | 100 | 27 | 86.27 | 57.16 | Completed |
| YOLO11s_full_frame seed42 | yolo11s | 30/30 | 100 | 30 | 79.23 | 49.04 | Completed |
| YOLO11s_full_frame seed123 | yolo11s | 30/30 | 100 | 30 | 78.94 | 49.32 | Completed |
| YOLO11s_full_frame seed2025 | yolo11s | 30/30 | 100 | 29 | 78.81 | 48.77 | Completed |
| YOLO11s_sahi_sliced seed42 | yolo11s | 30/30 | 100 | 26 | 85.70 | 55.69 | Completed |
| YOLO11s_sahi_sliced seed123 | yolo11s | 30/30 | 100 | 28 | 85.39 | 55.90 | Completed |
| YOLO11s_sahi_sliced seed2025 | yolo11s | 30/30 | 100 | 24 | 85.17 | 55.64 | Completed |
| **Early runs (dsV6)** | | | | | | | |
| yolo11n__dsV6__e80 | yolo11n | 80/80 | 20 | 66 | 83.10 | 52.31 | Completed |
| yolo11s__dsV6__e80 | yolo11s | 71/80 | 20 | 51 | 84.13 | 54.11 | Early stopping (patience=20) |

## Results

SAHI ablation, 3 seeds (42 / 123 / 2025), mean ± std. Paired t-test details are in `outputs/REVISION_3SEED_SAHI_ABLATION_FIXEDDATA/ablation_table.md`.

| Model | Layout | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO11m | full frame | 79.88 ± 1.00 | 73.45 ± 0.91 | 79.88 ± 0.16 | 50.41 ± 0.07 |
| YOLO11m | SAHI sliced | 80.98 ± 0.93 | 81.50 ± 0.59 | 86.35 ± 0.33 | 57.44 ± 0.40 |
| YOLO11s | full frame | 79.05 ± 1.03 | 72.65 ± 0.66 | 78.99 ± 0.22 | 49.04 ± 0.27 |
| YOLO11s | SAHI sliced | 79.80 ± 0.51 | 80.98 ± 0.63 | 85.42 ± 0.26 | 55.74 ± 0.14 |

Final YOLO11m (Optuna parameters, best epoch 18):

| Split | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validation (6,600 slices) | 81.68 | 82.56 | 87.53 | 57.95 |
| Test (5,359 slices) | 85.27 | 80.21 | 87.46 | 56.75 |

## Environment

Python 3.10.0, Ultralytics 8.3.240, PyTorch 2.7.1+cu128, NVIDIA RTX 5080 (16 GB).

## Notes

- The absolute project path in `args.yaml`, `data.yaml`, the log files and `study.db` was replaced with `<PROJECT_ROOT>`. Point the `data.yaml` files at your own data before running anything.
- `tools/eval_video_metrics.py` and `tools/video_smoking_review*.py` send video frames to a separate API server that is not part of this repository.
