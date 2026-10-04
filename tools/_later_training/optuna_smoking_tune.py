# tools/_later_training/optuna_smoking_tune_sqlite.py
# Optuna tuner for the SAHI-sliced (SIHA) data, sqlite storage.
# Clean first start: --fresh 1
# Resume after a power cut / interruption: --fresh 0

import argparse
import csv
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import optuna
from ultralytics import YOLO
import yaml


# -----------------------------
# Utils
# -----------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # for speed
    except Exception:
        pass


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def find_results_csv(run_dir: Path) -> Optional[Path]:
    p = run_dir / "results.csv"
    if p.exists():
        return p
    for cand in run_dir.rglob("results.csv"):
        return cand
    return None


def pick_metric_columns(header: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    map5095 = None
    map50 = None
    recall = None
    precision = None

    candidates_map5095 = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
        "metrics/mAP50-95_box",
        "metrics/mAP50-95(Box)",
    ]
    candidates_map50 = [
        "metrics/mAP50(B)",
        "metrics/mAP50",
        "metrics/mAP50_box",
        "metrics/mAP50(Box)",
    ]
    candidates_recall = [
        "metrics/recall(B)",
        "metrics/recall",
        "metrics/recall_box",
        "metrics/recall(Box)",
    ]
    candidates_precision = [
        "metrics/precision(B)",
        "metrics/precision",
        "metrics/precision_box",
        "metrics/precision(Box)",
    ]

    for c in candidates_map5095:
        if c in header:
            map5095 = c
            break
    for c in candidates_map50:
        if c in header:
            map50 = c
            break
    for c in candidates_recall:
        if c in header:
            recall = c
            break
    for c in candidates_precision:
        if c in header:
            precision = c
            break

    if map5095 is None:
        for h in header:
            if "mAP50-95" in h:
                map5095 = h
                break
    if map50 is None:
        for h in header:
            if "mAP50" in h and "95" not in h:
                map50 = h
                break
    if recall is None:
        for h in header:
            if "recall" in h.lower():
                recall = h
                break
    if precision is None:
        for h in header:
            if "precision" in h.lower():
                precision = h
                break

    return map5095, map50, recall, precision


def read_best_score(run_dir: Path, recall_bonus: float) -> Tuple[Optional[float], Dict[str, float]]:
    metrics = {"map50_95": -1.0, "map50": -1.0, "recall": -1.0, "precision": -1.0}
    results_csv = find_results_csv(run_dir)
    if not results_csv:
        return None, metrics

    with open(results_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        col_map5095, col_map50, col_recall, col_precision = pick_metric_columns(header)

        best_map5095 = -1.0
        best_map50 = -1.0
        best_recall = -1.0
        best_precision = -1.0

        for row in reader:
            if col_map5095:
                v = safe_float(row.get(col_map5095, None), None)
                if v is not None:
                    best_map5095 = max(best_map5095, v)
            if col_map50:
                v = safe_float(row.get(col_map50, None), None)
                if v is not None:
                    best_map50 = max(best_map50, v)
            if col_recall:
                v = safe_float(row.get(col_recall, None), None)
                if v is not None:
                    best_recall = max(best_recall, v)
            if col_precision:
                v = safe_float(row.get(col_precision, None), None)
                if v is not None:
                    best_precision = max(best_precision, v)

    if best_map5095 < 0:
        return None, metrics

    metrics["map50_95"] = best_map5095
    metrics["map50"] = best_map50
    metrics["recall"] = best_recall
    metrics["precision"] = best_precision

    score = best_map5095 + float(recall_bonus) * max(0.0, best_recall)
    return score, metrics


def find_siha_data_yaml(project_root: Path) -> str:
    old_base = project_root / "runs" / "siha_day_time"
    args_files = sorted(old_base.rglob("args.yaml"))
    if not args_files:
        raise FileNotFoundError("No args.yaml found under runs/siha_day_time.")

    for f in args_files:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            data_path = data.get("data")
            if data_path:
                return str(data_path)
        except Exception:
            continue

    raise RuntimeError("Could not read the data path from the earlier SIHA runs.")


@dataclass
class TuneConfig:
    data: str
    models: List[str]
    trials: int
    epochs: int
    imgsz: int
    batch: int
    device: str
    workers: int
    seed: int
    project_dir: Path
    dataset_tag: str
    patience: int
    cos_lr: bool
    recall_bonus: float
    study_name: str
    fresh: bool


def build_search_space(trial: optuna.Trial) -> Dict:
    params = {}

    params["optimizer"] = trial.suggest_categorical("optimizer", ["SGD", "AdamW"])

    params["lr0"] = trial.suggest_float("lr0", 1e-4, 5e-2, log=True)
    params["lrf"] = trial.suggest_float("lrf", 0.01, 0.35)
    params["momentum"] = trial.suggest_float("momentum", 0.85, 0.98)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)

    params["warmup_epochs"] = trial.suggest_float("warmup_epochs", 0.0, 5.0)
    params["warmup_momentum"] = trial.suggest_float("warmup_momentum", 0.60, 0.95)
    params["warmup_bias_lr"] = trial.suggest_float("warmup_bias_lr", 0.0, 0.20)

    params["box"] = trial.suggest_float("box", 4.0, 12.0)
    params["cls"] = trial.suggest_float("cls", 0.05, 1.50)
    params["dfl"] = trial.suggest_float("dfl", 0.5, 2.0)

    params["hsv_h"] = trial.suggest_float("hsv_h", 0.0, 0.05)
    params["hsv_s"] = trial.suggest_float("hsv_s", 0.0, 0.9)
    params["hsv_v"] = trial.suggest_float("hsv_v", 0.0, 0.9)

    params["degrees"] = trial.suggest_float("degrees", 0.0, 8.0)
    params["translate"] = trial.suggest_float("translate", 0.0, 0.15)
    params["scale"] = trial.suggest_float("scale", 0.20, 0.70)
    params["shear"] = trial.suggest_float("shear", 0.0, 4.0)
    params["perspective"] = trial.suggest_float("perspective", 0.0, 0.001)

    params["flipud"] = trial.suggest_float("flipud", 0.0, 0.20)
    params["fliplr"] = trial.suggest_float("fliplr", 0.3, 0.7)

    params["mosaic"] = trial.suggest_float("mosaic", 0.0, 1.0)
    params["mixup"] = trial.suggest_float("mixup", 0.0, 0.25)
    params["copy_paste"] = trial.suggest_float("copy_paste", 0.0, 0.20)

    params["close_mosaic"] = trial.suggest_int("close_mosaic", 5, 20)

    return params


def objective_factory(cfg: TuneConfig):
    root = Path(__file__).resolve().parents[2]  # project root

    if cfg.data == "AUTO_FROM_runs/siha_day_time/args.yaml":
        data_yaml = find_siha_data_yaml(root)
    else:
        data_yaml = str((root / cfg.data).resolve()) if not Path(cfg.data).is_absolute() else cfg.data

    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    def objective(trial: optuna.Trial) -> float:
        set_global_seed(cfg.seed + trial.number)

        model_name = trial.suggest_categorical("model", cfg.models)
        hp = build_search_space(trial)

        mstem = Path(model_name).stem
        run_name = f"optuna__{mstem}__{cfg.dataset_tag}__t{trial.number:03d}__e{cfg.epochs}__img{cfg.imgsz}__b{cfg.batch}"

        model = YOLO(model_name)
        model.train(
            data=str(data_yaml),
            epochs=cfg.epochs,
            batch=cfg.batch,
            imgsz=cfg.imgsz,
            project=str(cfg.project_dir),
            name=run_name,
            exist_ok=True,
            device=str(cfg.device),
            workers=int(cfg.workers),
            seed=int(cfg.seed),
            patience=int(cfg.patience),
            cache=False,
            cos_lr=bool(cfg.cos_lr),
            plots=False,
            verbose=False,
            amp=False,
            deterministic=False,
            **hp,
        )

        run_dir = cfg.project_dir / run_name
        score, metrics = read_best_score(run_dir, cfg.recall_bonus)

        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("metrics", metrics)

        if score is None:
            return -1.0

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return float(score)

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="AUTO_FROM_runs/siha_day_time/args.yaml")
    ap.add_argument("--models", nargs="*", default=["yolo11m.pt", "yolo11s.pt"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset_tag", default="siha_day_time")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cos_lr", type=int, default=1)
    ap.add_argument("--recall_bonus", type=float, default=0.05)
    ap.add_argument("--study_name", default="optuna_smoke_siha_ms")
    ap.add_argument("--fresh", type=int, default=0, help="1 = delete the old sqlite db and trial folders and start from scratch")

    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    project_dir = root / "runs" / "optuna_smoke_siha"
    project_dir.mkdir(parents=True, exist_ok=True)

    cfg = TuneConfig(
        data=args.data,
        models=args.models,
        trials=args.trials,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project_dir=project_dir,
        dataset_tag=args.dataset_tag,
        patience=args.patience,
        cos_lr=bool(args.cos_lr),
        recall_bonus=float(args.recall_bonus),
        study_name=args.study_name,
        fresh=bool(args.fresh),
    )

    print("=== OPTUNA START (SIHA + SQLITE) ===")
    print("data:", cfg.data)
    print("models:", cfg.models)
    print("trials:", cfg.trials, "epochs:", cfg.epochs, "imgsz:", cfg.imgsz, "batch:", cfg.batch)
    print("project_dir:", cfg.project_dir)
    print("recall_bonus:", cfg.recall_bonus)
    print("study_name:", cfg.study_name)
    print("fresh:", cfg.fresh)

    study_db = cfg.project_dir / "study.db"

    if cfg.fresh:
        if study_db.exists():
            study_db.unlink()
        for d in cfg.project_dir.glob("optuna__*"):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed, multivariate=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=cfg.study_name,
        storage=f"sqlite:///{study_db.as_posix()}",
        load_if_exists=True,
    )

    t0 = time.time()
    study.optimize(objective_factory(cfg), n_trials=cfg.trials, show_progress_bar=True)
    dt = time.time() - t0

    best = study.best_trial
    best_params = dict(best.params)
    best_metrics = best.user_attrs.get("metrics", {})
    best_run_dir = best.user_attrs.get("run_dir", "")

    out_dir = root / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "optuna_best_smoke_siha.json"
    out_json.write_text(
        json.dumps(
            {
                "best_value": best.value,
                "best_params": best_params,
                "best_metrics": best_metrics,
                "best_run_dir": best_run_dir,
                "seconds_total": dt,
                "study_db": str(study_db),
                "study_name": cfg.study_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== OPTUNA DONE ===")
    print("best_value:", best.value)
    print("best_params:", json.dumps(best_params, ensure_ascii=False, indent=2))
    print("best_metrics:", json.dumps(best_metrics, ensure_ascii=False, indent=2))
    print("best_run_dir:", best_run_dir)
    print("study_db:", study_db)
    print("saved:", out_json)

    model_name = best_params.get("model", "yolo11m.pt")
    extra_items = []
    skip_keys = {"model"}
    for k, v in best_params.items():
        if k in skip_keys:
            continue
        extra_items.append(f'--extra "{k}={v}"')

    data_for_cmd = cfg.data
    cmd = (
        f'python train_smoking_detection.py --data "{data_for_cmd}" --model "{model_name}" '
        f"--epochs 80 --imgsz {cfg.imgsz} --batch {cfg.batch} --device {cfg.device} --workers {cfg.workers} "
        f'--dataset_tag "{cfg.dataset_tag}" '
        + " ".join(extra_items)
        + ' --extra "patience=30" --extra "cache=False" --extra "cos_lr=True"'
    )

    print("\n=== FINAL TRAIN COMMAND (copy-paste) ===")
    print(cmd)


if __name__ == "__main__":
    main()
