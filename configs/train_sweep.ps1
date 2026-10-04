param(
    [string]$ConfigPath = ".\configs\sweep_p1.yaml"
)

$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $PROJECT

Import-Module powershell-yaml

if (!(Test-Path $ConfigPath)) {
    throw "Config not found: $ConfigPath"
}

$cfg = ConvertFrom-Yaml (Get-Content $ConfigPath -Raw)

$phaseRoot  = Join-Path $PROJECT ($cfg.export.root.Replace("/", "\"))
$dataYaml   = Join-Path $PROJECT ($cfg.train.data_yaml.Replace("/", "\"))
$epochs     = [int]$cfg.train.epochs
$imgsz      = [int]$cfg.train.imgsz
$batch      = [int]$cfg.train.batch
$device     = "$($cfg.train.device)"
$workers    = [int]$cfg.train.workers
$seed       = [int]$cfg.train.seed
$datasetTag = "$($cfg.project.dataset_tag)"
$models     = @($cfg.models)
$indexFile  = Join-Path $PROJECT ($cfg.export.index_csv.Replace("/", "\"))

function Ensure-Dir($p) { New-Item -ItemType Directory -Force -Path $p | Out-Null }

Ensure-Dir $phaseRoot
Ensure-Dir (Join-Path $PROJECT "runs")

if ($cfg.export.write_index_csv -and !(Test-Path $indexFile)) {
    "timestamp,model_family,experiment_dir,run_dir,best_pt" | Out-File $indexFile -Encoding utf8
}

$extraArgs = @()
if ($cfg.train.extra_args) {
    foreach ($a in $cfg.train.extra_args) {
        $extraArgs += "--extra"
        $extraArgs += "$a"
    }
}

foreach ($baseModel in $models) {

    $modelFamily = [System.IO.Path]::GetFileNameWithoutExtension($baseModel)
    $expName     = "${modelFamily}_${datasetTag}"
    $runName     = "${modelFamily}__${datasetTag}__e${epochs}__img${imgsz}__b${batch}"
    $ts          = Get-Date -Format "yyyy-MM-dd_HHmmss"

    $expDir      = Join-Path $phaseRoot $expName
    $notesDir    = Join-Path $expDir "00_notes"
    $trainDir    = Join-Path $expDir "01_train_run"
    $ckptDir     = Join-Path $expDir "02_checkpoints"
    $optunaDir   = Join-Path $expDir "03_optuna"
    $metricsDir  = Join-Path $expDir "08_metrics"
    $logsDir     = Join-Path $expDir "12_logs"
    $exportsDir  = Join-Path $expDir "13_exports"

    foreach ($d in @($notesDir,$trainDir,$ckptDir,$optunaDir,$metricsDir,$logsDir,$exportsDir)) {
        Ensure-Dir $d
    }

    Write-Host "============================================================"
    Write-Host "TRAIN START -> $baseModel"
    Write-Host "EXPERIMENT  -> $expName"
    Write-Host "============================================================"

    $runDir = Join-Path $PROJECT "runs\$runName"
    if (Test-Path $runDir) {
        Remove-Item -Recurse -Force $runDir
    }

    python .\train_smoking_detection.py `
        --data "$dataYaml" `
        --model "$baseModel" `
        --epochs $epochs `
        --batch $batch `
        --imgsz $imgsz `
        --name "$runName" `
        --dataset_tag "$datasetTag" `
        --device "$device" `
        --workers $workers `
        --seed $seed `
        @extraArgs

    if (!(Test-Path $runDir)) {
        Write-Host "Run dir not found: $runDir" -ForegroundColor Yellow
        continue
    }

    Get-ChildItem $trainDir -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item $runDir $trainDir -Recurse -Force

    Copy-Item "$runDir\weights\best.pt" "$ckptDir\best.pt" -Force -ErrorAction SilentlyContinue
    Copy-Item "$runDir\weights\last.pt" "$ckptDir\last.pt" -Force -ErrorAction SilentlyContinue

    Copy-Item $dataYaml (Join-Path $notesDir "data_used.yaml") -Force -ErrorAction SilentlyContinue
    Copy-Item $ConfigPath (Join-Path $notesDir "sweep_used.yaml") -Force -ErrorAction SilentlyContinue
    if (Test-Path ".\configs\realtime.yaml") {
        Copy-Item ".\configs\realtime.yaml" (Join-Path $notesDir "realtime_snapshot.yaml") -Force -ErrorAction SilentlyContinue
    }

    @"
MODEL CARD
Timestamp: $ts
ModelFamily: $modelFamily
BaseModel: $baseModel
Experiment: $expName
DatasetTag: $datasetTag

Train:
- epochs=$epochs
- imgsz=$imgsz
- batch=$batch
- device=$device
- workers=$workers
- seed=$seed
- extra_args: $($cfg.train.extra_args -join ", ")

RunDir: $runDir
ExpDir: $expDir
BestPt: $(Join-Path $ckptDir "best.pt")
"@ | Set-Content (Join-Path $notesDir "model_card.txt") -Encoding utf8

    if ($cfg.export.write_index_csv) {
        "$ts,$modelFamily,$expDir,$runDir,$ckptDir\best.pt" | Add-Content $indexFile -Encoding utf8
    }

    Write-Host "EXPORT OK -> $expDir" -ForegroundColor Green
}

Write-Host "SWEEP DONE." -ForegroundColor Green
