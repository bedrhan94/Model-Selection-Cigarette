$ROOT = $PSScriptRoot
Set-Location -LiteralPath $ROOT

# ============================================================
# ELECTRICITY-SAFE / RESUME-SAFE 3-SEED SAHI ABLATION
# ============================================================

$EPOCHS = 30
$BATCH  = 8
$IMGSZ  = 640

$SEEDS = @(42, 123, 2025)

$PROJECT = Join-Path $ROOT "outputs\REVISION_3SEED_SAHI_ABLATION_FIXEDDATA"

$DATASETS = @(
    @{
        condition = "full_frame"
        data = "outputs\REVISION_3SEED_CONFIGS\revision_full_frame_FIXED.yaml"
    },
    @{
        condition = "sahi_sliced"
        data = "outputs\REVISION_3SEED_CONFIGS\revision_sahi_sliced_FIXED.yaml"
    }
)

$MODELS = @(
    @{
        name = "YOLO11m"
        weight = "yolo11m.pt"
    },
    @{
        name = "YOLO11s"
        weight = "yolo11s.pt"
    }
)

New-Item -ItemType Directory -Force -Path $PROJECT | Out-Null

$MASTER_LOG = Join-Path $PROJECT "RUN_MASTER_LOG.txt"

function Log-Line {
    param([string]$Text)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$stamp] $Text"
    Write-Host $line
    Add-Content -Path $MASTER_LOG -Value $line -Encoding UTF8
}

function Get-LastEpochFromResults {
    param([string]$ResultsCsv)

    if (-not (Test-Path -LiteralPath $ResultsCsv)) {
        return -1
    }

    try {
        $rows = Import-Csv -LiteralPath $ResultsCsv
        if ($null -eq $rows -or $rows.Count -eq 0) {
            return -1
        }

        $last = $rows[-1]

        if ($last.PSObject.Properties.Name -contains "epoch") {
            return [int]$last.epoch
        }

        return $rows.Count - 1
    }
    catch {
        return -1
    }
}

function Get-RunStatus {
    param(
        [string]$RunDir,
        [int]$TargetEpochs
    )

    $resultsCsv = Join-Path $RunDir "results.csv"
    $lastPt = Join-Path $RunDir "weights\last.pt"
    $bestPt = Join-Path $RunDir "weights\best.pt"

    $lastEpoch = Get-LastEpochFromResults -ResultsCsv $resultsCsv

    # Ultralytics epochs usually start at 0.
    # If 30 epochs finished, the last epoch is usually 29.
    if ((Test-Path -LiteralPath $resultsCsv) -and (Test-Path -LiteralPath $bestPt) -and ($lastEpoch -ge ($TargetEpochs - 1))) {
        return "complete"
    }

    if (Test-Path -LiteralPath $lastPt) {
        return "resume"
    }

    return "new"
}

Log-Line "============================================================"
Log-Line "REVISION 3-SEED SAHI ABLATION - RESUME SAFE START"
Log-Line "Root    : $ROOT"
Log-Line "Project : $PROJECT"
Log-Line "Epochs  : $EPOCHS"
Log-Line "Batch   : $BATCH"
Log-Line "Imgsz   : $IMGSZ"
Log-Line "Seeds   : $($SEEDS -join ', ')"
Log-Line "============================================================"

# Check data YAML files
foreach ($dataset in $DATASETS) {
    $dataPath = Join-Path $ROOT $dataset.data

    if (-not (Test-Path -LiteralPath $dataPath)) {
        Log-Line "ERROR: DATA YAML NOT FOUND: $dataPath"
        exit 1
    }
}

# Try to disable Windows sleep mode.
# May fail without admin rights; not important.
try {
    powercfg /change standby-timeout-ac 0 | Out-Null
    powercfg /change monitor-timeout-ac 0 | Out-Null
    Log-Line "Tried to disable sleep/display timeout on AC power."
}
catch {
    Log-Line "powercfg could not be applied; admin rights may be required. Continuing."
}

foreach ($model in $MODELS) {
    foreach ($dataset in $DATASETS) {
        foreach ($seed in $SEEDS) {

            $RUN_NAME = "$($model.name)_$($dataset.condition)_e${EPOCHS}_b${BATCH}_img${IMGSZ}_seed${seed}"
            $RUN_DIR = Join-Path $PROJECT $RUN_NAME
            $LAST_PT = Join-Path $RUN_DIR "weights\last.pt"
            $RUN_LOG = Join-Path $PROJECT "$RUN_NAME.log"

            $status = Get-RunStatus -RunDir $RUN_DIR -TargetEpochs $EPOCHS

            Log-Line ""
            Log-Line "------------------------------------------------------------"
            Log-Line "RUN       : $RUN_NAME"
            Log-Line "MODEL     : $($model.weight)"
            Log-Line "DATA      : $($dataset.data)"
            Log-Line "SEED      : $seed"
            Log-Line "STATUS    : $status"
            Log-Line "RUN_DIR   : $RUN_DIR"
            Log-Line "------------------------------------------------------------"

            if ($status -eq "complete") {
                Log-Line "SKIPPED: this run looks complete."
                continue
            }

            if ($status -eq "resume") {
                Log-Line "RESUME: last.pt found, training will continue where it stopped."
                Log-Line "LAST_PT: $LAST_PT"

                $argsList = @(
                    "detect",
                    "train",
                    "model=$LAST_PT",
                    "resume=True",
                    "device=0",
                    "workers=0"
                )
            }
            else {
                Log-Line "NEW: training will start from scratch."

                $argsList = @(
                    "detect",
                    "train",
                    "model=$($model.weight)",
                    "data=$($dataset.data)",
                    "epochs=$EPOCHS",
                    "batch=$BATCH",
                    "imgsz=$IMGSZ",
                    "seed=$seed",
                    "deterministic=True",
                    "workers=0",
                    "device=0",
                    "save=True",
                    "save_period=1",
                    "project=$PROJECT",
                    "name=$RUN_NAME",
                    "exist_ok=True"
                )
            }

            Log-Line "Command: yolo $($argsList -join ' ')"

            try {
                & yolo @argsList 2>&1 | Tee-Object -FilePath $RUN_LOG -Append

                $afterStatus = Get-RunStatus -RunDir $RUN_DIR -TargetEpochs $EPOCHS
                Log-Line "STATUS AFTER RUN: $afterStatus"

                if ($afterStatus -eq "complete") {
                    Log-Line "COMPLETED: $RUN_NAME"
                }
                else {
                    Log-Line "WARNING: run did not finish or is incomplete. After a power cut/crash, rerunning this script resumes it."
                }
            }
            catch {
                Log-Line "ERROR: an error occurred during $RUN_NAME."
                Log-Line "$_"
                Log-Line "Rerunning this script resumes from last.pt if it exists."
            }
        }
    }
}

Log-Line ""
Log-Line "============================================================"
Log-Line "SCRIPT FINISHED."
Log-Line "Completed runs are skipped on rerun."
Log-Line "Unfinished runs resume from weights\last.pt."
Log-Line "Output folder: $PROJECT"
Log-Line "============================================================"

