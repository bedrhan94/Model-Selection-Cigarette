param(
  [string]$mode = "last",  # last | all | compare
  [int]$compare_n = 5
)

$proj = (Get-Location).Path
$plot = Join-Path $proj "tools\plot_metrics_dashboard.py"
if (!(Test-Path $plot)) { Write-Host "plot_metrics_dashboard.py not found: $plot"; exit 1 }

function RunDash($dir) {
  if (Test-Path $dir) {
    py -3.10 $plot --run_dir $dir
  }
}

# collect metrics_* folders
$metricsRoot = Join-Path $proj "outputs\04_metrics_active"
$runs = @()
if (Test-Path $metricsRoot) {
  $runs = Get-ChildItem $metricsRoot -Directory -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "metrics_*" } |
    Sort-Object LastWriteTime -Descending
}

if ($mode -eq "last") {
  $last = $runs | Select-Object -First 1
  if ($null -eq $last) { Write-Host "no metrics_* folders under outputs\04_metrics_active"; exit 0 }
  Write-Host "LAST RUN => $($last.FullName)"
  RunDash $last.FullName
  exit 0
}

if ($mode -eq "all") {
  if ($runs.Count -eq 0) { Write-Host "no metrics_* folders under outputs\04_metrics_active"; exit 0 }
  foreach ($r in $runs) { RunDash $r.FullName }
  Write-Host "DONE: all dashboards"
  exit 0
}

if ($mode -eq "compare") {
  $sel = $runs | Select-Object -First $compare_n
  if ($sel.Count -lt 2) { Write-Host "compare needs at least 2 runs"; exit 0 }

  # build args list
  $args = @("`"$($sel[0].FullName)`"")
  for ($i=1; $i -lt $sel.Count; $i++) { $args += "`"$($sel[$i].FullName)`"" }

  $cmd = "py -3.10 `"$plot`" --compare " + ($args -join " ")
  Write-Host $cmd
  iex $cmd

  Write-Host "OK -> outputs\07_master_charts\dashboard_compare.png"
  exit 0
}

Write-Host "mode: last|all|compare"
