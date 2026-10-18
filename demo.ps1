# One-click demo: opens the animation and result figures for presentation.
# Usage: right-click -> "Run with PowerShell", or run ".\demo.ps1" in a terminal.
# Add -Live to also re-run the PPO comparison live (slow, ~60s).

param(
    [switch]$Live
)

$root = $PSScriptRoot
$results = Join-Path $root "results"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Intelligent Elevator Scheduling - Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$gif = Join-Path $results "elevator_animated_demo.gif"
$charts = @(
    "ppo_training_reward.png",
    "average_waiting_time.png",
    "traffic_performance.png",
    "travel_distance.png"
)
$ablationCharts = @(
    "ablation_reward.png",
    "ablation_waiting_time.png",
    "ablation_unserved.png"
)

Write-Host ""
Write-Host "[1] Opening animated comparison GIF (3 panels: FCFS vs Nearest vs PPO)..." -ForegroundColor Green
if (Test-Path $gif) { Start-Process $gif } else { Write-Host "MISSING: $gif" -ForegroundColor Red }

Write-Host "[2] Opening result charts..." -ForegroundColor Green
foreach ($chart in $charts) {
    $full = Join-Path $results $chart
    if (Test-Path $full) { Start-Process $full } else { Write-Host "MISSING: $full" -ForegroundColor Red }
}

Write-Host "[3] Opening ablation study charts..." -ForegroundColor Green
foreach ($chart in $ablationCharts) {
    $full = Join-Path $results $chart
    if (Test-Path $full) { Start-Process $full } else { Write-Host "MISSING: $full" -ForegroundColor Red }
}

if ($Live) {
    Write-Host ""
    Write-Host "[4] Re-running the PPO comparison live on unseen seeds..." -ForegroundColor Green
    python -m elevator_rl.evaluation.compare_algorithms --seeds 106,107,108
}

Write-Host ""
Write-Host "Demo complete. Good luck!" -ForegroundColor Cyan