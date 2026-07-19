$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Demo = Join-Path $ProjectRoot "offline_demo\demo.py"
$Checkpoint = Join-Path $ProjectRoot "models\frozen\seizure_tcn_p20_baseline_review_0d6774d3.pt"
$LogDirectory = Join-Path $ProjectRoot "reports\prototype_demo\viva_runs"

if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python was not found: $Python"
}

if (-not (Test-Path $Demo)) {
    throw "Offline demo script was not found: $Demo"
}

if (-not (Test-Path $Checkpoint)) {
    throw "Frozen P20 TCN checkpoint was not found: $Checkpoint"
}

New-Item `
    -ItemType Directory `
    -Path $LogDirectory `
    -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDirectory "valid_demo_$Timestamp.txt"

Clear-Host

Write-Host "============================================================"
Write-Host " VALID METADATA - OFFLINE EEG RESEARCH PROTOTYPE"
Write-Host "============================================================"
Write-Host "Research prototype only."
Write-Host "Not clinically validated."
Write-Host "Not a medical device."
Write-Host ""

Push-Location $ProjectRoot

try {
    & $Python -u $Demo 2>&1 |
        Tee-Object -FilePath $LogPath

    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($ExitCode -ne 0) {
    throw "Valid prototype failed with exit code $ExitCode."
}

$LogText = Get-Content $LogPath -Raw

$Checks = @{
    "SHACL validation passed" = (
        $LogText -match "SHACL result\s*:\s*PASS"
    )
    "Frozen TCN verified" = (
        $LogText -match "Frozen TCN\s*:\s*VERIFIED"
    )
    "EEG input verified" = (
        $LogText -match "EEG input\s*:\s*VERIFIED"
    )
    "Inference completed" = (
        $LogText -match "Inference\s*:\s*COMPLETED"
    )
    "Prototype passed" = (
        $LogText -match "PROTOTYPE EXECUTION:\s*PASS"
    )
}

Write-Host ""
Write-Host "=== VALID-DEMO VERIFICATION ==="

$FailedChecks = @()

foreach ($Check in $Checks.GetEnumerator()) {
    Write-Host (
        "{0,-30}: {1}" -f `
        $Check.Key,
        $(if ($Check.Value) { "PASS" } else { "FAIL" })
    )

    if (-not $Check.Value) {
        $FailedChecks += $Check.Key
    }
}

if ($FailedChecks.Count -gt 0) {
    throw "Valid-demo verification failed: $($FailedChecks -join ', ')"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " VALID OFFLINE DEMONSTRATION: PASS"
Write-Host "============================================================"
Write-Host "Evidence log: $LogPath"
