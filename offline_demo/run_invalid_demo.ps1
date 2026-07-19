$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Demo = Join-Path $ProjectRoot "offline_demo\demo.py"

$InvalidTtlRecord = Join-Path `
    $ProjectRoot `
    "reports\prototype_demo\selected_invalid_ttl.txt"

$LogDirectory = Join-Path `
    $ProjectRoot `
    "reports\prototype_demo\viva_runs"

if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python was not found: $Python"
}

if (-not (Test-Path $Demo)) {
    throw "Offline demo script was not found: $Demo"
}

if (-not (Test-Path $InvalidTtlRecord)) {
    throw "The selected-invalid-TTL record was not found: $InvalidTtlRecord"
}

$InvalidTtl = (
    Get-Content $InvalidTtlRecord -Raw
).Trim()

if (-not (Test-Path $InvalidTtl)) {
    throw "Invalid TTL fixture was not found: $InvalidTtl"
}

New-Item `
    -ItemType Directory `
    -Path $LogDirectory `
    -Force | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDirectory "invalid_demo_$Timestamp.txt"

Clear-Host

Write-Host "============================================================"
Write-Host " INVALID METADATA - SEMANTIC QUALITY-GATE DEMONSTRATION"
Write-Host "============================================================"
Write-Host "Expected behaviour:"
Write-Host "SHACL FAIL -> inference blocked before model loading"
Write-Host ""
Write-Host "Invalid TTL: $InvalidTtl"
Write-Host ""

Push-Location $ProjectRoot

try {
    & $Python -u $Demo `
        --ttl $InvalidTtl 2>&1 |
        Tee-Object -FilePath $LogPath

    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

# The demo intentionally returns 2 when SHACL blocks inference.
if ($ExitCode -ne 2) {
    throw "Expected exit code 2, but received $ExitCode."
}

$LogText = Get-Content $LogPath -Raw

$Checks = @{
    "SHACL validation failed" = (
        $LogText -match "SHACL result\s*:\s*FAIL"
    )
    "Inference blocked" = (
        $LogText -match "INFERENCE BLOCKED"
    )
    "Model was not loaded" = -not (
        $LogText -match "FROZEN TCN VERIFICATION"
    )
    "EEG was not loaded" = -not (
        $LogText -match "PROCESSED EEG VERIFICATION"
    )
    "Inference was not executed" = -not (
        $LogText -match "OFFLINE TCN INFERENCE"
    )
}

Write-Host ""
Write-Host "=== NEGATIVE-PATH VERIFICATION ==="

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
    throw "Negative-path verification failed: $($FailedChecks -join ', ')"
}

Write-Host ""
Write-Host "============================================================"
Write-Host " NEGATIVE SEMANTIC-GATE DEMONSTRATION: PASS"
Write-Host "============================================================"
Write-Host "Invalid metadata was rejected before model loading."
Write-Host "Evidence log: $LogPath"
