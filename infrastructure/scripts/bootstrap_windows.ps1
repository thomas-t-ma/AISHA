$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Core = Join-Path $RepoRoot "services\aisha-core"
Set-Location $Core

py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

if (-not (Test-Path ".env")) {
  Copy-Item (Join-Path $RepoRoot ".env.example") ".env"
}

Write-Host "AISHA Core environment is ready."
Write-Host "Set AISHA_PROFILE=nvidia-2070 in .env on the RTX 2070 machine."
