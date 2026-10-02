$ErrorActionPreference = "Stop"

param(
    [ValidateSet("mock", "mac-m2max-96gb", "nvidia-2070", "nvidia-5080", "nvidia-high")]
    [string]$Profile = "mock",
    [switch]$SkipInstall
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Core = Join-Path $RepoRoot "services\aisha-core"
$Studio = Join-Path $RepoRoot "apps\aisha-studio"
$Python = Join-Path $Core ".venv\Scripts\python.exe"
$NodeVersionFile = Join-Path $Studio ".node-version"

function Resolve-Fnm {
    $command = Get-Command fnm -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $wingetRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (Test-Path $wingetRoot) {
        $candidate = Get-ChildItem $wingetRoot -Filter fnm.exe -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1 -ExpandProperty FullName
        if ($candidate) {
            return $candidate
        }
    }

    throw "fnm.exe was not found. Install it once with: winget install Schniz.fnm"
}

if (-not (Test-Path $Python)) {
    throw "AISHA Python environment is missing at $Python. Run infrastructure\scripts\bootstrap_windows.ps1 once."
}

$Fnm = Resolve-Fnm
$NodeVersion = (Get-Content $NodeVersionFile -Raw).Trim()

# Initialize fnm in this process without relying on the user's PowerShell profile.
& $Fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression
& $Fnm use $NodeVersion | Out-Host

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node did not become available after fnm initialization."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm did not become available after fnm initialization."
}

if (-not $SkipInstall -and -not (Test-Path (Join-Path $Studio "node_modules"))) {
    Write-Host "Installing Studio dependencies..."
    Push-Location $Studio
    try {
        & npm.cmd install
        if ($LASTEXITCODE -ne 0) {
            throw "npm install failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$BackendCommand = @"
`$env:AISHA_PROFILE = '$Profile'
Set-Location '$Core'
& '$Python' -m uvicorn aisha.main:app --reload --host 127.0.0.1 --port 8000
"@

$FrontendCommand = @"
Set-Location '$Studio'
& '$Fnm' env --use-on-cd --shell powershell | Out-String | Invoke-Expression
& '$Fnm' use '$NodeVersion' | Out-Host
& npm.cmd run dev
"@

Write-Host ""
Write-Host "Starting AISHA Core ($Profile) and AISHA Studio..."
Write-Host "Studio: http://127.0.0.1:5173"
Write-Host ""

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $BackendCommand
)

Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command", $FrontendCommand
)
