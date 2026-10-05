param(
    [string]$Profile = "mock",
    [switch]$Vision,
    [switch]$Voice,
    [switch]$Listen
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Core = Join-Path $RepoRoot "services\aisha-core"
$Studio = Join-Path $RepoRoot "apps\aisha-studio"
$VenvPython = Join-Path $Core ".venv\Scripts\python.exe"
$NodeVersionFile = Join-Path $Studio ".node-version"
$PackageJson = Join-Path $Studio "package.json"

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

    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "fnm is missing and winget is unavailable. Install fnm, then rerun this bootstrap."
    }

    Write-Host "Installing fnm..."
    & winget install Schniz.fnm --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install fnm."
    }

    $candidate = Get-ChildItem $wingetRoot -Filter fnm.exe -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $candidate) {
        throw "fnm was installed but fnm.exe could not be located."
    }
    return $candidate
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating Python virtual environment..."
    $created = $false

    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($version in @("3.13", "3.12")) {
            & py "-$version" -m venv (Join-Path $Core ".venv")
            if ($LASTEXITCODE -eq 0) {
                $created = $true
                break
            }
        }
    }

    if (-not $created -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -m venv (Join-Path $Core ".venv")
        if ($LASTEXITCODE -eq 0) {
            $created = $true
        }
    }

    if (-not $created) {
        throw "Could not create the Python environment. Install Python 3.13 or 3.12."
    }
}

Write-Host "Installing AISHA Core dependencies..."
Push-Location $Core
try {
    & $VenvPython -m pip install --upgrade pip
    $extras = @("dev")
    if ($Vision) { $extras += "vision-local" }
    if ($Voice) { $extras += "voice-local" }
    if ($Listen) { $extras += "stt-local" }
    $coreExtra = ".[" + ($extras -join ",") + "]"
    & $VenvPython -m pip install -e $coreExtra
    if ($LASTEXITCODE -ne 0) {
        throw "AISHA Core dependency installation failed."
    }

    if ($Vision) {
        & $VenvPython scripts\setup_local_vision.py --download-model
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA local vision setup failed."
        }
    }

    if ($Voice) {
        & $VenvPython scripts\setup_local_voice.py --download-models
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA local voice setup failed."
        }
    }

    if ($Listen) {
        & $VenvPython scripts\setup_local_stt.py --model small.en --download-model
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA local listening setup failed."
        }
    }
}
finally {
    Pop-Location
}

if (-not (Test-Path (Join-Path $Core ".env"))) {
    Copy-Item (Join-Path $RepoRoot ".env.example") (Join-Path $Core ".env")
}

$Fnm = Resolve-Fnm
& $Fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression

$NodeVersion = (Get-Content $NodeVersionFile -Raw).Trim()
& $Fnm install $NodeVersion | Out-Host
& $Fnm use $NodeVersion | Out-Host

$package = Get-Content $PackageJson -Raw | ConvertFrom-Json
$npmVersion = [string]$package.packageManager
if ($npmVersion.StartsWith("npm@")) {
    $npmVersion = $npmVersion.Substring(4)
    $currentNpm = (& npm.cmd --version).Trim()
    if ($currentNpm -ne $npmVersion) {
        Write-Host "Installing npm $npmVersion..."
        & npm.cmd install --global "npm@$npmVersion"
    }
}

Write-Host "Installing AISHA Studio dependencies..."
Push-Location $Studio
try {
    & npm.cmd install
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "AISHA Windows development environment is ready."
Write-Host "Profile: $Profile"
Write-Host "Vision: $Vision"
Write-Host "Voice: $Voice"
Write-Host "Listen: $Listen"
Write-Host "Start everything from the repository root with:"
$visionFlag = if ($Vision) { " -Vision" } else { "" }
$voiceFlag = if ($Voice) { " -Voice" } else { "" }
$listenFlag = if ($Listen) { " -Listen" } else { "" }
Write-Host "  .\Start-AISHA.ps1 -Profile $Profile$visionFlag$voiceFlag$listenFlag"
