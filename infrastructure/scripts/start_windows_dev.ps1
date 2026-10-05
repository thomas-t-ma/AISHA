param(
    [ValidateSet("mock", "mac-m2max-96gb", "nvidia-2070", "nvidia-5080", "nvidia-high")]
    [string]$Profile = "mock",
    [switch]$Vision,
    [switch]$Voice,
    [switch]$Listen,
    [switch]$Body,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Core = Join-Path $RepoRoot "services\aisha-core"
$Studio = Join-Path $RepoRoot "apps\aisha-studio"
$BodyApp = Join-Path $RepoRoot "apps\aisha-body"
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

if ($Vision) {
    Push-Location $Core
    try {
        & $Python -c "import cv2, mediapipe" 2>$null
        $visionPackagesReady = $LASTEXITCODE -eq 0

        if (-not $visionPackagesReady) {
            if ($SkipInstall) {
                throw "Vision packages are missing. Run without -SkipInstall once."
            }
            Write-Host "Installing optional AISHA local vision dependencies..."
            & $Python -m pip install -e ".[vision-local]"
            if ($LASTEXITCODE -ne 0) {
                throw "Could not install AISHA local vision dependencies."
            }
        }

        if ($SkipInstall) {
            & $Python scripts\setup_local_vision.py
        }
        else {
            & $Python scripts\setup_local_vision.py --download-model
        }
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA local vision setup is incomplete."
        }
    }
    finally {
        Pop-Location
    }
}

if ($Voice) {
    Push-Location $Core
    try {
        & $Python -c "import kokoro_onnx, soundfile" 2>$null
        $voicePackagesReady = $LASTEXITCODE -eq 0

        if (-not $voicePackagesReady) {
            if ($SkipInstall) {
                throw "Voice packages are missing. Run without -SkipInstall once."
            }
            Write-Host "Installing optional AISHA local voice dependencies..."
            & $Python -m pip install -e ".[voice-local]"
            if ($LASTEXITCODE -ne 0) {
                throw "Could not install AISHA local voice dependencies."
            }
        }

        if ($SkipInstall) {
            & $Python scripts\setup_local_voice.py
        }
        else {
            & $Python scripts\setup_local_voice.py --download-models
        }
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA local voice setup is incomplete."
        }
    }
    finally {
        Pop-Location
    }
}

if ($Listen) {
    Push-Location $Core
    try {
        & $Python -c "import faster_whisper" 2>$null
        $listenPackagesReady = $LASTEXITCODE -eq 0

        if (-not $listenPackagesReady) {
            if ($SkipInstall) {
                throw "Listening packages are missing. Run without -SkipInstall once."
            }
            Write-Host "Installing optional AISHA local listening dependencies..."
            & $Python -m pip install -e ".[stt-local]"
            if ($LASTEXITCODE -ne 0) {
                throw "Could not install AISHA local listening dependencies."
            }
        }

        if ($SkipInstall) {
            & $Python scripts\setup_local_stt.py --model small.en
        }
        else {
            & $Python scripts\setup_local_stt.py --model small.en --download-model
        }
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA local listening setup is incomplete."
        }
    }
    finally {
        Pop-Location
    }
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

if ($Body -and -not (Test-Path (Join-Path $BodyApp "node_modules"))) {
    if ($SkipInstall) {
        throw "AISHA Body dependencies are missing. Run without -SkipInstall once."
    }
    Write-Host "Installing AISHA Body dependencies..."
    Push-Location $BodyApp
    try {
        & npm.cmd install
        if ($LASTEXITCODE -ne 0) {
            throw "AISHA Body npm install failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$VisionEnvironment = if ($Vision) {
    "`$env:AISHA_VISION_PROVIDER = 'local-mediapipe'"
}
else {
    ""
}
$VoiceEnvironment = if ($Voice) {
    "`$env:AISHA_TTS_PROVIDER = 'local-kokoro'"
}
else {
    ""
}
$ListenEnvironment = if ($Listen) {
    "`$env:AISHA_STT_PROVIDER = 'local-faster-whisper'"
}
else {
    ""
}

$BackendCommand = @"
`$env:AISHA_PROFILE = '$Profile'
$VisionEnvironment
$VoiceEnvironment
$ListenEnvironment
Set-Location '$Core'
& '$Python' -m uvicorn aisha.main:app --reload --host 127.0.0.1 --port 8000
"@

$FrontendCommand = @"
Set-Location '$Studio'
& '$Fnm' env --use-on-cd --shell powershell | Out-String | Invoke-Expression
& '$Fnm' use '$NodeVersion' | Out-Host
& npm.cmd run dev
"@

$BodyCommand = @"
Set-Location '$BodyApp'
& '$Fnm' env --use-on-cd --shell powershell | Out-String | Invoke-Expression
& '$Fnm' use '$NodeVersion' | Out-Host
& npm.cmd run dev
"@

Write-Host ""
$VisionLabel = if ($Vision) { "local vision enabled" } else { "vision disabled" }
$VoiceLabel = if ($Voice) { "local voice enabled" } else { "voice disabled" }
$ListenLabel = if ($Listen) { "local listening enabled" } else { "listening disabled" }
$BodyLabel = if ($Body) { "body enabled" } else { "body disabled" }
Write-Host "Starting AISHA Core ($Profile, $VisionLabel, $VoiceLabel, $ListenLabel, $BodyLabel) and AISHA Studio..."
Write-Host "Studio: http://127.0.0.1:5173"
if ($Body) {
    Write-Host "Body:   http://127.0.0.1:5174"
}
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

if ($Body) {
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-Command", $BodyCommand
    )
}
