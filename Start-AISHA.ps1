param(
    [string]$Profile = "mock",
    [switch]$Vision,
    [switch]$Voice,
    [switch]$Listen,
    [switch]$Body,
    [switch]$SkipInstall
)

$launcher = Join-Path $PSScriptRoot "infrastructure\scripts\start_windows_dev.ps1"
& $launcher -Profile $Profile -Vision:$Vision -Voice:$Voice -Listen:$Listen -Body:$Body -SkipInstall:$SkipInstall
