param(
    [string]$Endpoint = "http://172.31.102.189:8189",
    [string]$Config = "config\config.yaml",
    [int]$PollSeconds = 30,
    [string]$PipelineLog = "logs\h3_continuous_video.log",
    [string]$LauncherLog = "logs\h3_continuous_launcher.log"
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = "1"

if ($PollSeconds -lt 5) {
    throw "PollSeconds must be at least 5"
}

$launcherLogPath = Join-Path $root $LauncherLog
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $launcherLogPath) | Out-Null

function Write-LauncherLog([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $launcherLogPath -Encoding UTF8 -Value "$stamp $Message"
}

$healthUrl = $Endpoint.TrimEnd("/") + "/api/health"
Write-LauncherLog "launcher_started waiting_for=$healthUrl"
$lastState = ""
$checks = 0

while ($true) {
    $checks++
    try {
        $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 15
        $state = "status=$($health.status) comfyui=$($health.comfyui)"
        if ($state -ne $lastState -or $checks % 20 -eq 0) {
            Write-LauncherLog "health $state"
            $lastState = $state
        }
        if ($health.status -eq "ok" -and [bool]$health.comfyui) {
            break
        }
    }
    catch {
        $state = "error=$($_.Exception.Message)"
        if ($state -ne $lastState -or $checks % 20 -eq 0) {
            Write-LauncherLog "health $state"
            $lastState = $state
        }
    }
    Start-Sleep -Seconds $PollSeconds
}

$duplicate = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python" -and $_.CommandLine -match "scripts[\\/]run_full\.py"
} | Select-Object -First 1
if ($duplicate) {
    Write-LauncherLog "pipeline_already_running pid=$($duplicate.ProcessId); launcher_exiting"
    exit 0
}

$python = Join-Path $root ".venv\Scripts\python.exe"
Write-LauncherLog "h3_ready starting_pipeline"
& $python -u "scripts\run_full.py" `
    --config $Config `
    --from-stage video `
    --to-stage video `
    --log $PipelineLog
$exitCode = $LASTEXITCODE
Write-LauncherLog "pipeline_exited code=$exitCode"
exit $exitCode
