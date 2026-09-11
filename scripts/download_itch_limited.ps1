param(
    [double]$MaxGB = 30,
    [int]$Parallel = 3,
    [string]$Proxy = "http://127.0.0.1:10808"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$downloadDir = Join-Path $repoRoot "data\raw\itch_pixel"
$toolDir = Join-Path $repoRoot ".tools\itch-dl"
$python = "E:\anaconda3\envs\MC_Gen\python.exe"
$logPath = Join-Path $downloadDir "download_limited.log"
$statusPath = Join-Path $downloadDir "download_limited.status.log"
$limitBytes = [int64]($MaxGB * 1GB)

if (-not $env:ITCH_API_KEY) {
    throw "Set ITCH_API_KEY before running this script."
}

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
$env:PYTHONPATH = $toolDir
$env:HTTP_PROXY = $Proxy
$env:HTTPS_PROXY = $Proxy

$arguments = @(
    "-u", "-m", "itch_dl",
    "https://itch.io/game-assets/free/tag-pixel-art",
    "--api-key", $env:ITCH_API_KEY,
    "--download-to", $downloadDir,
    "--parallel", $Parallel
)

$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $logPath -RedirectStandardError "$logPath.err"

"started pid=$($process.Id), limit=$MaxGB GiB" | Add-Content $statusPath

while (-not $process.HasExited) {
    $bytes = (Get-ChildItem -LiteralPath $downloadDir -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { $bytes = 0 }
    if ($bytes -ge $limitBytes) {
        "limit reached: $([math]::Round($bytes / 1GB, 3)) GiB; stopping pid=$($process.Id)" | Add-Content $statusPath
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        break
    }
    Start-Sleep -Seconds 10
    $process.Refresh()
}

$finalBytes = (Get-ChildItem -LiteralPath $downloadDir -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
"finished: $([math]::Round($finalBytes / 1GB, 3)) GiB" | Add-Content $statusPath
