param(
    [string]$Proxy = 'http://127.0.0.1:10808',
    [string]$Python = 'E:\anaconda3\envs\MC_Gen\python.exe'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$target = Join-Path $root 'data\raw\opengameart_oga_by_3'
$log = Join-Path $target 'download.log'
$status = Join-Path $target 'download-status.txt'
New-Item -ItemType Directory -Force -Path $target | Out-Null

$files = @(
    @{ Name = '2D_Art_00.zip'; ExpectedBytes = 1850261199; Url = 'https://huggingface.co/datasets/nyuuzyou/OpenGameArt-OGA-BY-3.0/resolve/main/2D_Art_00.zip?download=true' },
    @{ Name = '2D_Art_01.zip'; ExpectedBytes = 657007636; Url = 'https://huggingface.co/datasets/nyuuzyou/OpenGameArt-OGA-BY-3.0/resolve/main/2D_Art_01.zip?download=true' }
)

"running $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $status
$env:HTTPS_PROXY = $Proxy
$env:HTTP_PROXY = $Proxy

function Invoke-Native {
    param([string]$Exe, [string[]]$Arguments, [string]$LogPath, [switch]$Quiet)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Quiet) {
            & $Exe @Arguments *> $null
        } else {
            & $Exe @Arguments *>> $LogPath
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) {
        throw "$Exe failed with exit code $code ($($Arguments -join ' '))"
    }
}

try {
    foreach ($item in $files) {
        $output = Join-Path $target $item.Name
        "downloading $($item.Name) $(Get-Date -Format o)" | Add-Content -Encoding UTF8 $log
        $isComplete = $item.ExpectedBytes -and (Test-Path $output) -and `
            ((Get-Item $output).Length -eq $item.ExpectedBytes)
        if (-not $isComplete) {
            $downloader = Join-Path $root 'scripts\resumable_http_download.py'
            Invoke-Native -Exe $Python -Arguments @('-u', $downloader, $item.Url, $output, '--proxy', $Proxy, '--retries', '100') -LogPath $log
        }
        Invoke-Native -Exe 'tar.exe' -Arguments @('-tf', $output) -Quiet
        "validated $($item.Name) $(Get-Date -Format o)" | Add-Content -Encoding UTF8 $log
    }
    "processing $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $status
    $processor = Join-Path $root 'src\data\pixel_sheet_processor.py'
    $processed = Join-Path $root 'data\processed\opengameart_2d32'
    Invoke-Native -Exe $Python -Arguments @('-u', $processor, '--root', $target, '--out', $processed, '--target', '32') -LogPath $log
    "building-stage1 $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $status
    $builder = Join-Path $root 'src\data\pixel_training_builder.py'
    $stage1 = Join-Path $root 'data\build\stage1_32'
    $mcDataset = Join-Path $root 'data\build\mc_text2image32'
    $manifests = @(
        (Join-Path $root 'data\processed\kenney_components32\manifest.jsonl'),
        (Join-Path $root 'data\processed\itch_components32\manifest.jsonl'),
        (Join-Path $root 'data\processed\kaggle_pixel32\manifest.jsonl'),
        (Join-Path $root 'data\processed\alucard_sprites32\manifest.jsonl'),
        (Join-Path $root 'data\processed\opengameart_2d32\manifest.jsonl')
    )
    $builderArgs = @('-u', $builder) + $manifests + @('--dataset', $mcDataset, '--out', $stage1, '--size', '32')
    Invoke-Native -Exe $Python -Arguments $builderArgs -LogPath $log
    "complete $(Get-Date -Format o)" | Set-Content -Encoding UTF8 $status
} catch {
    "failed $(Get-Date -Format o): $($_.Exception.Message)" | Set-Content -Encoding UTF8 $status
    throw
}
