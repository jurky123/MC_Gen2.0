param(
    [string]$Dir = "data\raw\matsynth",
    [string]$Meta = "data\raw\matsynth\matsynth_metadata.jsonl",
    [int]$Interval = 5
)

$prevSize = 0
$prevCount = 0
$prevTime = Get-Date
$start = Get-Date

while ($true) {
    $count = (Get-Content $Meta -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $size = (Get-ChildItem $Dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    $now = Get-Date
    $dt = ($now - $prevTime).TotalSeconds
    $speed = if ($dt -gt 0) { ($size - $prevSize) / $dt / 1MB } else { 0 }
    $rate = if ($dt -gt 0) { ($count - $prevCount) / $dt * 60 } else { 0 }
    $elapsed = [int](($now - $start).TotalMinutes)

    Write-Host ("{0}  elapsed={1}min  materials={2}  data={3:N1}MB  speed={4:N2}MB/s  ~{5:N1}mat/min" -f `
        $now.ToString("HH:mm:ss"), $elapsed, $count, ($size / 1MB), $speed, $rate)

    $prevSize = $size
    $prevCount = $count
    $prevTime = $now
    Start-Sleep -Seconds $Interval
}
