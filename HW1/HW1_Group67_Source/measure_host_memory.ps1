param([Parameter(Mandatory)][string]$Executable)

$ErrorActionPreference = 'Stop'
$app = (Resolve-Path -LiteralPath $Executable).Path
$log = Join-Path (Get-Location).Path 'qft_28q_memory_run.txt'
$process = Start-Process -FilePath $app -WorkingDirectory (Split-Path -Parent $app) -ArgumentList '28','5' -WindowStyle Hidden -PassThru -RedirectStandardOutput $log
$peakWorkingSet = 0L
$peakPrivate = 0L
while (!$process.HasExited) {
    $process.Refresh()
    if ($process.HasExited) { break }
    $peakWorkingSet = [Math]::Max($peakWorkingSet, $process.PeakWorkingSet64)
    $peakPrivate = [Math]::Max($peakPrivate, $process.PrivateMemorySize64)
    Start-Sleep -Milliseconds 100
}
$process.WaitForExit()
if ($process.ExitCode -ne 0) { throw "QFT exited with code $($process.ExitCode)" }
Get-Content -LiteralPath $log | Out-Host
[pscustomobject]@{
    PeakWorkingSetBytes = $peakWorkingSet
    PeakSampledPrivateBytes = $peakPrivate
}
