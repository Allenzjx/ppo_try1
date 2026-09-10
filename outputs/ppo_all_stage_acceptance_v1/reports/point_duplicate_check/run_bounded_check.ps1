$ErrorActionPreference = 'Stop'
$taskFreeBytes = [int64](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1024
if ($taskFreeBytes -lt 2GB) { Write-Output "SKIPPED_FREE_MEMORY_BYTES=$taskFreeBytes"; exit 3 }
$taskStart = [Diagnostics.ProcessStartInfo]::new()
$taskStart.FileName = 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe'
$taskStart.Arguments = '-B "' + (Join-Path $PSScriptRoot 'check_standalone_usd.py') + '"'
$taskStart.UseShellExecute = $false
$taskStart.CreateNoWindow = $true
$taskStart.RedirectStandardOutput = $true
$taskStart.RedirectStandardError = $true
foreach ($taskName in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PXR_WORK_THREAD_LIMIT')) { $taskStart.EnvironmentVariables[$taskName] = '1' }
$taskStart.EnvironmentVariables['CUDA_VISIBLE_DEVICES'] = ''
$taskProcess = [Diagnostics.Process]::new()
$taskProcess.StartInfo = $taskStart
$taskWatch = [Diagnostics.Stopwatch]::StartNew()
[void]$taskProcess.Start()
Write-Output "OWNED_OFFLINE_PID=$($taskProcess.Id)"
$taskStdout = $taskProcess.StandardOutput.ReadToEndAsync()
$taskStderr = $taskProcess.StandardError.ReadToEndAsync()
$taskPeakBytes = 0L
$taskStopReason = $null
while (-not $taskProcess.WaitForExit(100)) {
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) { break }
    $taskPeakBytes = [Math]::Max($taskPeakBytes, $taskProcess.WorkingSet64)
    if ($taskPeakBytes -gt 768MB) { $taskStopReason = 'WORKING_SET_OVER_768_MiB' }
    if ($taskWatch.Elapsed.TotalSeconds -ge 120) { $taskStopReason = 'DEADLINE_120_SECONDS' }
    if ($null -ne $taskStopReason) {
        # This exact Process object is the child started above; no name-based kill.
        $taskProcess.Kill()
        $taskProcess.WaitForExit()
        break
    }
}
$taskWatch.Stop()
Write-Output ($taskStdout.GetAwaiter().GetResult())
Write-Output ($taskStderr.GetAwaiter().GetResult())
[pscustomobject]@{owned_pid=$taskProcess.Id;exit_code=$taskProcess.ExitCode;stop_reason=$taskStopReason;elapsed_s=$taskWatch.Elapsed.TotalSeconds;peak_working_set_bytes=$taskPeakBytes;free_bytes_before_start=$taskFreeBytes} | ConvertTo-Json -Compress
$taskExit = $taskProcess.ExitCode
$taskProcess.Dispose()
exit $taskExit
