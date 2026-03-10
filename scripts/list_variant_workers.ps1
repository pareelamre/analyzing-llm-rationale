$names = @(
  'run_full_then_nullfix_v3.py',
  'run_full_then_nullfix_v4.py',
  'run_full_then_nullfix_v5.py',
  'run_full_then_nullfix_v6.py'
)

foreach ($n in $names) {
  $procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match [regex]::Escape($n) }
  $pids = $procs | Select-Object -ExpandProperty ProcessId
  Write-Output ("$n`t" + ($pids -join ','))
}
