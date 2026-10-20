$ErrorActionPreference = 'Stop'
$outDir = 'E:\xiangmu\shibie\qc_motion\scan'
$log = Join-Path $outDir 'scan_log.txt'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$hostDir = Get-ChildItem 'E:\xiangmu\shibie' -Directory | Where-Object { Test-Path (Join-Path $_.FullName '.venv\Scripts\python.exe') } | Select-Object -First 1
if (-not $hostDir) { throw 'python venv not found' }
$python = Join-Path $hostDir.FullName '.venv\Scripts\python.exe'
$grab = 'E:\xiangmu\shibie\qc_motion\_grab_at.py'
$same = 'E:\xiangmu\shibie\qc_motion\_same_frame.py'
$settleMs = 1000

function Log([string]$line) {
  Write-Output $line
  Add-Content -Path $log -Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $line)
}

function Read-Text([int]$seconds, [string]$until) {
  $deadline = (Get-Date).AddSeconds($seconds)
  $text = ''
  while ((Get-Date) -lt $deadline) {
    $chunk = $p.ReadExisting()
    if ($chunk) { $text += $chunk }
    if ($until -and $text -match $until) {
      Start-Sleep -Milliseconds 250
      $text += $p.ReadExisting()
      break
    }
    Start-Sleep -Milliseconds 50
  }
  return $text
}

function Shoot([int]$n, [int]$x, [int]$y) {
  $script:lastFile = Join-Path $outDir ('scan_{0:d2}_x{1}_y{2}.jpg' -f $n, $x, $y)
  Log "settle ${settleMs}ms"
  Start-Sleep -Milliseconds $settleMs
  Log "photo $($script:lastFile | Split-Path -Leaf)"
  & $python $grab $script:lastFile
  if ($LASTEXITCODE -ne 0) { throw "photo failed $x $y" }
}

function Move-To([int]$x, [int]$y) {
  if ($x -gt 300 -or $y -gt 300 -or $x -lt 0 -or $y -lt 0) {
    throw "point $x $y is outside 0..300"
  }
  $xy = '{0} {1}' -f $x, $y
  $want = 'x={0}.000 y={1}.000' -f $x, $y
  Log "move $xy"
  $p.DiscardInBuffer()
  $p.Write("m $xy`n")
  $text = Read-Text 20 'pos\s+x='
  if ($text -match 'stopped') { throw "stopped $xy :: $text" }
  if ($text -match 'err') { throw "move err $xy :: $text" }
  if ($text -notmatch [regex]::Escape($want)) { throw "pos mismatch want $want got $text" }
  Log "arrived $want"
}

$points = @()
for ($yi = 0; $yi -lt 6; $yi++) {
  $y = $yi * 60
  for ($xi = 0; $xi -lt 6; $xi++) {
    $x = $xi * 60
    if ($yi % 2 -eq 1) { $x = (5 - $xi) * 60 }
    $points += ,@($x, $y)
  }
}
if ($points.Count -ne 36) { throw "expected 36 points, got $($points.Count)" }

$p = New-Object System.IO.Ports.SerialPort 'COM7', 115200
$p.Parity = 'None'
$p.DataBits = 8
$p.StopBits = 'One'
$p.DtrEnable = $false
$p.RtsEnable = $false
$p.ReadTimeout = 500
$p.WriteTimeout = 2000
$prev = $null
try {
  $p.Open()
  Start-Sleep -Milliseconds 400
  $boot = $p.ReadExisting()
  if ($boot -match 'qc_motion boot') { throw "mcu reset, no move sent: $boot" }
  $p.DiscardInBuffer()
  $p.Write("pos`n")
  $where = Read-Text 5 'pos\s+x='
  Log ("where " + ($where -replace '\s+', ' '))
  if ($where -notmatch 'x=0\.000 y=0\.000') {
    throw "not at 0 0, no move sent: $where"
  }
  $n = 0
  foreach ($pt in $points) {
    $n += 1
    if (-not ($pt[0] -eq 0 -and $pt[1] -eq 0)) {
      Move-To $pt[0] $pt[1]
    }
    Shoot $n $pt[0] $pt[1]
    if ($prev) {
      & $python $same $prev $script:lastFile
      if ($LASTEXITCODE -eq 2) { throw "frame did not move at $($pt[0]) $($pt[1]), scan stopped" }
      if ($LASTEXITCODE -ne 0) { throw "compare failed at $($pt[0]) $($pt[1])" }
    }
    $prev = $script:lastFile
  }
  Log 'full scan done 36'
}
finally {
  if ($p.IsOpen) { $p.Close() }
}
