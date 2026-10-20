$ErrorActionPreference = 'Stop'
$outDir = 'E:\xiangmu\shibie\qc_motion\scan40'
$log = Join-Path $outDir 'scan_log.txt'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$hostDir = Get-ChildItem 'E:\xiangmu\shibie' -Directory | Where-Object { Test-Path (Join-Path $_.FullName '.venv\Scripts\python.exe') } | Select-Object -First 1
if (-not $hostDir) { throw 'python venv not found' }
$python = Join-Path $hostDir.FullName '.venv\Scripts\python.exe'
$grab = 'E:\xiangmu\shibie\qc_motion\_grab_keep.py'
$same = 'E:\xiangmu\shibie\qc_motion\_same_frame.py'
$settleMs = 1000

function Log([string]$line) {
  Write-Output $line
  Add-Content -Path $log -Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $line) -Encoding utf8
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
  $script:lastFile = Join-Path $outDir ('scan40_{0:d2}_x{1}_y{2}.jpg' -f $n, $x, $y)
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
  $text = Read-Text 45 'pos\s+x='
  if ($text -match 'stopped') { throw "stopped $xy :: $text" }
  if ($text -match 'err') { throw "move err $xy :: $text" }
  if ($text -notmatch [regex]::Escape($want)) { throw "pos mismatch want $want got $text" }
  Log "arrived $want"
}

$stops = @(0, 40, 80, 120, 160, 200, 240, 280, 300)
$points = @()
for ($yi = 0; $yi -lt $stops.Count; $yi++) {
  $y = $stops[$yi]
  for ($xi = 0; $xi -lt $stops.Count; $xi++) {
    $x = $stops[$xi]
    if ($yi % 2 -eq 1) { $x = $stops[$stops.Count - 1 - $xi] }
    $points += ,@($x, $y)
  }
}
if ($points.Count -ne 81) { throw "expected 81 points, got $($points.Count)" }
foreach ($pt in $points) {
  if ($pt[0] -gt 300 -or $pt[1] -gt 300) { throw "point $($pt[0]) $($pt[1]) past 300" }
}

$p = New-Object System.IO.Ports.SerialPort 'COM7', 115200
$p.Parity = 'None'
$p.DataBits = 8
$p.StopBits = 'One'
$p.DtrEnable = $false
$p.RtsEnable = $false
$p.ReadTimeout = 500
$p.WriteTimeout = 2000
$prev = $null
$prevX = 0
$prevY = 0
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
  $prev = Join-Path $outDir 'scan40_03_x80_y0.jpg'
  if (-not (Test-Path $prev)) { $prev = $null }
  foreach ($pt in $points) {
    $n += 1
    if ($n -lt 4) { continue }
    Log ("point {0}/81" -f $n)
    Move-To $pt[0] $pt[1]
    Shoot $n $pt[0] $pt[1]
    if ($prev) {
      & $python $same $prev $script:lastFile 0.95
      if ($LASTEXITCODE -eq 2) { throw "corr above 95% at $($pt[0]) $($pt[1]), scan stopped" }
      if ($LASTEXITCODE -ne 0) { throw "compare failed at $($pt[0]) $($pt[1])" }
    }
    $prev = $script:lastFile
  }
  Log 'scan40 done 81'
}
finally {
  if ($p.IsOpen) { $p.Close() }
}
