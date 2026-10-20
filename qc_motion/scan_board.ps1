$ErrorActionPreference = 'Stop'
$outDir = 'E:\xiangmu\shibie\qc_motion\scan'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$hostDir = Get-ChildItem 'E:\xiangmu\shibie' -Directory | Where-Object { Test-Path (Join-Path $_.FullName '.venv\Scripts\python.exe') } | Select-Object -First 1
if (-not $hostDir) { throw 'python venv not found' }
$python = Join-Path $hostDir.FullName '.venv\Scripts\python.exe'
$grab = Join-Path $hostDir.FullName 'qc_host\vision\_grab_scan.py'
$env:PYTHONPATH = ($hostDir.FullName + ';' + (Join-Path $hostDir.FullName 'qc_host\vision'))

# Stroke of both axes is 300 mm. Do not add a point past that.
$points = @(
  '0 0','60 0','120 0','180 0','240 0','300 0',
  '300 100','240 100','180 100','120 100','60 100','0 100',
  '0 200','60 200','120 200','180 200','240 200','300 200',
  '300 300','240 300','180 300','120 300','60 300','0 300'
)
foreach ($xy in $points) {
  foreach ($n in $xy.Split(' ')) {
    if ([int]$n -gt 300) { throw "point $xy is past the 300 mm stroke" }
  }
}

$p = New-Object System.IO.Ports.SerialPort 'COM7', 115200
$p.Parity = 'None'
$p.DataBits = 8
$p.StopBits = 'One'
$p.DtrEnable = $false
$p.RtsEnable = $false
$p.ReadTimeout = 40000
$p.WriteTimeout = 2000
$p.Open()
Start-Sleep -Milliseconds 300

function Move-To([string]$xy) {
  $parts = $xy.Split(' ')
  $want = 'x={0}.000 y={1}.000' -f [int]$parts[0], [int]$parts[1]
  $p.DiscardInBuffer()
  $p.Write("m $xy`n")
  $deadline = (Get-Date).AddSeconds(40)
  $text = ''
  while ((Get-Date) -lt $deadline) {
    $text += $p.ReadExisting()
    if ($text -match 'stopped') { throw "stopped $xy :: $text" }
    if ($text -match 'pos\s+x=') {
      Start-Sleep -Milliseconds 250
      $text += $p.ReadExisting()
      break
    }
    Start-Sleep -Milliseconds 60
  }
  if ($text -notmatch [regex]::Escape($want)) { throw "pos mismatch want $want got $text" }
  return $want
}

try {
  $n = 0
  foreach ($xy in $points) {
    $n += 1
    $stamp = '{0:d2}' -f $n
    $name = $xy.Replace(' ', '_')
    $line = "move {0} {1}" -f $stamp, $xy
    Write-Output $line
    Add-Content -Path (Join-Path $outDir 'scan_log.txt') -Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $line)
    $got = Move-To $xy
    $line = "arrived {0} {1}" -f $stamp, $got
    Write-Output $line
    Add-Content -Path (Join-Path $outDir 'scan_log.txt') -Value ((Get-Date -Format 'HH:mm:ss') + ' ' + $line)
    Start-Sleep -Milliseconds 400
    $file = Join-Path $outDir ("scan_{0}_x{1}.jpg" -f $stamp, $name)
    & $python $grab $file
    if ($LASTEXITCODE -ne 0) { throw "photo failed at $xy" }
  }
  Write-Output 'scan done'
}
finally {
  if ($p.IsOpen) { $p.Close() }
}
