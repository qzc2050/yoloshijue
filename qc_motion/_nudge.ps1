param(
    [Parameter(Mandatory = $true)][ValidateSet('x', 'y')][string]$Axis,
    [Parameter(Mandatory = $true)][double]$Delta
)
$ErrorActionPreference = 'Stop'
if ([math]::Abs($Delta) -lt 0.001 -or [math]::Abs($Delta) -gt 0.2) {
    throw "step must be within 0.2 mm"
}

function Read-Text([System.IO.Ports.SerialPort]$port, [int]$seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    $text = ''
    while ((Get-Date) -lt $deadline) {
        $chunk = $port.ReadExisting()
        if ($chunk) { $text += $chunk }
        if ($text -match 'pos\s+x=') {
            Start-Sleep -Milliseconds 200
            $text += $port.ReadExisting()
            break
        }
        Start-Sleep -Milliseconds 40
    }
    return $text
}

$p = New-Object System.IO.Ports.SerialPort 'COM7', 115200
$p.Parity = 'None'
$p.DataBits = 8
$p.StopBits = 'One'
$p.DtrEnable = $false
$p.RtsEnable = $false
$p.ReadTimeout = 500
$p.WriteTimeout = 2000
try {
    $p.Open()
    Start-Sleep -Milliseconds 400
    $boot = $p.ReadExisting()
    if ($boot -match 'qc_motion boot') { throw "mcu reset, no move" }
    $p.DiscardInBuffer()
    $p.Write("pos`n")
    $where = Read-Text $p 5
    if ($where -notmatch 'x=(-?\d+\.\d+)\s+y=(-?\d+\.\d+)') { throw "no position: $where" }
    $x = [double]$Matches[1]
    $y = [double]$Matches[2]
    if ($Axis -eq 'x') { $x = [math]::Round($x + $Delta, 3) } else { $y = [math]::Round($y + $Delta, 3) }
    if ($x -lt -2 -or $y -lt -2 -or $x -gt 300 -or $y -gt 300) {
        throw "target $x $y is outside -2..300"
    }
    $p.DiscardInBuffer()
    $p.Write(("m {0:0.000} {1:0.000}`n" -f $x, $y))
    $reply = Read-Text $p 8
    if ($reply -match 'stopped|err') { throw "move failed: $reply" }
    $hits = [regex]::Matches($reply, 'x=(-?\d+\.\d+)\s+y=(-?\d+\.\d+)')
    if ($hits.Count -lt 1) { throw "no arrival: $reply" }
    $last = $hits[$hits.Count - 1]
    Write-Output ("x={0} y={1}" -f $last.Groups[1].Value, $last.Groups[2].Value)
}
finally {
    if ($p.IsOpen) { $p.Close() }
}
