$p = New-Object System.IO.Ports.SerialPort('COM7', 115200)
$p.ReadTimeout = 5000
$p.Open()
Start-Sleep -Milliseconds 200

for ($i = 0; $i -lt 60; $i++) {
    try {
        $line = $p.ReadLine()
        Write-Host $line
    } catch {
        break
    }
}
$p.Close()
