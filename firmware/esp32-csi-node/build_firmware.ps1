# Remove MSYS environment variables that trigger ESP-IDF's MinGW rejection
Remove-Item env:MSYSTEM -ErrorAction SilentlyContinue
Remove-Item env:MSYSTEM_CARCH -ErrorAction SilentlyContinue
Remove-Item env:MSYSTEM_CHOST -ErrorAction SilentlyContinue
Remove-Item env:MSYSTEM_PREFIX -ErrorAction SilentlyContinue
Remove-Item env:MINGW_CHOST -ErrorAction SilentlyContinue
Remove-Item env:MINGW_PACKAGE_PREFIX -ErrorAction SilentlyContinue
Remove-Item env:MINGW_PREFIX -ErrorAction SilentlyContinue

$env:IDF_PATH = "C:\Users\ruv\esp\v5.4\esp-idf"
$env:IDF_TOOLS_PATH = "C:\Espressif\tools"
$env:IDF_PYTHON_ENV_PATH = "C:\Espressif\tools\python\v5.4\venv"
$env:PATH = "C:\Espressif\tools\xtensa-esp-elf\esp-14.2.0_20241119\xtensa-esp-elf\bin;C:\Espressif\tools\cmake\3.30.2\cmake-3.30.2-windows-x86_64\bin;C:\Espressif\tools\ninja\1.12.1;C:\Espressif\tools\ccache\4.10.2\ccache-4.10.2-windows-x86_64;C:\Espressif\tools\idf-exe\1.0.3;C:\Espressif\tools\python\v5.4\venv\Scripts;$env:PATH"

Set-Location "C:\Users\ruv\Projects\wifi-densepose\firmware\esp32-csi-node"

$python = "$env:IDF_PYTHON_ENV_PATH\Scripts\python.exe"
$idf = "$env:IDF_PATH\tools\idf.py"

Write-Host "=== Cleaning stale build cache ==="
& $python $idf fullclean

Write-Host "=== Building firmware (SSID=ruv.net, target=192.168.1.20:5005) ==="
& $python $idf build

if ($LASTEXITCODE -eq 0) {
    Write-Host "=== Build succeeded! Flashing to COM7 ==="
    & $python $idf -p COM7 flash
} else {
    Write-Host "=== Build failed with exit code $LASTEXITCODE ==="
}
