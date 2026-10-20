@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1

cd /d "%~dp0.."

if exist ".venv\Scripts\pythonw.exe" (
  set "PYW=.venv\Scripts\pythonw.exe"
) else if exist ".venv\Scripts\python.exe" (
  set "PYW=.venv\Scripts\python.exe"
) else (
  set "PYW=pythonw"
)

start "" /D "%cd%" "%PYW%" -m qc_host
if errorlevel 1 (
  echo 启动失败。请确认已安装 Python，且工程根目录有 .venv。
  echo 也可在工程根目录手动运行：  python -m qc_host
  pause
)
