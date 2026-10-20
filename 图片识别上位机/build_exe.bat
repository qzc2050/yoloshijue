@echo off
setlocal

chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

if exist dist\*.exe del /q dist\*.exe

"%PY%" -m pip install -q --upgrade opencv-python-headless Pillow PyMuPDF pyinstaller
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name "铜环视觉检测" --hidden-import cv2 --hidden-import enhance --hidden-import hikrobot_camera --collect-all cv2 app.py

echo.
echo Build finished. The exe is in dist\铜环视觉检测.exe
