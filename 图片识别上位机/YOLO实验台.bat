@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set TEMP=E:\tmp
set TMP=E:\tmp
if not exist E:\tmp mkdir E:\tmp

cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
  start "" /D "%cd%" ".venv\Scripts\pythonw.exe" yolo_tools\yolo_lab.py
) else (
  echo .venv not found. Run from the project folder that contains .venv
  pause
)
