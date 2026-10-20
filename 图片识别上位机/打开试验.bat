@echo off
cd /d "%~dp0"
pythonw -m qc_host.vision.lab_view
if errorlevel 1 (
    python -m qc_host.vision.lab_view
    echo.
    pause
)
