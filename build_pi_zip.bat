@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Vision Attendance System - Pack Raspberry Pi Source Build
echo ============================================================

:: Check if virtual environment Python exists
if not exist "venv\Scripts\python.exe" (
    echo Error: venv directory or Python executable not found!
    echo Please make sure you have created the virtual environment.
    exit /b 1
)

echo [1/2] Running Python packager script...
.\venv\Scripts\python.exe build_pi_zip.py

if %ERRORLEVEL% NEQ 0 (
    echo Error: Packaging failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/2] Package process complete.
echo ============================================================
pause
