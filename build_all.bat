@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Vision Attendance System - Build Script
echo ============================================================

:: 1. Setup Environment
echo [1/4] Activating Virtual Environment...
if not exist "venv" (
    echo Error: venv directory not found!
    exit /b 1
)

:: Clean old build artifacts
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

:: 2. PyInstaller Build
echo [2/4] Running PyInstaller (Clean Build)...
.\venv\Scripts\python -m PyInstaller vision_attendance.spec --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo Error: PyInstaller build failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Searching for Inno Setup Compiler (ISCC)...

:: Search paths for ISCC.exe
set "ISCC_PATH="
for %%P in (
    "C:\Inno Setup 7\ISCC.exe"
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    "C:\Program Files\Inno Setup 6\ISCC.exe"
    "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
) do (
    if exist "%%~P" set "ISCC_PATH=%%~P"
)

if "!ISCC_PATH!"=="" (
    echo.
    echo WARNING: ISCC.exe not found in standard paths.
    echo Please compile 'installer.iss' manually using Inno Setup.
) else (
    echo Found ISCC at: !ISCC_PATH!
    echo [4/4] Compiling Installer...
    "!ISCC_PATH!" installer.iss
    
    if !ERRORLEVEL! EQU 0 (
        echo.
        echo SUCCESS: VisionAttendance_Setup.exe has been generated!
    ) else (
        echo.
        echo Error: Inno Setup compilation failed!
    )
)

echo.
echo ============================================================
echo   Build process complete.
echo ============================================================
pause
