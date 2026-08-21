@echo off
title Building Vision Attendance Executable
echo =========================================================
echo    Vision Attendance System - Building Standalone EXE
echo =========================================================

:: 1. Detect and Activate Virtual Environment
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
    echo [INFO] Activated virtual environment: ..\venv
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [INFO] Activated virtual environment: venv
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [INFO] Activated virtual environment: .venv
) else if exist "env\Scripts\activate.bat" (
    call env\Scripts\activate.bat
    echo [INFO] Activated virtual environment: env
) else (
    echo [INFO] Using current active Python environment.
)

:: 2. Clean old build outputs
echo [INFO] Cleaning previous builds...
if exist build ( rmdir /s /q build )
if exist dist\VisionAttendance ( rmdir /s /q dist\VisionAttendance )

:: 3. Run PyInstaller via Python Module
echo [INFO] Running PyInstaller compilation on vision_attendance.spec...
python -m PyInstaller vision_attendance.spec --clean --noconfirm

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Check the error log above.
    pause
    exit /b %ERRORLEVEL%
)

:: 4. Ensure data directory structure exists in dist folder
echo [INFO] Finalizing runtime folder structure in dist\VisionAttendance...
if not exist dist\VisionAttendance\data ( mkdir dist\VisionAttendance\data )
if not exist dist\VisionAttendance\data\pending_snapshots ( mkdir dist\VisionAttendance\data\pending_snapshots )
if not exist dist\VisionAttendance\data\temp_recordings ( mkdir dist\VisionAttendance\data\temp_recordings )

echo.
echo =========================================================
echo    BUILD SUCCESSFUL!
echo    Executable ready at: dist\VisionAttendance\VisionAttendance.exe
echo =========================================================
pause

