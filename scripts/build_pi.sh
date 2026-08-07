#!/bin/bash

# Vision Attendance System - Raspberry Pi Build Script
# ----------------------------------------------------
# NOTE: PyInstaller does not support cross-compiling. 
# Run this script ON the Raspberry Pi to build the native Pi executable.

echo "============================================================"
echo "  Vision Attendance System - Raspberry Pi Build Script"
echo "============================================================"

# 1. Activate Virtual Environment
echo "[1/3] Activating Virtual Environment..."
if [ ! -d "venv" ]; then
    echo "Error: 'venv' directory not found! Please set up the environment first (see RASPBERRY_PI_SETUP.md)."
    exit 1
fi
source venv/bin/activate

# 2. Run PyInstaller
echo "[2/3] Running PyInstaller..."
# Install PyInstaller if not present in venv
pip install pyinstaller --quiet

# Clean old builds
rm -rf build dist

# Compile using the spec file
python -m PyInstaller vision_attendance.spec --noconfirm

if [ $? -ne 0 ]; then
    echo "Error: PyInstaller build failed!"
    exit 1
fi

# 3. Rename Output to 'pi_exe'
echo "[3/3] Renaming output executable to 'pi_exe'..."
if [ -d "dist/VisionAttendance" ]; then
    mv dist/VisionAttendance dist/pi_exe
    echo "SUCCESS: Compiled application is located in 'dist/pi_exe/'"
    echo "You can launch the app on the Raspberry Pi by running: ./dist/pi_exe/VisionAttendance"
else
    echo "Warning: Compiled output folder not found at dist/VisionAttendance."
fi

echo "============================================================"
echo "  Build Complete."
echo "============================================================"
