# Raspberry Pi Setup Instructions (Linux ARM64 / Python 3.13)

To run the Face Attendance System on a Raspberry Pi, configure the environment for the ARM architecture using system packages for heavy dependencies, then compile the native executable.

---

## 1. Package the Code on Windows
1. Double-click `build_pi_zip.bat` on your Windows PC.
2. This generates a clean `VisionAttendance_Pi.zip` containing all python files, spec sheets, and models (excluding redundant raw backups).
3. Copy `VisionAttendance_Pi.zip` to your Raspberry Pi.

---

## 2. Install System Dependencies
On your Raspberry Pi, open a terminal and install the pre-compiled system libraries for OpenCV and PySide6 to avoid long ARM compilation times:
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv python3-pyside6 libhdf5-dev build-essential python3-dev
```

---

## 3. Set Up the Python Environment
Create the virtual environment enabling access to the system packages so that `cv2` (OpenCV) and `PySide6` are inherited:
```bash
# Extract files
unzip -o VisionAttendance_Pi.zip -d ~/visionattendance
cd ~/visionattendance

# Create venv with system site packages enabled
python3 -m venv --system-site-packages venv
source venv/bin/activate
```

---

## 4. Install Python Packages & Build InsightFace
Install other Python dependencies and compile the `insightface` C++ extensions inside the virtual environment:
```bash
# Install cython (required to build insightface)
pip install cython

# Install all other python requirements
pip install -r requirements.txt

# Install and build insightface
pip install insightface
```

---

## 5. Compile the Native Executable
Force PyInstaller to compile inside the virtual environment:
```bash
# Give execute permission to the build script
chmod +x scripts/build_pi.sh

# Run the build script
./scripts/build_pi.sh
```
*Note: Your compiled standalone folder will be generated at `dist/pi_exe/`.*

---

## 6. Run the Application
* **To run directly via Python**:
  ```bash
  python gui.py
  ```
* **To run the compiled binary**:
  ```bash
  ./dist/pi_exe/VisionAttendance
  ```

---

## 7. Enable Desktop Shortcut (Double-Click Launch)
To run the executable by double-clicking a desktop icon:
1. Copy the desktop launcher to your Pi Desktop:
   ```bash
   cp AttendanceSystem.desktop ~/Desktop/
   ```
2. You can now launch the app directly from your desktop grid!

