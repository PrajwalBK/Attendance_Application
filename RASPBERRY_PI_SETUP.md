# Raspberry Pi Setup Instructions

To run the Face Attendance System on a Raspberry Pi, you need to configure the environment for the ARM architecture and use the provided desktop launcher.

## 1. Move Files to the Raspberry Pi
1. Copy the entire `desktop_application-v1` folder to your Raspberry Pi. We recommend placing it in the home directory: `/home/prosper123/desktop_application-v1`.

## 2. Install System Dependencies
Open a terminal on your Raspberry Pi and run:
```bash
sudo apt update
sudo apt install python3-pip python3-venv python3-opencv libatlas-base-dev libhdf5-dev libhdf5-serial-dev
```

## 3. Set Up Python Environment
Create and activate a virtual environment:
```bash
cd /home/prosper123/desktop_application-v1
python3 -m venv venv
source venv/bin/activate
```

## 4. Install Python Packages
Install the required packages. **Crucially**, you must install the standard CPU version of `onnxruntime`, as the GPU version (`onnxruntime-gpu`) does not work on standard Raspberry Pi OS.

```bash
# Install OpenCV and basic requirements
pip install -r requirements.txt

# Manually uninstall any existing onnxruntime packages to prevent conflicts
pip uninstall -y onnxruntime onnxruntime-gpu

# Install the ARM-compatible CPU version
pip install onnxruntime
```

## 5. Configure for CPU Fallback
The `config/config.py` file is already set up to include `CPUExecutionProvider` as a fallback. When the app runs on the Pi, it will gracefully fall back from CUDA (GPU) to the CPU. 
_Note: Inference will be slower on the Pi CPU compared to a PC._

## 6. Enable Desktop Shortcut (Double-Click Launch)
We have provided `run_attendance.sh` and `AttendanceSystem.desktop` to let you launch the app directly from your desktop.

1. Ensure the script is executable:
   ```bash
   cd /home/prosper123/desktop_application-v1
   chmod +x run_attendance.sh
   ```
2. Open the `AttendanceSystem.desktop` file in a text editor.
   - Verify the `Exec` path matches where you put the folder (e.g., `/home/prosper123/desktop_application-v1`).
   - Verify the `Icon` path matches the location of the `prosper1.png` file.
3. Move or copy `AttendanceSystem.desktop` to your Raspberry Pi's Desktop:
   ```bash
   cp AttendanceSystem.desktop /home/prosper123/Desktop/
   ```
4. On your Raspberry Pi Desktop, double-click the **Attendance AI System** icon. 
   - You may be prompted to trust or make the file executable the first time you click it. Accept to launch the app.
   
If the app closes immediately when double-clicked, open the `pi_gui_log.txt` file generated in the application folder to see the error details.
