# Vision Attendance System (Standalone Node)

🚀 **The Zero-Touch AI Face Recognition Attendance System**

This system is designed for autonomous, unattended operation with high-speed AI processing and real-time cloud synchronization.

---

## 📦 Production Deployment (Ready-to-Run)

The application has been bundled into a standalone Windows package. 
**No Python, database, or library installation is required on the target machine.**

### 1. Find the EXE
Navigate to the `dist/VisionAttendance` folder created by the build process.

### 2. First-Time Setup
1. Run `VisionAttendance.exe`.
2. Go to **Cloud Setup** in the sidebar.
3. Enter your API URL and credentials, then click **Establish Cloud Link**.
4. Go to **Cameras** and select your NVR channels.

### 3. Zero-Touch Operation
- The app automatically registers itself to start with Windows.
- Upon launch, it will automatically connect to your cameras and start the AI workers without any human intervention.
- All recordings are processed and deleted instantly to save disk space.

---

## 🛠️ Development Setup (Optional)

If you wish to modify the code or run from source:

1. **Environment Setup**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Models**
   Ensure the `data/models` folder contains the `buffalo_l` insightface models.

3. **Run**
   ```powershell
   python gui.py
   ```

---

## ✨ Features
- **Parallel AI Workers**: 6 high-speed slots for simultaneous camera and backlog processing.
- **Hybrid Path Manager**: Keeps models bundled inside the app while storing data externally.
- **Aggressive RTSP Handshaking**: 5-second connection timeouts for maximum UI responsiveness.
- **Auto-Sync Heartbeats**: Real-time arrival and "Last Seen" updates on the cloud dashboard.
