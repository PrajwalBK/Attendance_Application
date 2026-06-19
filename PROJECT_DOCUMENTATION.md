# Vision Attendance System - Project Documentation

Welcome to the comprehensive documentation for the **Vision Attendance System**. This document provides an in-depth look at every file, folder, and functionality within this project, explained in a simple and structured way.

---

## 1. Project Overview
The **Vision Attendance System** is an AI-powered security and attendance solution. It uses advanced Face Recognition to identify employees or visitors via CCTV cameras (RTSP) or local webcams.

### Key Unique Selling Points (USPs):
- **Forensic Processing**: Unlike other systems that process live video (and drop frames when CPU is busy), this system records video to disk in small segments and processes them sequentially. This ensures **100% detection accuracy** without frame loss.
- **Zero-Touch Configuration**: Automatically discovers NVRs and IP cameras on the network.
- **Site Migration**: Automatically reconfigures itself if moved to a new network/office.
- **Cloud Sync**: Synchronizes face identities and logs with a central cloud server.

---

## 2. Directory Architecture

```text
vt_application_v6/
├── config/             # System settings & configuration managers
├── core/               # Main logic: AI, Camera Handling, Recording
├── database/           # Data persistence (Local MySQL & Cloud API)
├── data/               # Models, temp recordings, and face snapshots
├── ui/                 # User Interface (PySide6 / Qt)
├── logs/               # Application & Debug logs
├── scripts/            # Build and utility scripts
├── tests/              # Unit and integration tests
├── gui.py              # Main GUI entry point
├── main.py             # CLI entry point (alternative)
└── requirements.txt    # Library dependencies
```

---

## 3. Detailed File breakdown

### 📂 Root Directory
| File | Functionality |
| :--- | :--- |
| `gui.py` | **The Brain of the UI.** Handles the main window, navigation sidebar, and button controls. |
| `main.py` | **CLI Mode.** A terminal-based version of the system for lightweight registration or testing. |
| `api.py` | **Local Server.** A Flask/FastAPI based interface (if used) for external data access. |
| `installer.iss` | **InnoSetup Script.** Used to build the Windows `.exe` installer. |

---

### 📂 `config/` (Settings Management)
This folder handles everything related to how the app remembers your settings.
- `config.py`: Central configuration logic (API URLs, directory paths, thresholds).
- `cam_config_manager.py`: Specifically manages `cam_config.json` (Camera IPs, roles, and types).
- `auth_manager.py`: Securely stores and loads login credentials.
- `db_config_manager.py`: Stores database connection strings.

---

### 📂 `core/` (The Engine Room)
This is where the heavy lifting happens.
- `gui_workers.py`: Manages the background "Backend Controller" thread that coordinates cameras and AI.
- `multiprocess_handler.py`: **AI Worker.** Runs in a separate process to detect faces in recorded clips.
- `camera.py`: **Threaded Camera.** High-performance video capture that doesn't freeze the UI.
- `recorder.py`: **Video Recorder.** Safely saves video streams into 30-second segments.
- `face_recognition.py`: **Recognition Engine.** Uses InsightFace to turn face images into mathematical vectors.
- `attendance_tracker.py`: Decides if a detected face counts as a "Login" or "Logout".
- `video_processor.py`: Draws boxes and names on the video frames.
- `startup_manager.py`: Handles "Start on Boot" functionality.

---

### 📂 `ui/` (The Visuals)
Built using Python PySide6 (Qt).
- `camera_page.py`: The "Grid View" showing all active camera streams.
- `records_page.py`: Shows history of detections, times, and face snapshots.
- `cctv_setup.py`: A setup wizard to help users add new cameras.
- `registration_page.py`: Interface to add new people to the system.
- `theme_manager.py`: Handles the "Dark Mode" and "Glassmorphism" aesthetics.

---

### 📂 `database/` (Data & Cloud)
- `database.py`: Handles local MySQL storage for offline reliability.
- `api_client.py`: The bridge to the cloud server (handles uploads and downloads).
- `offline_storage.py`: Temporarily stores data if the internet goes down.

---

## 4. How the System Works (Workflow)

1.  **Capture**: `camera.py` pulls video from your CCTV NVR.
2.  **Segment**: `recorder.py` saves this video into small files (segments) every 30 seconds.
3.  **Analyze**: `multiprocess_handler.py` picks up the file, scans every frame, and finds faces.
4.  **Recognize**: `face_recognition.py` matches found faces against the "Registered Faces" list.
5.  **Log**: The system checks if the person is coming "In" or going "Out" and saves it to the DB.
6.  **Sync**: `api_client.py` sends the attendance record and a snapshot to the Cloud Dashboard.

---

## 5. Technology Stack
- **Language**: Python 3.11+
- **GUI Framework**: PySide6 (Qt for Python)
- **AI Libraries**: InsightFace (Deep Learning), ONNX Runtime (CPU Optimization), OpenCV.
- **Database**: MySQL (Local) + REST API (Cloud).
- **Video Processing**: FFmpeg (via OpenCV).

---

## 6. Maintenance & Troubleshooting
- **Logs**: If something goes wrong, check `logs/` or `worker_debug.log`.
- **Snapshots**: Found faces are saved as images in `data/attendance_snapshots`.
- **Corrupt Clips**: If a camera fails, records are moved to `data/failed_recordings`.

---
*Documentation generated for the Vision Attendance System.*
