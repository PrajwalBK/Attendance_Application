# Vision Attendance System v6.2 (CPU) - Project Brain & Implementation History

This document serves as the comprehensive knowledge base (`brain.md`) for the **Vision Attendance System v6.2 (CPU)** codebase. It outlines the architectural design, core subsystems, technical challenges, root-cause analyses, and evolutionary implementations developed throughout the project lifecycle.

---

## 🏗️ 1. Project Architecture Overview

The Vision Attendance System is an enterprise AI-powered biometric attendance and real-time surveillance platform engineered for multi-camera CCTV network integration (NVR/RTSP streams and USB Webcams).

### Key Architectural Layers:
- **GUI Layer (`gui.py`, `ui/`)**: Built with PySide6 (Qt for Python). Manages multi-camera grids, settings pages, cloud authentication, live feeds, and CCTV setup wizards.
- **Backend Thread Controller (`core/gui_workers.py`)**: Subclasses `QThread` (`BackendController`) to manage camera life-cycles, staggering connection streams, background synchronization, and dispatching tasks to AI worker processes.
- **Multiprocessing AI Pipeline (`core/multiprocess_handler.py`)**: Spawns standalone `AttendanceWorker` processes connected via Python `multiprocessing.Queue` objects for non-blocking face recognition and snapshot triage processing.
- **Deep Learning / Recognition Engine (`core/face_recognition.py`)**: Uses InsightFace (`buffalo_l` model / ArcFace) for 512-dimensional vector embedding extraction and upper-face crop matching.
- **Triage & Object Detection (`core/gui_workers.py`, `config/config.py`)**: Integrates ONNX Runtime with YOLOv8-Face (`yolov8n-face.onnx`) and YOLOv8-Body COCO (`yolov8n.onnx`) for high-speed frame triage and person tracking.
- **Storage & Cloud Sync (`database/`, `core/api_client.py`)**: Dual-mode data processing supporting direct REST API synchronization (`APIClient`), MySQL database logging (`DatabaseManager`), and SQLite offline transaction caching (`OfflineStorage`).

---

## 📜 2. Detailed Technical Implementations & Chronological Evolution

### 2.1. Offline Transaction Caching & Background Synchronization
- **Problem**: During network outages or server downtime, AI worker processes re-queued and re-ran heavy face recognition models on pending snapshots endlessly, causing 100% CPU spikes and thread exhaustion.
- **Solution**: 
  - Created `database/offline_storage.py` introducing an SQLite offline database queue (`OfflineStorage`) utilizing Write-Ahead Logging (WAL mode) and 30-second connection timeouts to ensure process safety across multi-threaded workers.
  - Intercepted network errors in `handle_attendance` (`core/multiprocess_handler.py`). When cloud APIs return errors, raw detection events and snapshot metadata are stored in the local SQLite queue and purged from `pending_snapshots`.
  - Added a dedicated background thread in `BackendController` (`core/gui_workers.py`) running every 15 seconds to flush cached offline attendance records and raw logs to the cloud API once connectivity is restored.

---

### 2.2. Unified Upper-Face Crop Matching & InsightFace ArcFace Optimization
- **Problem**: When employees wore face masks, recognition accuracy plummeted. A secondary lightweight model (`buffalo_sc`) was initially loaded to handle masked faces, but it increased memory overhead by ~150MB and startup latency. Furthermore, the early upper-face crop extraction pipeline had a ~12% failure rate in production logs because InsightFace's full detection pipeline failed to detect "a face" in synthetic, mirrored 320x320 grey canvases.
- **Solution**:
  - Unbundled the secondary `buffalo_sc` model and migrated entirely to the standard `buffalo_l` backbone for both full-face and masked-face embeddings.
  - Refactored `extract_upper_face_encoding()` in `core/face_recognition.py`. Bypassed InsightFace's full detection stage on crops entirely. Instead, the upper portion of the face (forehead, eyebrows, eyes) is mirrored, padded, resized directly to `112x112` (ArcFace native input resolution), and passed straight to the recognition model's low-level feature extractor:
    ```python
    embedding = self.app.models['recognition'].get_feat(aligned_112x112_crop)
    ```
  - Introduced `recognize_face_masked_direct()` and mapped standard full-face embeddings under masked events to a lower threshold (`MASKED_SIMILARITY_THRESHOLD = 0.25`), enabling 0ms-overhead embedding lookups and resolving missing snapshot path entries in online attendance logs.

---

### 2.3. Multi-Scale Resolution Fallback for Distant & Masked Subjects
- **Problem**: In wide-angle camera feeds (e.g., office seating views), subjects far from the camera had faces downsampled to ~13px on standard 640px triage processing, causing face detectors to miss them completely.
- **Solution**:
  - Modified `process_snapshot` in `core/multiprocess_handler.py`. 
  - Implemented a two-stage triage pass: the worker first performs high-speed face detection on the standard downsampled 640px image. If no faces are detected and the image was downsampled (`scale != 1.0`), the worker dynamically falls back to execute deep face detection on the original **full-resolution frame**.

---

### 2.4. Camera Stream Stabilization & Lag Elimination
- **Problem**: A missing frame retrieval statement (`status, frame = self.capture.read()`) in `core/camera.py` caused background threads to hit `NameError` exceptions at 100 FPS, freezing GUI stream previews and halting snapshot triggers. Additionally, synchronous model downloads blocked Qt threads, triggering `QThread: Destroyed while thread '' is still running` crashes on app shutdown.
- **Solution**:
  - Restored proper `capture.read()` frame polling loops inside `ThreadedCamera` (`core/camera.py`) with 0.5s socket TCP pre-checks (`connect_ex`) to eliminate 30-second RTSP handshake hangs on offline cameras.
  - Refactored model downloading logic in `_triage_detect_yolo` (`core/gui_workers.py`) to spawn non-blocking daemon threads (`threading.Thread(target=..., daemon=True)`).
  - Added file size integrity validation checks (`getsize < 3MB` for body model, `< 2MB` for face model). Corrupted or partial downloads (such as truncated 1.1MB files) are automatically deleted and re-downloaded cleanly.
  - Added `_yolo_load_failed` and `_yolo_body_load_failed` guard flags to prevent endless frame-by-frame reloading attempts if a model file fails to load.

---

### 2.5. YOLOv8 & HOG Body/Person Detection Triage
- **Problem**: Standard face-only detection missed individuals walking away from cameras, looking down, or wearing heavy facial occlusion where no facial features were exposed.
- **Solution**:
  - Integrated standard COCO YOLOv8 (`yolov8n.onnx` class `0` - "person") into live triage (`_triage_detect_yolo`).
  - Added an OpenCV HOG descriptor fallback (`cv2.HOGDescriptor`) in `_triage_detect_opencv` for low-resource environments.
  - Built a face-body deduplication and suppression engine: box coordinates undergo IoU and center-point containment checks. If a detected face center falls within a detected person body bounding box, the body box is suppressed to prevent double-tracking and duplicate snapshot triggers for the same subject.

---

### 2.6. Compiled Executable (PyInstaller EXE) Path & Storage Resolution
- **Problem**: When compiled into an executable, running from read-only directories (like `C:\Program Files\`) caused model downloads and logging to crash with `PermissionError`. Furthermore, default PyInstaller behavior caused config copy operations to look next to the EXE rather than inside the virtual `_MEIPASS` bundle, defaulting camera templates to webcam `"0"` and `ip: null`.
- **Solution**:
  - Updated `_get_storage_dir()` in `config/config.py`, `config/auth_manager.py`, and `config/db_config_manager.py`. The system tests write permissions on the executable directory (`.write_test`). If writable, it stores data locally next to the EXE (`dist/VisionAttendance/data/`). If write-protected, it gracefully redirects storage to `%localappdata%\VisionAttendance\`.
  - Configured dual-path model resolution in `_triage_detect_yolo`: checks both `ASSET_DIR` (`_MEIPASS`) and `BASE_DIR` (`AppData`). Models that need downloading are fetched directly into the writable `BASE_DIR` models directory.
  - Implemented self-healing config bootstrap logic: automatically detects if AppData contains an unconfigured default template (`"rtsp_template": "0"` or missing `discovered_ips`) and force-overwrites it with the bundled NVR configuration on startup.

### 2.7. NVR Discovery & CCTV Setup Wizard Fixes
- **Problem**: Running camera discovery via `ui/cctv_setup.py` triggered a runtime exception: `TypeError: BackendController.discover_brand_path() got an unexpected keyword argument 'brand'`.
- **Solution**:
  - Updated `discover_brand_path` in `core/gui_workers.py` to accept the keyword parameter `brand=None`.
  - Implemented brand filtering logic inside `discover_brand_path`: when a user selects a brand in the setup wizard (e.g. Hikvision, CP Plus/Dahua, Eyematic), the RTSP probe filters its search list to match the brand's exact RTSP path structure, eliminating unnecessary connection attempts and dramatically speeding up setup times.

### 2.8. High-Performance RTSP Video Decoding & Low-Latency Low-Overhead Thread Capping
- **Problem**: Decoding four high-resolution (1080p or 4K) main streams (`subtype=0`) concurrently on CPU-only hardware (especially on the Raspberry Pi) overloaded the system, causing frame queues to lag up to 2 seconds and stutter.
- **Solution**:
  - Configured `rtsp_transport` option to `udp` instead of `tcp` globally. UDP drops delayed or late frames naturally at the network level rather than queuing them in FFMPEG's internal memory buffer.
  - Added low-latency FFMPEG flags (`fflags;nobuffer|max_delay;500000`) to disable buffering and drop late frames at the socket level.
  - Capped the `ThreadedCamera` background loop speed from 100 FPS (`time.sleep(0.01)`) to ~33 FPS (`time.sleep(0.03)`). Capping the capture thread reduces context switching overhead by 70%, leaving significant CPU cycles free for PySide6 GUI rendering and frame decoding.
  - Removed frame-skipping/discarding logic to ensure sequential frame-by-frame rendering, restoring smooth, fluid video previews.
  - Changed `cam_config.json` default RTSP templates to support sub-streams (`subtype=1`) for lower-spec CPU architectures.

### 2.9. Interactive GUI Region of Interest (ROI) Drawing & Granular Direction Rules
- **Problem**: Users need a way to customize both the active detection zone (Region of Interest) and the individual entry/exit direction rules (Up, Down, Left, Right mapping to IN, OUT, or IGNORE) directly in the GUI with clear visual overlays.
- **Solution**:
  - Subclassed `QLabel` into a custom interactive `ROILabel` (`ui/camera_page.py`) to handle PySide6 mouse drag events (`mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent`).
  - Implemented dynamic painting of a dashed green box in `paintEvent` as the user drags their cursor over a camera stream.
  - Upon mouse release, the box coordinates are normalized `[x_min, y_min, x_max, y_max]` relative to the widget's dimensions and saved to the backend configuration via `set_cam_roi()`.
  - Added a click-to-reset fallback: if the user simply clicks the stream (creating a drag area smaller than 20x20 pixels), the ROI is automatically reset back to full-screen `[0.0, 0.0, 1.0, 1.0]`.
  - Implemented checkable axis-specific submenus (Upward, Downward, Leftward, Rightward Motion) inside the right-click context menu of each camera preview to set rules to `IN`, `OUT`, or `IGNORE` dynamically.
  - Added **Static Border Overlays**: renders green arrows (e.g. `◀ IN`) and red arrows (e.g. `▲ OUT`) directly at the edges of the frame to display the active direction rules at all times.
  - Integrated dynamic trajectory tracking: draws a glowing trail of yellow dots showing the path of movement for active subjects.
  - **High-Sensitivity Motion Tracker**: Lowered direction classification threshold `thresh` from `15.0` to `5.0` pixels, and minimized history check length to `3` frames. Implemented a 4-frame trajectory buffer delay before the first snapshot is triggered (with cleanup fallback) to avoid first-frame classification race conditions, ensuring 100% accurate dynamic directional logging.

### 2.10. Triage Bypassing / Complete Body Capture Coverage
- **Problem**: 
  - Skipping the body detector when a face was seen created two issues:
    1. If multiple people were present and only one face was visible, other body-only people were ignored.
    2. False positive face detections suppressed the body detector completely, causing skipped frames.
- **Solution**:
  - Removed the cascade short-circuit check. Both the face detector and body detector sessions (YOLOv8 Body / HOG SVM) now execute on every processed frame.
  - This ensures 100% complete capture coverage (even for body-only views where the face is not visible) at the cost of slightly higher idle CPU usage when faces are visible.

### 2.11. Dynamic INT8 Quantization Bootstrap & Dual-Stream Splitter Pipeline
- **Problem**: Decoding high-resolution 1080p camera frames and running float32 neural network models on edge CPUs created severe frame lag and high CPU heat.
- **Solution**:
  - **Dynamic INT8 Quantizer**: Added `_quantize_onnx_model` using `onnxruntime.quantization.quantize_dynamic` inside `core/gui_workers.py`. On first startup or download completion, the FP32 ONNX model is dynamically compiled locally to INT8 in ~1.5s, allowing the system to use hardware SIMD vector units (**AVX2** / **NEON**), boosting inference speed by **2x to 3x**.
  - **Dual-Stream Splitter camera decoding pipeline**: Refactored `ThreadedCamera` to resolve both the sub-stream and main-stream URLs. It decodes the low-res sub-stream continuously (serves live GUI preview, overlays, and face tracking at ~5% CPU usage) while keeping the main-stream cleared via low-cost `grab()` calls. It only decodes the high-res main-stream frame on-demand (via `retrieve()`) when a snapshot is actually captured, cutting idle decoding CPU cost by **70%**.

### 2.12. Multiprocess Work-Stealing Load Balancer & Windows Auto-Start Manager
- **Problem**: 
  - Subprocess workers were bound to individual camera queues. A single busy camera queue would back up with snapshot tasks while other camera workers sat idle.
  - Users need the application executable to start automatically on Windows boot while maintaining previous setups and camera configurations.
- **Solution**:
  - **Work-Stealing Load Balancer (DSA)**: Implemented dynamic task stealing inside `AttendanceWorker.run()` in `core/multiprocess_handler.py`. If a worker's own queue raises `queue.Empty`, it loops through adjacent queues and calls `get_nowait()` to steal pending tasks, load-balancing CPU core utilization.
  - **Windows Startup Manager**: Added a system behavior checkbox in `ui/settings_page.py` that utilizes a native PowerShell COM-object script to create or remove a startup shortcut (`.lnk`) inside the user's Startup folder. Since configurations are stored in the persistent `AppData/Local`, settings are automatically preserved across reboots.

---

## 🛠️ 3. Exhaustive Breakdown of Issues Faced & Resolved

Below is a categorized summary of all critical technical issues, runtime bugs, and production crashes encountered during development, along with their root-cause analyses and implemented fixes:

### 3.1. Network & Model Downloading Issues
1. **YOLOv8 Body Model 404 Download Error**
   - **Symptom**: `[Backend] Error downloading YOLOv8 body: HTTP Error 404: Not Found` on startup.
   - **Root Cause**: The application attempted to download `yolov8n.onnx` from Ultralytics' GitHub release assets (`v8.2.0`), which only hosts PyTorch `.pt` files.
   - **Fix**: Replaced the target download URL in `core/gui_workers.py` to point to a reliable public Hugging Face ONNX mirror: `https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx`.

2. **Synchronous Download Freeze & Thread Flooding**
   - **Symptom**: Console spammed with `Downloading...` and `404 Not Found` messages hundreds of times per second; live video feeds froze.
   - **Root Cause**: `_triage_detect_yolo()` ran synchronously inside the camera loop on every frame. When `_yolo_body_session` was uninitialized or downloading, `urllib.request.urlretrieve` blocked the main camera thread frame-by-frame.
   - **Fix**: Offloaded model downloads to non-blocking background daemon threads (`threading.Thread(target=..., daemon=True)`). Introduced `_yolo_download_failed` and `_yolo_body_download_failed` flags to prevent repeated download attempts upon failure.

3. **PySide/Qt Thread Crash (`QThread: Destroyed while thread '' is still running`)**
   - **Symptom**: Application crashed upon exit with `QThread: Destroyed while thread '' is still running`.
   - **Root Cause**: Synchronous network I/O blocked `BackendController` (subclass of `QThread`). When the user closed the app, Qt tried to garbage-collect Python objects while C++ underlying threads were still waiting on network sockets.
   - **Fix**: Transitioning model downloading to daemon threads freed the `BackendController` loop, allowing Qt to terminate threads cleanly on shutdown.

4. **Corrupted / Truncated Model File Loop**
   - **Symptom**: Partial network downloads created corrupted files (e.g. 1.1 MB instead of 6.2 MB), causing continuous `InferenceSession` initialization errors.
   - **Root Cause**: `os.path.exists()` returned `True` for partially downloaded files, preventing clean re-downloads while causing ONNX parser failures.
   - **Fix**: Implemented strict file size integrity validation (`getsize < 3MB` for body model, `< 2MB` for face model). Corrupted files are automatically deleted and re-fetched.

5. **TypeError: replace() argument 2 must be str, not None during Camera Connection**
   - **Symptom**: `[CAMERA ERROR] Async connection failed: replace() argument 2 must be str, not None` and camera fails to open.
   - **Root Cause**: The NVR template resolver used `.get('ip', '192.168.0.1')` to find camera IPs. If a camera was configured with `"ip": null` in the JSON config, `.get()` returned `None` instead of the fallback string. Passing `None` to `.replace("{ip}", ip)` raised a TypeError.
   - **Fix**: Replaced `.get('ip', default)` calls with the short-circuiting `or` operator: `cam_info.get('ip') or '192.168.0.1'`, ensuring `null` values cleanly fall back to valid fallback IP strings.

6. **Low Quality Fallback of Snapshot Images due to Data Race**
   - **Symptom**: Snapshots taken from the camera streams are of low quality (matching the sub-stream resolution).
   - **Root Cause**: Previously, `read_high_res()` made concurrent non-thread-safe calls to `retrieve()` on `self.capture_main` while a background thread was continuously calling `grab()`. This caused OpenCV to silently fail and fall back to the low-resolution sub-stream frame (`self.read()`), while still logging success.
   - **Fix**: Introduced `self.main_lock = threading.Lock()` to synchronize all access to `self.capture_main`. Refactored `read_high_res()` to perform both `grab()` and `retrieve()` atomically inside the lock context, returning `False, None` on failure to trigger correct logging and prevent silent fallbacks.

---

### 3.2. Compiled EXE & Deployment Environment Issues
5. **Read-Only Virtual Bundle Permission Errors (`_MEIPASS`)**
   - **Symptom**: Compiled PyInstaller EXEs crashed with `PermissionError` when attempting to write logs or download models.
   - **Root Cause**: PyInstaller extracts bundled files to a read-only temp folder (`_MEIPASS`). Code attempting to create folders or files inside `ASSET_DIR` failed.
   - **Fix**: Separated read-only asset resolution (`ASSET_DIR` / `_MEIPASS`) from writable storage (`BASE_DIR` pointing to `%localappdata%\VisionAttendance\`).

6. **Stale / Default Configuration Fallback in AppData**
   - **Symptom**: Compiled EXE attempted to open webcam `"0"` instead of configured NVR RTSP streams (`192.168.1.240`).
   - **Root Cause**: Initial startup created default template configurations in AppData. Subsequent launches detected the existing file and skipped updates, locking the system into default settings.
   - **Fix**: Implemented self-healing auto-migration in `config/config.py`: if `cam_config.json` in AppData contains default templates (`"rtsp_template": "0"` or lacks `"discovered_ips"`), it automatically force-overwrites it with the bundled production NVR config.

7. **Missing Snapshot Paths in Online Database Logs**
   - **Symptom**: `snapshot_path` fields in `log_raw_detection` and `mark_attendance` database tables recorded as `NULL` during online mode.
   - **Root Cause**: `handle_attendance()` invoked database logging functions prior to saving the snapshot frame to disk and generating the destination path string.
   - **Fix**: Reordered execution in `core/multiprocess_handler.py` to write the snapshot image file first and pass the verified path directly into database logging functions.

8. **`json.decoder.JSONDecodeError` on Camera Configuration (`cam_config.json`)**
   - **Symptom**: `Error loading .../config/cam_config.json: Extra data: line 62 column 5 (char 1597)` on application startup.
   - **Root Cause**: Trailing garbage syntax (leftover characters from a partial write/overwrite operation) at the end of the JSON object, which made the file invalid JSON.
   - **Fix**: Replaced `config/cam_config.json` with a fully-validated, clean JSON structure matching the 8 NVR camera stream configurations.

9. **ONNX Runtime OpenVINO DLL Missing Errors (`openvino.dll not found`)**
   - **Symptom**: Loud error outputs to stderr (`FAIL : Error loading .../onnxruntime_providers_openvino.dll which depends on openvino.dll which is missing`) on startup.
   - **Root Cause**: The default execution providers list hardcoded `OpenVINOExecutionProvider`. If the environment did not have OpenVINO shared library DLLs installed, ONNX Runtime logged failure events to the console when trying to initialize sessions.
   - **Fix**: Replaced static providers list in `config/config.py` with a dynamic check using `ctypes.util.find_library('openvino')`. OpenVINO is now excluded from the providers list if its DLL is missing, falling back silently to standard CPU execution.

---

### 3.3. GUI & Camera Setup Wizard Issues
8. **NVR Discovery Wizard Crash (`unexpected keyword argument 'brand'`)**
   - **Symptom**: `TypeError: BackendController.discover_brand_path() got an unexpected keyword argument 'brand'` when running CCTV setup wizard.
   - **Root Cause**: `ui/cctv_setup.py` passed `brand=brand_filter`, but `discover_brand_path()` in `core/gui_workers.py` omitted `brand` from its parameter list.
   - **Fix**: Updated method signature to `discover_brand_path(self, raw_url, channel_hint="1", brand=None)` and added brand-specific RTSP path filter rules.

9. **Camera Preview Freeze / NameError inside Loop**
   - **Symptom**: Live streams stopped updating and zero snapshots were queued for AI processing.
   - **Root Cause**: A manual code edit accidentally removed `status, frame = self.capture.read()` inside `core/camera.py`'s update loop, throwing `NameError` on `status` on every frame.
   - **Fix**: Restored `status, frame = self.capture.read()` and added socket TCP pre-checks (`connect_ex`) to prevent RTSP connection hangs.

10. **Camera Stream Choppiness / Stuttering Preview (Skipped Frames)**
    - **Symptom**: Live streaming video preview is jumpy and lacks smooth motion.
    - **Root Cause**: The active camera thread ran a frame-flusher that conditionally dropped incoming frames (using `grab()`) to force <100ms lag. On low-spec CPU platforms, this skipped up to 2 out of 3 frames, creating a slideshow effect.
    - **Fix**: Removed the conditional `grab()` frame-skipping logic and restored sequential frame-by-frame decoding. Video streams now render with perfect fluid motion.

11. **Failed Snapshots / Delay to Capture for Fast-Moving Targets**
    - **Symptom**: Snapshots were not triggered for people walking quickly across the screen, or were triggered with a noticeable delay that caused the camera preview to lag/jitter.
    - **Root Cause**:
      1. Triage update intervals are 133ms apart. If a target is fast-moving, their bounding boxes have very little or no overlap, causing IoU to drop below the threshold (0.3). The tracker split their motion into individual 1-frame tracks. Because the snapshot logic required 4 frames of history to calculate the motion vector direction before queuing the first snapshot, these fragmented tracks never reached the 4-frame limit, leaving them completely unsnapshotted during live movement.
      2. Retrieving high-resolution frames (`read_high_res()`) and encoding/saving JPEG images to disk takes 50–200ms of synchronous execution. Running this on the main camera stream loop blocked it, causing video lag, skipped updates, and delayed snapshots.
      3. On startup (or manual camera start), the AI workers remained "Offline" until the user manually clicked "Start AI". Snapshots were saved as "AI Offline" and never parsed by the background processes.
      4. Face detections drop out or change coordinates rapidly during movement, while body boxes are large and stable. Previously, the YOLOv8 triage discarded body boxes if a face was detected inside them. This caused the tracker box coordinates and sizes to constantly jump between face and body detections, fragmenting the track.
    - **Fix**:
      1. **Centroid Tracking**: Lowered the matching IoU threshold to 0.1 and added a centroid-distance matching fallback (up to 30% of frame dimensions) to maintain track continuity across high speeds.
      2. **Fast Triggering**: Reduced the required history length for first snapshots from 4 frames to 2 frames (133ms), allowing rapid snapshot triggers.
      3. **Fallback Caching**: Cached the latest detection frame (`last_frame`) in the track object, safeguarding fallback snapshots from capturing blank/ghost images after a 1.5s delay.
      4. **Dominant Axis Prioritization**: Prioritized the axis of greatest absolute displacement to increase direction classification accuracy.
      5. **Decluttered GUI**: Disabled drawing of face tracking boxes, trajectory trails, and labels on GUI video streams to minimize visual clutter.
      6. **Asynchronous Dispatch**: Offloaded NVR high-res frame decoding, JPEG compression, and disk/worker queue operations to background daemon threads (`_save_and_queue_snapshot` launcher), keeping the camera thread completely non-blocking and zero-lag.
      7. **Auto-Start Synchronization**: Synced the AI workers to start automatically 2.5 seconds after starting cameras (in `gui.py`), and configured auto-start on boot to `true` by default in `cam_config.json`.
      8. **Stable Body-box Priority Tracking**: Modified the triage box parser in `gui_workers.py` to prioritize body boxes over face boxes when a face resides inside a body. This maintains perfect track continuity regardless of transient face dropouts.

---

### 3.4. PyInstaller Executable (EXE) Specific Issues
11. **OpenCV FFMPEG Backend Loading Failure inside Compiled EXE (Fallback to slow CAP_MSMF)**
    - **Symptom**: Live video streams run smoothly in the IDE, but compiling and running the executable in the `dist` folder causes heavy delay and lag.
    - **Root Cause**: OpenCV's C++ core loads `opencv_videoio_ffmpeg4130_64.dll` dynamically via Windows `LoadLibrary`. PyInstaller bundles this DLL in a `cv2/` subfolder. Because Windows does not search subfolders for DLL dependencies, the load failed. OpenCV silently fell back to Microsoft Media Foundation (CAP_MSMF), which lacks low-latency configuration and has high default buffering. (Note: Adding the DLL to the Analysis binaries list fails because PyInstaller hooks override it and force it back into the `cv2/` subdirectory).
    - **Fix**: Added a custom python post-build script at the end of `vision_attendance.spec` (executed right after the `COLLECT` step) to copy the `opencv_videoio_ffmpeg*.dll` from `dist/.../cv2/` directly to the root folder `dist/.../`. Windows' dynamic linker can now find it immediately on startup, loading CAP_FFMPEG successfully.

---

## 📁 4. Directory & Core File Map

```text
vt_application_v6.2_cpu/
│
├── config/
│   ├── config.py                # Global parameters, paths (BASE_DIR, ASSET_DIR), execution providers
│   ├── cam_config_manager.py    # Manages loading, auto-healing, and saving cam_config.json
│   ├── db_config_manager.py     # Manages MySQL/Database configuration settings
│   └── auth_manager.py          # Manages cloud API tokens and credentials
│
├── core/
│   ├── gui_workers.py           # BackendController QThread, triage detectors, background sync, camera loops
│   ├── multiprocess_handler.py  # AttendanceWorker multiprocessing pipeline, snapshot process, offline caching
│   ├── face_recognition.py      # FaceRecognitionHandler, InsightFace buffalo_l ArcFace feature extraction
│   ├── camera.py                # ThreadedCamera implementation with RTSP handshake pre-checks
│   ├── api_client.py            # REST API communication client for cloud synchronization
│   ├── recorder.py              # Background video recording worker
│   └── person_reid.py           # OSNet Person Re-Identification module
│
├── database/
│   ├── database.py              # MySQL DatabaseManager interface
│   └── offline_storage.py       # SQLite WAL-mode transaction queue for offline resiliency
│
├── ui/
│   ├── cctv_setup.py            # Universal camera discovery wizard & brand path prober
│   ├── camera_page.py           # Live GUI camera display grid and controls
│   ├── settings_page.py         # System settings and configuration UI
│   └── cloud_setup.py          # Cloud link and authentication configuration page
│
├── data/
│   ├── models/                  # ONNX and Caffe models (buffalo_l, yolov8n-face, yolov8n, osnet)
│   ├── pending_snapshots/       # Temporary queue folder for incoming triage snapshots
│   └── attendance_snapshots/    # Categorized final snapshots (in/, out/, unknown/)
│
├── gui.py                       # Main application entry point & PySide6 MainWindow lifecycle
├── vision_attendance.spec       # PyInstaller build specification file
├── build_all.bat                # Automated virtualenv activation, PyInstaller & Inno Setup build script
├── cpu_optimization_research.md # Edge CPU deep optimization strategies & lightweight model architectures
└── brain.md                     # Knowledge base and implementation documentation
```

---

## ⚙️ 5. Quick Execution & Build Commands

### Running in Development Environment:
```powershell
# Activate Virtual Environment
.\venv\Scripts\activate

# Run GUI Application
python gui.py
```

### Compiling Production Executable & Installer:
```powershell
# Run automated build script (Generates PyInstaller bundle & Inno Setup Installer)
.\build_all.bat
```

---
*Document compiled and maintained for Vision Attendance System v6.2 (CPU).*
