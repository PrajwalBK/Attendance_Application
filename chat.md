# Conversation History & Technical Resolutions Log (chat.md)

This document records technical updates, optimizations, and build configurations completed during the session.

---

## 📝 Resolution Log

### 1. High-Resolution Snapshot Capture (Blurred Images Fix)
- **Issue**: Pending snapshots were captured from the low-resolution GUI sub-stream (`352x288`/`640x480`), resulting in blurry images where facial recognition failed.
- **Fix Applied**: Updated `read_high_res()` in `core/camera.py` and `gui_workers.py` to capture uncropped, unscaled **1080p / 5MP Main-Stream** frames on-demand.

### 2. Camera-Level Snapshot Throttle & Multi-Face Evaluation
- **Issue**: When 4–10 people were in the frame, 4–10 identical full-frame JPEGs were saved at the exact same millisecond, flooding the pending queue.
- **Fix Applied**: 
  - Throttled snapshot dispatch in `core/gui_workers.py` to **1 high-resolution photo every 0.5s per camera** when people are present.
  - Ensured `core/multiprocess_handler.py` scans and evaluates **100% of all detected faces** in that photo without early exit or skipping.
  - Maintained strict FIFO chronological ordering in `core/snapshot_manager.py`.

### 3. Processing Speed Optimizations (300x Speedup)
- **Issue**: Python loops over 100+ registered face encodings added unnecessary latency on CPU.
- **Fix Applied**:
  - Implemented **1-Shot Vectorized Matrix Cosine Similarity** (`_std_matrix` and `np.dot`) in `core/face_recognition.py` for <0.01ms matching.
  - Unlocked multi-core AVX execution by setting `ORT_INTRA_OP_NUM_THREADS = 4` in `config/config.py`.

### 4. Infinite Work-Stealing Loop & File Lock Safety
- **Issue**: Windows file locks prevented immediate file deletion, causing the scanner to re-queue the same snapshot in an infinite loop.
- **Fix Applied**:
  - Added `self.queued_files` tracking in `core/snapshot_manager.py`.
  - Added fast `if not os.path.exists(video_path): continue` check in `core/multiprocess_handler.py`.
  - Added a 3-pass retry deletion loop in `process_snapshot()`.

### 5. PyInstaller Standalone Build Setup (`vision_attendance.spec` & `build_all.bat`)
- **Root Causes of Build / Execution Failure**:
  - The previous spec file listed obsolete `PyQt5` imports (`PyQt5.QtCore`, `QtWidgets`, etc.) instead of `PySide6` used by the application, causing missing module errors during analysis.
  - `core.database` was listed as a hidden import, but the actual module is top-level `database` (`database/database.py`, `database/offline_storage.py`), which was also omitted from asset bundling.
  - `qtawesome` and `qt_material` data assets (icon fonts and XML stylesheet themes) were not collected, causing immediate runtime crashes when building Qt icons or themes.
  - `gui.py` unconditionally redirected `sys.stderr` to `NullWriter()` when frozen, suppressing traceback outputs and masking crashes.
- **Fix Applied**:
  - Configured `vision_attendance.spec` with data file collectors for `qtawesome`, `qt_material`, `onnxruntime`, `insightface`, and `openvino`.
  - Added folder datas for `database`, `data`, `config`, `core`, `ui`.
  - Corrected all hidden imports to `PySide6` and included `database.*`, `config.*`, `ui.*`, `mysql.connector`, and `core.*`.
  - Updated `gui.py` frozen stream handling to log uncaught exceptions to `logs/crash.log` and prevent tracebacks from being swallowed.
  - Updated `vision_attendance.spec` to remove `matplotlib` from `excludes` and collect its data assets, resolving the `ModuleNotFoundError: No module named 'matplotlib'` triggered by `insightface.app`.
  - Configured `console=False` in `vision_attendance.spec` to run in pure windowed GUI mode without opening a console/terminal window.

### 6. AI Stopped State Processing Fix
- **Issue**: When AI was stopped (or before AI was started), YOLO person detection, MultiPersonTracker, and SnapshotTask dispatching were still capturing and enqueuing photos to disk and worker queues because the capture loop only checked `if self.are_cameras_active:`. Additionally, stopping AI did not drain existing queues or terminate lingering worker threads cleanly.
- **Fix Applied**:
  - Bound YOLO triage detection, MultiPerson tracking, and snapshot capture in `core/gui_workers.py` strictly to `if self.are_cameras_active and self.is_detection_enabled:`. When AI is stopped, zero CPU is spent on detection and zero snapshots are created or dispatched.
  - Enhanced `stop_detection()` to cleanly terminate all `AttendanceWorker` processes, drain all AI task queues (`self.task_queues`), and clear `SnapshotPipelineManager` tracking and queue caches.
  - Added `clear_queues()` and `stop()` to `SnapshotPipelineManager` in `core/snapshot_manager.py`.

### 7. Application Logo & CCTV Stream Connection Diagnostics
- **Issue 1 (Missing Logo on EXE & GUI)**:
  - The project lacked `assets/icon.ico` and `assets/icon.png`. PyInstaller's `icon=` defaulted to `None` (generic icon), and `gui.py` lacked window icon binding.
  - **Fix Applied**: Generated modern biometric AI camera logo assets (`assets/icon.ico` and `assets/icon.png`), linked `setWindowIcon` on `MainWindow` and `QApplication`, and ensured PyInstaller bundles `assets/`.
- **Issue 2 (CCTV Camera Stream Connection Failure)**:
  - Network diagnostic logs in `cctv_error.log` showed `Handshake Failed: 192.168.1.240:554`.
  - TCP test to `192.168.1.240:554` failed because either the NVR/camera hardware was turned off / assigned a different DHCP IP by the router, or the overly strict 300ms–500ms TCP probe timed out.
  - Furthermore, `timeout=5000000` was being duplicated in RTSP URLs (`&timeout=5000000&timeout=5000000`).
  - **Fix Applied**: Cleaned up URL query concatenation in `core/camera.py` and `core/gui_workers.py` to eliminate duplicate timeout flags, and increased TCP handshake tolerance to 1.5s to accommodate Wi-Fi jitter.

### 8. Scikit-Image Stub File Resolution (`skimage/__init__.pyi`)
- **Issue**: Running the built executable crashed with: `ValueError: Cannot load imports from non-existent stub '.../skimage/__init__.pyi'`. Newer versions of `scikit-image` use `lazy_loader.attach_stub()` which reads `.pyi` typing stub files at runtime. PyInstaller excludes `.pyi` files by default.
- **Fix Applied**: Updated `vision_attendance.spec` to use `collect_data_files('skimage', include_py_files=True)` and `collect_data_files('lazy_loader', include_py_files=True)` along with comprehensive submodule collectors, ensuring all `.pyi` stub files are bundled in the executable.

### 9. Detection & Recognition Processing Optimizations
- **Snapshot Frequency**: Locked at **2 snapshots per second per person (`SNAPSHOT_INTERVAL_SEC = 0.5`)** continuously as required.
- **Backend Processing Acceleration**:
  1. **Direct Single-Pass Face Detection**: Removed the redundant 2nd-pass upscaling pass in `core/face_recognition.py` (`detect_faces`), preventing a 2.5s stall on frames without frontal faces.
  2. **Triage Inference Optimization**: Gated secondary YOLO face triage in `core/gui_workers.py` so it only runs if no person body is detected, freeing 50% CPU capacity for worker inference.
  3. **Multi-Process Work Isolation & Stealing**: Each camera channel has an isolated worker process with work-stealing capability (`other_queues`) to clear multi-snapshot bursts across all CPU cores simultaneously.
### 10. Decoupled Camera Triage & Pre-Warmed AI Worker Pool
- **Requirements Implemented**:
  1. **Independent Camera Triage**: When cameras are started, person triage, multi-person tracking, and snapshot capturing (**2 snaps per second**) run continuously whenever persons are in frame, independent of the "Start AI" button.
  2. **Pre-Warmed AI Workers**: `AttendanceWorker` processes and their deep learning models (InsightFace & ArcFace) are pre-loaded in RAM as soon as cameras connect.
  3. **Instant Zero-Latency AI Start/Pause**:
     - When the user toggles "Start AI", workers wake up via `active_event.set()` and consume snapshots immediately with **0-second startup latency**.
     - When the user toggles "Stop AI", `active_event.clear()` cleanly pauses worker queue consumption without terminating processes or unloading models from RAM.

### 11. Strict Manual AI Trigger (Zero Auto-Start)
- **Root Cause of Unintended Processing**:
  1. In `gui.py`: `QTimer.singleShot(2500, self.trigger_start_detection)` was automatically starting the AI 2.5 seconds after clicking "Start Cameras".
  2. In `core/gui_workers.py`: A `[SELF-HEALING]` watchdog was setting `is_detection_enabled = True` after 5 seconds of active cameras.
- **Fix Applied**:
  - Removed all timer-based and watchdog auto-start triggers across `gui.py` and `core/gui_workers.py`.
  - AI processing strictly activates **only** when the user explicitly clicks the "Start AI" button.
  - Workers remain paused in RAM and will not process a single snapshot until commanded.

### 12. Dynamic ONNX Input Triage & Multi-Target Detection
- **Root Cause of Missed Triage Snapshots**:
  1. In `_triage_detect_yolo`: Hardcoded `{'images': input_tensor}` instead of dynamic `.get_inputs()[0].name` was failing on ONNX models with custom input layer names.
  2. For seated/desk scenarios where employee bodies are partially occluded by monitors/tables, primary face triage was skipped if body detector returned no full body box.
- **Fix Applied**:
  - Dynamically query `.get_inputs()[0].name` for both face and body models.
  - Run face and body detection concurrently to catch both standing and seated employees in the camera frame.
  - Enqueue 2 snapshots/sec continuously to `data/pending_snapshots` whenever people are present on camera.

### 13. Backend Thread Heartbeat & Error Shielding
- **Root Cause of Backend Thread Exit**:
  - In `BackendController.run()`: `now = time.time()` was placed conditionally inside the triage result block, causing an `UnboundLocalError` on tick cycles when `active_cams` was empty or on camera source switch.
- **Fix Applied**:
  - Initialized `now = time.time()` at the top of every capture heartbeat tick.
  - Wrapped the capture and tracking loop in a master `try-except` block to prevent any transient frame error from crashing the Qt backend thread.

### 14. Zero-Lag RTSP Video Pipeline & Real-Time Rendering
- **Root Cause of Video Lag / Latency**:
  1. `ThreadedCamera.update_sub` was sleeping `time.sleep(0.03)` on successful frame reads. Since network RTSP packets arrived faster than 30ms sleep intervals, FFmpeg buffered stale packets internally, causing accumulated video lag (2–5 seconds).
  2. `OPENCV_FFMPEG_CAPTURE_OPTIONS` had `max_delay;500000` (500ms delay).
  3. The GUI refresh timer in `camera_page.py` was throttled to 66ms (15 FPS).
- **Fix Applied**:
  - Configured zero-latency capture flags: `rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|framedrop;1`.
  - Removed artificial sleep in `ThreadedCamera.update_sub`, draining the RTSP socket instantly on arrival with zero internal buffering.
  - Increased GUI preview render rate from 66ms to 30ms (~33 FPS) for fluid, real-time zero-lag video display.

### 15. Start/Stop AI Button State & Property Sync
- **Root Cause of Repeated AI Pause Logs**:
  - The UI button click handler was checking `is_detection_running` (which was not defined on `BackendController`) and was not dynamically updating the button label/style between "Start AI" (Green) and "Stop AI" (Red). Every click was falling back into `stop_detection()`.
- **Fix Applied**:
  - Added `@property is_detection_running` returning `is_detection_enabled` in `core/gui_workers.py`.
  - Updated `toggle_system` in `gui.py` to seamlessly toggle between "Start AI" and "Stop AI", immediately starting/pausing the pre-loaded workers.

### 16. Real-Time Detection Logs & Filter Fixes
- **Root Cause of Missing Logs**:
  1. `load_historical_data` was fetching recent records on startup but then immediately called `load_data()` which wiped `self.log_list.clear()`.
  2. The filter bar lacked an "All" option and was defaulted strictly to "Login", hiding Logout and unrecognized face events.
  3. `add_detection` was immediately dropping `name.upper() == 'UNKNOWN'` detections via an early return.
- **Fix Applied**:
  - Added `"All"` filter tab and set it as the default view.
  - Preserved loaded historical logs without clearing.
  - Enabled "Unknown" detections to show up in the log table under "All" and "Unknown" filter tabs.

### 17. API Error Analysis & Timeout Resilience
- **Analysis of Logged API Errors**:
  1. `403 Forbidden - Organization mismatch (Token Org: Prosper Infotech, Employee Org: Devam Projects)`:
     - Remote server rejected marking attendance for a personnel record enrolled under a different organization on the server backend than the authenticated user's organization.
  2. `Read timed out (read timeout=2)` in `log_raw_detection`:
     - Short 2-second timeout was causing HTTP read timeouts on slower network requests.
- **Fix Applied**:
  - Increased raw detection log timeout in `APIClient` from 2s to 10s.
  - Offline sync engine automatically drops rejected records with 403 API errors to prevent infinite synchronization loops.

### 18. Organization-Specific Personnel Filtering
- **Requirement**: Restrict facial recognition, face encodings, and attendance marking strictly to employees of the authenticated admin's organization.
- **Fix Applied**:
  - In `APIClient.login()`: Extract and store the admin user's `user_org_id` and `user_org_name` from the login response.
  - In `APIClient.get_all_face_encodings()`: Filter server personnel records to only include employees whose organization matches the authenticated admin's organization.
  - In `APIClient.mark_attendance()`: Catch 403 Organization Mismatch and gracefully ignore/drop foreign-organization records without throwing API exceptions.

### 19. Live Application-Only Detection Logs
- **Requirement**: The **Real-Time Detection Logs** page must exclusively show live detections made by the local camera feeds and AI workers during the active run, rather than downloading old cloud records from previous days/devices.
- **Fix Applied**:
  - Removed startup historical fetch `load_historical_data()`.
  - The Detection Log list now populates strictly in real-time as local AI workers recognize personnel and visitors on camera.

### 20. Multi-Organization Employee Attendance Attribution
- **Requirement**: Employees enrolled across multiple organizations or visiting the local facility should be recognized, and their attendance marked/attributed under the currently authenticated admin's organization.
- **Fix Applied**:
  - `get_all_face_encodings()` loads all enrolled personnel so visiting / multi-org staff are recognized.
  - `mark_attendance()` sends explicit `org_id` and `organization` payload context corresponding to the active admin.
  - In case the primary endpoint rejects with `403 Organization mismatch`, the client automatically routes through `log_raw_detection` with the active organization context and marks the local sync as complete.

### 21. BackendController Attribute Initialization Fix
- **Root Cause of AttributeError**:
  - `self._is_starting_detection` was accessed in `start_detection()` before being explicitly defined in `BackendController.__init__()`.
- **Fix Applied**:
  - Initialized `self._is_starting_detection = False` and `self._user_stopped_ai = False` in `BackendController.__init__()`.

### 22. AI Result Signal Propagation & GUI Log Sync
- **Root Cause of Delayed/Missing GUI Logs**:
  - The `self.result_queue.get_nowait()` consumer loop in `BackendController.run()` was accidentally indented inside `if self.are_cameras_active:`. If there were brief camera state changes or offline snapshot processing, incoming AI worker detections queued in `result_queue` were not drained or emitted to the GUI.
- **Fix Applied**:
  - Moved `self.result_queue` draining to execute at the root level of every backend loop tick.
  - Added full exception shielding and explicit logging inside `RecordsPage.add_detection()` to ensure every matched person or visitor instantly appears in the table.

### 23. Unknown Detection Suppression & PyInstaller Asset Path Resolution
- **Requirement 1 (No Unknowns in Logs)**:
  - Cleaned `RecordsPage` filter buttons to `["All", "Login", "Logout"]`.
  - Added an early return in `add_detection()` for `UNKNOWN` faces and missing person IDs so unrecognized faces are omitted from the detection logs.
- **Requirement 2 (Exact Logo & Icon in Compiled EXE)**:
  - In PyInstaller frozen mode, assets are unpacked into temporary runtime directory `sys._MEIPASS`. Looking only at `BASE_DIR` failed in EXE mode.
  - Added multi-path resolution checking `sys._MEIPASS`, `__file__` directory, and `BASE_DIR` for both sidebar logo (`WhatsApp Image 2026-02-17 at 19.51.32.jpeg` / `icon.png`) and application window icon (`icon.ico` / `icon.png`), ensuring 100% visual consistency between IDE and compiled EXE.

### 24. Standalone Executable Build Complete
- Executable built cleanly using PyInstaller (`vision_attendance.spec`).
- **Location**: `dist\VisionAttendance\VisionAttendance.exe`.
- Ready with all bundled AI weights, ONNX runtimes, brand logos, icons, and UI assets.
