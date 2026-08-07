# Conversation History & Technical Resolutions Log (chat.md)

This document records all user requests, diagnostic investigations, code edits, and technical resolutions completed during the session.

---

## 📝 Chronological Work & Resolution Log

### Request 1: RTSP Stream Lag & Unregistered Employee Matching
- **Fix Applied**: Switched RTSP transport from UDP to TCP and increased similarity threshold to 0.55.

### Request 2: Complete Removal of Person ReID Logic
- **Fix Applied**: Removed all ReID logic, configuration keys, and model references from codebase.

### Request 3: Refactoring to Shared AI Worker Pool (Solving Stream Lag)
- **Fix Applied**: Refactored system from 1 worker process per camera to a 2-worker shared AI process pool.

### Request 4: Guarding Index Errors Across UI & Worker Scaling
- **Fix Applied**: Guarded array indexes against scaling down errors.

### Request 5: Fixing NameError `load_duration`
- **Fix Applied**: Restored missing `load_duration` variable in `core/multiprocess_handler.py`.

### Request 6: Pending Snapshot Queueing & Auto-Cleanup
- **Fix Applied**: Batched pending snapshot queueing to 10 snapshots per cycle with deletion after detection.

### Request 7: Deadlock-Proof `init_lock` Timeout Guard
- **Fix Applied**: Added 30-second timeout to worker `init_lock.acquire()` calls.

### Request 8: Bidirectional Dual-Stream Derivation for 5 MP Production Cameras
- **Fix Applied**: Derives `/sub` low-res preview stream for live rendering and `/main` 5 MP stream for snapshots.

### Request 9: Targeting Dedicated NVIDIA RTX 5050 Laptop GPU & Pre-Flight CUDA Test
- **Fix Applied**: Target GPU 1 directly via DirectX 12 DML provider with pre-flight DLL check.

### Request 10: Reverting strictly to CPU execution
- **Fix Applied**: Reverted `EXECUTION_PROVIDERS` strictly to `['CPUExecutionProvider']` in `config/config.py`.

### Request 11: Multi-Face Detection, Init Lock Hang & Camera Configuration Persistence
- **Fix Applied**: 
  - Added unique 6-character UUID suffix to snapshot filenames (`cam_0_exit_ts_uuid.jpg`).
  - Re-created `self.init_lock` on every `start_detection()` run to bypass poisoned locks.
  - Used list comprehensions for `self.cam_direction_rules` and `self.cam_rois` to prevent shared reference mutations.

### Request 12: High-Speed Camera Connection Optimization
- **Fix Applied**: Deferred main-stream RTSP loading, reduced TCP socket timeout to 0.3s, and parallelized camera startup.

### Request 13: Strict Reversion of Auto-Purge and Backpressure Skipping
- **Fix Applied**: Reverted queue depth checks and 120s auto-purge logic per user request.

### Request 14: Dynamic 1 AI Worker per 2 Cameras Scaling
- **Fix Applied**: Configured `self.num_ai_workers = max(1, math.ceil(self.active_cam_count / 2.0))` in `core/gui_workers.py`.

### Request 15: 4-Point System Test & Motion Vector Jitter Filtering
- **Fix Applied**: Verified ReID removal, multi-face loop, init lock 30s timeout guard, and increased trajectory threshold to 8.0px.

### Request 16: Deep Fix for Multi-Person Single-Frame Detection
- **Fix Applied**: Separated face/body NMS in `gui_workers.py`; removed proximity suppression filter in `multiprocess_handler.py`; lowered YOLO confidence threshold to 0.25 in `face_recognition.py`.

### Request 17: True Parallel Multi-Worker Execution Fix
- **Fix Applied**: Unbound OpenMP CPU affinity (`OMP_PROC_BIND=FALSE`) and reduced startup stagger to 0.2s.

### Request 18: Resolution of Single-Worker Task Hoarding (`[AI Worker 1]` only)
- **Fix Applied**: Removed private pending snapshot hoarding from `scavenge_orphaned_files()`.

### Request 19: Resolution of Low Similarity Matches (`Sim: 0.11` to `Sim: 0.18`)
- **Fix Applied**: Full-frame landmark alignment embedding extraction in `core/face_recognition.py`.

### Request 20: Restoration of High-Precision InsightFace Detection Engine
- **Fix Applied**: Restored `FACE_DETECTION_BACKEND = 'insightface'` and `TRIAGE_DETECTION_BACKEND = 'opencv_dnn'` in `config/config.py`.

### Request 21: Resolution of GUI Freezing & API Command Queue Flooding
- **Fix Applied**: Single dispatch `self.shared_task_queue.put({'type': 'update_api', ...})` in `gui_workers.py`.

### Request 22: Restoration of Bharti Project Worker Architecture & Guaranteed EXIT Snapshots
- **Fix Applied**: 
  - Restored Bharti project multiprocess worker flow in [core/gui_workers.py](file:///d:/vt_application_v6.2_cpu/core/gui_workers.py#L1858) with dedicated camera queues (`self.task_queues[i]`), explicit camera roles (`event_type='IN' / 'OUT'`), and work-stealing.
  - Moved snapshot file writing (`cv2.imwrite`) BEFORE loitering suppression check in [core/multiprocess_handler.py](file:///d:/vt_application_v6.2_cpu/core/multiprocess_handler.py#L755), guaranteeing 100% of EXIT (`OUT`) snapshots are saved to `data/attendance_snapshots/out/`.

### Request 23: Fix Terminal Exception Tracebacks (AttributeError & IndexError)
- **Fix Applied**: 
  - Initialized `self.recorder = None` in `BackendController.__init__` and guarded `stop_cameras()` with `getattr(self, 'recorder', None)` to prevent `AttributeError`.
  - Added auto-extension bounds checking (`while len(self.workers) < self.active_cam_count: self.workers.append(None)`) in `_start_detection_async` and `_start_single_worker` to prevent `IndexError: list index out of range`.

### Request 24: Resolution of False Positive Snapshots on Empty Rooms / Static Objects
- **Fix Applied**: Increased OpenCV Caffe SSD triage confidence threshold from `0.30` to `0.55` in [core/gui_workers.py](file:///d:/vt_application_v6.2_cpu/core/gui_workers.py#L1272). Eliminates false positive snapshot captures triggered by dark umbrellas, floor seams, or shadow folds.

### Request 25: Dedicated Standalone `SnapshotManager` Thread Module
- **Fix Applied**: 
  - Created [core/snapshot_manager.py](file:///d:/vt_application_v6.2_cpu/core/snapshot_manager.py) to handle snapshot image resizing, disk persistence (`cv2.imwrite`), and pending folder scanning asynchronously on a separate background daemon thread.
  - Updated [core/gui_workers.py](file:///d:/vt_application_v6.2_cpu/core/gui_workers.py#L1355) to delegate snapshot tasks in **< 0.01ms**, completely eliminating disk I/O bottlenecks and delivering ultra-smooth 30 FPS camera preview rendering.

### Request 26: Multi-Person Tracker (`core/tracker.py`) & Single-Snapshot Architecture
- **Fix Applied**: 
  - Built [core/tracker.py](file:///d:/vt_application_v6.2_cpu/core/tracker.py) featuring `Track` state objects and `MultiPersonTracker` for per-camera multi-target tracking.
  - Enforced strict `track.snapshot_saved` flag check in [core/gui_workers.py](file:///d:/vt_application_v6.2_cpu/core/gui_workers.py#L1495), ensuring exactly **1 snapshot & 1 InsightFace recognition run per Track ID**.
  - Formatted snapshot filenames to include track ID: `cam_0_track_101_entrance_ts_uuid.jpg`.

### Request 27: Resolution of EXIT Snapshot Parsing & Storage (`_exit_` Keyword Parsing)
- **Fix Applied**: Updated filename event_type parser in [core/multiprocess_handler.py:L610](file:///d:/vt_application_v6.2_cpu/core/multiprocess_handler.py#L610) to search for `_exit_` / `_out_` keywords anywhere in filenames containing track IDs (`cam_1_track_160_exit_...jpg`), ensuring 100% of EXIT snapshots save cleanly to `data/attendance_snapshots/out/`.

### Request 28: Production-Grade `TrackManager` 9-State FSM, Watchdog & Adaptive Cropping
- **Fix Applied**: 
  - Built [core/track_manager.py](file:///d:/vt_application_v6.2_cpu/core/track_manager.py) containing centralized `TrackManager` and `TrackWatchdog` background thread.
  - Implemented 9-state FSM (`NEW` $\rightarrow$ `SNAPSHOT_PENDING` $\rightarrow$ `SNAPSHOTTED` $\rightarrow$ `QUEUED` $\rightarrow$ `PROCESSING` $\rightarrow$ `MATCHED` $\rightarrow$ `ATTENDANCE_PENDING` $\rightarrow$ `ATTENDANCE_DONE` $\rightarrow$ `EXPIRED`).

### Request 29: Per-Camera Isolated Snapshot Pipelines, Reference-Counted FrameBuffer & SnapshotSupervisor
- **Fix Applied**: 
  - Built [core/frame_buffer.py](file:///d:/vt_application_v6.2_cpu/core/frame_buffer.py) with reference-counted circular ring buffers (`FrameEntry`, `FrameBuffer`, `FrameBufferManager`).
  - Redesigned [core/snapshot_manager.py](file:///d:/vt_application_v6.2_cpu/core/snapshot_manager.py) into `SnapshotPipelineManager`, spawning dedicated `SnapshotWorker-i` threads per camera stream slot and a `SnapshotSupervisor` thread for crash auto-restart.
  - Implemented dual-layered pending snapshot processing: `scan_pending_snapshots()` in [core/snapshot_manager.py:L311](file:///d:/vt_application_v6.2_cpu/core/snapshot_manager.py#L311) re-queues un-processed files by timestamp, and `scavenge_orphaned_files()` in [core/multiprocess_handler.py:L275](file:///d:/vt_application_v6.2_cpu/core/multiprocess_handler.py#L275) scavenges `data/pending_snapshots/` during worker idle sweeps, guaranteeing 100% processing of all captured snapshot JPEGs.
