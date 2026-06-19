import os
# Configure thread limits BEFORE any scientific library (numpy, opencv, onnxruntime) is imported
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_ARENA_EXTEND_STRATEGY"] = "kSameAsRequested"

# AGGRESSIVE RTSP TIMEOUT: 5 seconds (in microseconds)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000|stimeout;5000000"

import multiprocessing
import cv2
import time
import queue
import threading
import shutil
from datetime import datetime
import warnings

# Suppress InsightFace and other library specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")

import sys
import numpy as np

# Global lock to serialize ONNX and OpenCV model instantiation
# Prevents C++ thread deadlocks when workers are run as Threads instead of Processes
model_init_lock = threading.Lock()

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class AttendanceWorker(multiprocessing.Process):
    """
    Forensic AI Worker.
    Processes VIDEO FILES instead of live frames.
    Scans every single frame for maximum accuracy.
    """
    def __init__(self, task_queue, result_queue, worker_id, assigned_cam_index=0, api_user=None, api_pass=None, init_lock=None, event_type=None):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.assigned_cam_index = assigned_cam_index # [FIX] Only process own files
        self.api_user = api_user
        self.api_pass = api_pass
        self.init_lock = init_lock
        self.event_type = event_type # [NEW] Explicit Role (IN, OUT, or None)
        self.stop_event = multiprocessing.Event()
        self.daemon = True 
        
    def stop(self):
        self.stop_event.set()

    def _log_debug(self, msg):
        try:
            with open(f"worker_{self.worker_id}_debug.log", "a") as f:
                f.write(f"[{datetime.now()}] {msg}\n")
        except: pass
        print(f"[AI Worker {self.worker_id}] {msg}", flush=True)

    def _autosave_reid_cache(self):
        """Save the ReID gallery to disk if autosave interval has elapsed."""
        if self._reid_cache is None or self.reid is None:
            return
        now = time.time()
        if now - self._reid_last_save >= self.reid_autosave_interval:
            merged = self._reid_cache.save(self.reid.get_gallery())
            if merged:
                self.reid.load_gallery(merged)
            self._reid_last_save = now

    def run(self):
        self._log_debug("Initializing (Forensic Mode)...")
        try:
            from core.face_recognition import FaceRecognitionHandler
            from config.config import get_config
            from config import config as cfg_module # [NEW] For direct attr access
            
            # --- PHASE 1: Initialize DB & API (PARALLEL) ---
            # Increase staggering to prevent high-CPU and login storms
            startup_delay = self.worker_id * 3.0
            self._log_debug(f"Staggering start (Delay: {startup_delay}s)...")
            time.sleep(startup_delay)
            
            self._log_debug("Starting Network/DB Initialization...")
            config = get_config()
            
            use_api = config.get('use_api', True)
            if use_api:
                from core.api_client import APIClient
                self._log_debug(f"Connecting to Remote API: {config.get('api_base_url')}...")
                self.db = APIClient(config.get('api_base_url'), api_user=self.api_user, api_password=self.api_pass)
            else:
                self._log_debug("Connecting to Local Database directly...")
                from database.database import DatabaseManager
                self.db = DatabaseManager()
            
            # --- PHASE 2: Initialize ML Models (SERIALIZED) ---
            # Added more diagnostic logging for the serialized bottleneck
            lock = self.init_lock if self.init_lock else model_init_lock
            self._log_debug("Waiting for Init Lock (another worker is loading models)...")
            
            start_load = time.time()
            with lock:
                self._log_debug("Acquired Init Lock. Loading AI Engine...")
                # We use lazy_sync=True to keep the lock time minimal (model load only)
                self.face_handler = FaceRecognitionHandler(self.db, lazy_sync=True)
                
                # --- PERSON REID (Stage 3 fallback) ---
                self.reid = None
                self._reid_cache = None
                self._reid_last_save = time.time()
                self.reid_enabled = config.get('reid_enabled', True)
                self.reid_autosave_interval = config.get('reid_autosave_interval', 60)
                
                if self.reid_enabled:
                    try:
                        from core.person_reid import PersonReID
                        from core.reid_cache_manager import ReIDCacheManager
                        
                        self._reid_cache = ReIDCacheManager(config.get('reid_cache_dir'))
                        gallery = self._reid_cache.load_today()
                        
                        self.reid = PersonReID(
                            model_path=config.get('reid_model_path'),
                            similarity_threshold=config.get('reid_similarity_threshold', 0.72),
                            execution_providers=config.get('execution_providers', ["CPUExecutionProvider"]),
                            temporal_window=config.get('reid_temporal_window', 300.0),
                        )
                        self.reid.load_gallery(gallery)
                        self._log_debug("Person ReID enabled (OSNet).")
                    except Exception as e:
                        self._log_debug(f"ReID init failed: {e}. ReID disabled.")
                        self.reid = None
                
                load_duration = time.time() - start_load
                self._log_debug(f"AI Engine and ReID Loaded in {load_duration:.1f}s.")

            # --- PHASE 3: Sync Face Data (PARALLEL) ---
            # Now that we've left the lock, we can sync face encodings in parallel
            self._log_debug("Synchronizing Face Memory from source...")
            reg_count = self.face_handler.reload_face_encodings()
            self._log_debug(f"Face Handler Ready. Loaded {reg_count} registered faces.")
                
            # Released lock here. Thread is free to execute independently now.
            # [FIX] Small delay to ensure parent process is ready to receive from queue
            time.sleep(1)
            
            # Emit initial stats back to UI
            if self.result_queue:
                self.result_queue.put({
                    'type': 'stats',
                    'total': reg_count
                })
            config = get_config()
            self.snapshots_dir = config.get('snapshots_dir', 'data/attendance_snapshots')
            self.cooldown_seconds = config.get('attendance_cooldown', 30)
            
            # Ensure categorized snapshot directories exist
            for category in ['in', 'out', 'unknown']:
                cat_dir = os.path.join(self.snapshots_dir, category)
                if not os.path.exists(cat_dir):
                    os.makedirs(cat_dir, exist_ok=True)
                    self._log_debug(f"Created directory: {cat_dir}")
            
            # Persistent Cooldown Dictionary: {person_id: timestamp_of_last_log_or_snapshot}
            # This persists across multiple video files for the life of the worker
            self.cooldowns = {} 
            self.last_in_time = {} # [NEW] Loitering Protection 
            
            self._log_debug("Ready. Waiting for files...")
            
            # --- STARTUP RECOVERY: Process orphaned files ---
            self.recording_dir = config.get('temp_recordings_dir', 'data/temp_recordings')
            if not os.path.isabs(self.recording_dir):
                from config.config import BASE_DIR
                self.recording_dir = os.path.join(BASE_DIR, self.recording_dir)

            self.failed_dir = config.get('failed_recordings_dir', 'data/failed_recordings')
            if not os.path.isabs(self.failed_dir):
                from config.config import BASE_DIR
                self.failed_dir = os.path.join(BASE_DIR, self.failed_dir)

            os.makedirs(self.failed_dir, exist_ok=True)
            self.scavenged_files = self.scavenge_orphaned_files()
            
        except Exception as e:
            self._log_debug(f"CRITICAL INIT ERROR: {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            idle_counter = 0
            while not self.stop_event.is_set():
                try:
                    # Check scavenged files first, then wait for file path from queue
                    if self.scavenged_files:
                        task = self.scavenged_files.pop(0)
                        idle_counter = 0
                    else:
                        task = self.task_queue.get(timeout=1.0)
                        idle_counter = 0
                    
                    # [FIX] Handle commands instead of just filenames
                    if isinstance(task, dict):
                        t_type = task.get('type')
                        if t_type == 'reload':
                            self._log_debug("Received CLOUD SYNC signal. Refreshing face memory...")
                            self.face_handler.reload_face_encodings()
                            new_count = self.face_handler.get_registered_count()
                            self.result_queue.put({'type': 'stats', 'total': new_count})
                            continue
                        elif t_type == 'update_api':
                            new_url = task.get('url')
                            new_user = task.get('user')
                            new_pass = task.get('pass')
                            self._log_debug(f"Received API UPDATE signal. Switching to: {new_url}")
                            from core.api_client import APIClient
                            # Use self.db consistently so FaceRecognitionHandler stays in sync
                            self.db = APIClient(new_url, api_user=new_user, api_password=new_pass)
                            if hasattr(self, 'face_handler'):
                                self.face_handler.db_manager = self.db
                                self._log_debug("Syncing new API memory...")
                                self.face_handler.reload_face_encodings()
                            continue
                        elif t_type == 'update_db_mode':
                            self._log_debug("Received DB MODE UPDATE signal. Re-initializing database client...")
                            from config.config import get_config
                            config = get_config()
                            if config.get('use_api', True):
                                from core.api_client import APIClient
                                self.db = APIClient(config.get('api_base_url'), api_user=self.api_user, api_password=self.api_pass)
                            else:
                                from database.database import DatabaseManager
                                self.db = DatabaseManager()
                            
                            if hasattr(self, 'face_handler'):
                                self.face_handler.db_manager = self.db
                                self._log_debug("Syncing face memory with new backend...")
                                self.face_handler.reload_face_encodings()
                            continue
                        elif t_type == 'update_role':
                            self.event_type = task.get('event_type')
                            self._log_debug(f"Received UPDATE ROLE signal. Switching to: {self.event_type}")
                            continue
                    
                    video_path = task
                except queue.Empty:
                    idle_counter += 1
                    # Every 30 seconds of idleness, do a cleanup sweep
                    if idle_counter >= 30: 
                        self._log_debug("Idle sweep: Scanning for missed recordings...")
                        self.scavenged_files.extend(self.scavenge_orphaned_files())
                        
                        # If we have 0 faces, try to re-sync now
                        if self.face_handler.get_registered_count() == 0:
                             self._log_debug("Idle sweep: Attempting to reload face identities...")
                             self.face_handler.reload_face_encodings()
                        
                        idle_counter = 0
                    continue

                self._log_debug(f"Processing: {video_path}")
                try:
                    if video_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        success = self.process_snapshot(video_path)
                    else:
                        success = self.process_video_file(video_path)
                    
                    # If processing failed (e.g. DB error or temporary network drop), Re-Queue
                    if not success:
                        self._log_debug(f"Recovered error (Network/DB?). Re-queueing {os.path.basename(video_path)} in 10s...")
                        time.sleep(5) 
                        self.task_queue.put(video_path)
                        
                except Exception as e:
                    self._log_debug(f"CRITICAL ERROR processing {video_path}: {e}")
                    import traceback
                    traceback.print_exc()
        except KeyboardInterrupt:
            self._log_debug("Worker process interrupted by KeyboardInterrupt. Exiting...")
        finally:
            self._log_debug("Stopped.")
        
    def scavenge_orphaned_files(self):
        """Finds AVI files (or extension-less orphans) in recording dir and JPG files in pending_snapshots that were left over from a crash/restart"""
        recovered_files = []
        try:
            # 1. Scavenge pending snapshots
            pending_dir = "data/pending_snapshots"
            if os.path.exists(pending_dir):
                self._log_debug(f"Scanning for orphaned snapshots in {pending_dir}...")
                prefix = f'cam_{self.assigned_cam_index}_'
                files = [f for f in os.listdir(pending_dir) if f.startswith(prefix) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for f in files:
                    full_path = os.path.join(pending_dir, f)
                    recovered_files.append(full_path)
                self._log_debug(f"Recovered {len(files)} orphaned snapshot files.")

            # 2. Scavenge video recordings
            if os.path.exists(self.recording_dir):
                self._log_debug(f"Scanning for orphaned video files in {self.recording_dir}...")
                prefix = f'cam_{self.assigned_cam_index}_'
                extensions = ('.avi', '.mp4', '.mkv', '.dav', '.tmp')
                files = [f for f in os.listdir(self.recording_dir) 
                         if f.startswith(prefix) and (f.lower().endswith(extensions) or '.' not in f)]
                
                self._log_debug(f"Found orphaned cam_* files: {len(files)}")
                
                # Sort by timestamp (in filename) to process in order
                def parse_ts(fname):
                    import re
                    m = re.search(r'_(\d+)(?:\.|$)', fname)
                    return int(m.group(1)) if m else 0
                    
                files.sort(key=parse_ts)
                
                count = 0
                for f in files:
                    full_path = os.path.join(self.recording_dir, f)
                    
                    time_since_mod = time.time() - os.path.getmtime(full_path)
                    if time_since_mod < 5.0:
                        continue

                    if '.' not in f or f.endswith('.tmp'):
                        base_name = f.replace('.tmp', '')
                        new_name = base_name + '.avi'
                        new_path = os.path.join(self.recording_dir, new_name)
                        try:
                            if os.path.exists(new_path): os.remove(new_path)
                            os.rename(full_path, new_path)
                            self._log_debug(f"Renamed legacy orphan: {f} -> {new_name}")
                            full_path = new_path
                            f = new_name
                        except Exception as rename_err:
                            self._log_debug(f"Failed to rename {f}: {rename_err}")
                            continue
                    
                    try:
                        if self.db.is_video_processed(f):
                            self._log_debug(f"Cleaning up already processed orphan: {f}")
                            try: os.remove(full_path)
                            except: pass
                            continue
                    except: pass
                    
                    if time.time() - os.path.getmtime(full_path) < 5.0:
                        continue
                    
                    recovered_files.append(full_path)
                    count += 1
                    
                self._log_debug(f"Recovered {count} orphaned recording files.")
            
        except Exception as e:
            self._log_debug(f"Scavenger Error: {e}")
            
        return recovered_files

    def process_video_file(self, video_path):
        from config.config import get_config
        config = get_config()
        filename = os.path.basename(video_path)
        
        # 0. Deduplication Check
        try:
            if self.db.is_video_processed(filename):
                self._log_debug(f"Video {filename} ALREADY PROCESSED. Moving to processed storage.")
                try:
                    target_path = os.path.join(self.processed_dir, filename)
                    if os.path.exists(target_path): os.remove(target_path)
                    shutil.move(video_path, target_path)
                    self._log_debug(f"Moved duplicate: {filename}")
                except Exception as e:
                    self._log_debug(f"Error moving duplicate file {filename}: {e}")
                return True
        except Exception as db_err:
             self._log_debug(f"DB Check Failed: {db_err}. Will proceed cautiously.")
             # If we can't check DB, we should probably retry later?
             # But if we return False here, we loop.
             return False

        if not os.path.exists(video_path):
            print(f"[AI Worker {self.worker_id}] File not found: {video_path}")
            return True # Treat as success (handled)
            
        # [NEW] Check for empty/corrupt recordings immediately
        if os.path.getsize(video_path) < 100: # Less than 100 bytes is definitely corrupt
            self._log_debug(f"Corrupt/Empty recording found ({os.path.getsize(video_path)} bytes). Preserving in failed folder.")
            try:
                target_path = os.path.join(self.failed_dir, filename)
                if os.path.exists(target_path): os.remove(target_path)
                shutil.move(video_path, target_path)
            except Exception as e:
                self._log_debug(f"Could not move corrupt file (OS locked?): {e}")
            return True # Handled (Moved or Ignored safely)

        # [FIX] Enhanced Race condition prevention with Recorder: Wait for file to finalize
        cap = None
        # Increase retries to 10, total window ~10-12s to account for OS file locking
        for attempt in range(10):
            time.sleep(1.0) # Wait for OS/Disk to catch up
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                break
            self._log_debug(f"Attempt {attempt+1} - File NOT ready: {video_path}")
            if cap: cap.release()
            cap = None
            
        if cap is None:
            self._log_debug(f"CRITICAL: Could not open {video_path} after 10 attempts. File likely corrupt. Deleting.")
            try: os.remove(video_path)
            except: pass
            return True # Handled (Deleted to stop loop)

        # 1. Extract Start Timestamp from Filename
        # Format expected: cam_X_1678912345123.avi (Milliseconds)
        start_ts = time.time()  # Fallback
        try:
            # Look for the last sequence of digits before the extension
            import re
            match = re.search(r'_(\d+)\.', filename)
            if match:
                raw_ts = float(match.group(1))
                # Check if milliseconds (e.g., > 10^11) - typical epoch ms is 13 digits
                if raw_ts > 1e11:
                    start_ts = raw_ts / 1000.0
                else:
                    start_ts = raw_ts
        except Exception as e:
            self._log_debug(f"Timestamp parse error: {e}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0 # Default fallback
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_logged_percent = -1
        
        self._log_debug(f"Video opened: {filename} | FPS: {fps:.1f} | Total Frames: {total_frames} | Start TS: {start_ts}")

        frame_count = 0
        
        # [NEW] Tracking to skip redundant recognition
        # Format: { 'box': (x1, y1, x2, y2), 'identity': pid, 'last_seen': frame_count }
        tracked_faces = [] 
        
        start_proc_time = time.time()
        db_operation_failed = False
        
        # [NEW] Clip Summary: Accumulate unique detections for this video
        clip_detections = {} # {person_id: {data}}
        unknown_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: 
                self._log_debug(f"Reached end of video file at frame {frame_count}")
                break
            
            # [CRITICAL] Instant Abort Check: Drop everything if the system is stopping
            if self.stop_event.is_set():
                self._log_debug("STOP SIGNAL received during processing. Aborting file.")
                break
            
            frame_count += 1
            
            # Progress Logging (Every 10%)
            if total_frames > 0:
                percent = int((frame_count / total_frames) * 100)
                if percent % 10 == 0 and percent != last_logged_percent:
                    self._log_debug(f"Processing {filename}... [{percent}%]")
                    last_logged_percent = percent

            # Calculate Exact Capture Time
            capture_ts = start_ts + (frame_count / fps)
            
            # PERFORMANCE OPTIMIZATION: Frame Skipping
            # Increase skip rate significantly for CPU-bound forensic mode
            # Process one frame every ~0.1s of video (Improved for fast motion)
            skip_rate = int(fps * 0.1) if fps > 0 else 3
            if skip_rate < 1: skip_rate = 1
            if frame_count % skip_rate != 0:
                continue
                
            # Forensic Mode: Optimized Analysis
            try:
                # STAGE 1: Fast Triage (SSD)
                # ~5ms check to skip empty frames
                has_face = self.face_handler.quick_scan_faces(frame)
                
                # STAGE 2: Heavy Recognition (InsightFace)
                faces = []
                if has_face:
                    faces = self.face_handler.app.get(frame)
                
                if len(faces) > 0:
                    filtered_faces = []
                    for face in faces:
                        box = face.bbox.astype(int)
                        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                        
                        is_known = False
                        for tracked in tracked_faces:
                            tx1, ty1, tx2, ty2 = tracked['box']
                            if abs(cx - (tx1+tx2)//2) < (frame.shape[1] * 0.05) and \
                               abs(cy - (ty1+ty2)//2) < (frame.shape[0] * 0.05):
                                if frame_count - tracked['last_seen'] < (fps * 2): # Match within 2s (More reactive)
                                    is_known = True
                                    tracked['box'] = box
                                    tracked['last_seen'] = frame_count
                                    break
                        if not is_known:
                            filtered_faces.append(face)

                    if len(filtered_faces) > 0:
                        # --- V5 PIPELINE: recognize_multiple_faces handles mask detection internally ---
                        recognized = self.face_handler.recognize_multiple_faces(filtered_faces, frame)
                        
                        for person in recognized:
                            pid = person.get('person_id')
                            is_masked = person.get('is_masked', False)
                            rec_method = person.get('recognition_method')
                            box = person.get('bbox', [0,0,0,0])
                            
                            # Try ReID fallback if face is not recognized
                            if not pid or pid == 'UNKNOWN':
                                if self.reid is not None and self.reid.get_registered_count() > 0:
                                    reid_id, reid_name, reid_sim = self.reid.identify(frame, box, timestamp=capture_ts)
                                    if reid_id:
                                        person['person_id'] = reid_id
                                        person['person_name'] = reid_name
                                        person['similarity'] = reid_sim
                                        person['recognition_method'] = 'body_reid'
                                        pid = reid_id
                                        self._log_debug(f"[ReID] Face unrecognized, but ReID body match: {reid_name} ({reid_id}) | Sim: {reid_sim:.2f}")
                                        
                            # If unknown, assign a unique temporary ID for this clip summary
                            if not pid or pid == 'UNKNOWN':
                                unknown_count += 1
                                pid = f"UNKNOWN_{unknown_count}"
                            else:
                                # Cache body embedding for future ReID (only if recognized by face, not by ReID fallback)
                                if self.reid is not None and person.get('recognition_method') != 'body_reid':
                                    self.reid.register_body(pid, person.get('person_name'), frame, box, timestamp=capture_ts)
                                    self._autosave_reid_cache()
                            
                            # Log mask-aware recognition events
                            if is_masked and rec_method == 'upper_face' and pid and not pid.startswith('UNKNOWN'):
                                self._log_debug(f"[MASK] Recognized {person.get('person_name')} via upper-face (Sim: {person.get('similarity', 0):.2f})")
                            
                            # Update tracker
                            tracked_faces.append({'box': box, 'identity': pid, 'last_seen': frame_count})
                            
                            # [REFINED] Use explicit role if provided, otherwise fallback to filename match
                            event_type = self.event_type.lower() if self.event_type else 'monitor'
                            if not self.event_type:
                                try:
                                    from config.cam_config_manager import CamConfigManager
                                    conf = CamConfigManager.load_config()
                                    cams_list = conf.get('cams', [])
                                    if self.assigned_cam_index < len(cams_list):
                                        role = cams_list[self.assigned_cam_index].get('role', 'monitor').lower()
                                        event_type = 'in' if role == 'entrance' else ('out' if role == 'exit' else 'monitor')
                                    else:
                                        event_type = 'monitor'
                                except Exception:
                                    if 'cam_0' in filename: event_type = 'in'
                                    elif 'cam_1' in filename: event_type = 'out'
                                    else: event_type = 'in' if self.assigned_cam_index % 2 == 0 else 'out'
                            
                            # Pass DB failure status back
                            status, snapshot_path = self.handle_attendance(person, frame, capture_ts, event_type)
                            if status is False:
                                db_operation_failed = True
                            
                            if pid not in clip_detections:
                                clip_detections[pid] = {
                                    'type': 'match',
                                    'event_type': event_type.upper(),
                                    'id': pid,
                                    'name': person.get('person_name') or 'UNKNOWN',
                                    'sim': person.get('similarity', 0.0),
                                    'timestamp': datetime.fromtimestamp(capture_ts).strftime('%H:%M:%S'),
                                    'worker': self.worker_id,
                                    'snapshot_path': snapshot_path,
                                    'is_masked': is_masked,
                                    'recognition_method': rec_method
                                }
                else:
                    # Stage 3: Body ReID fallback (for back-turned/occluded people)
                    if self.reid is not None and self.reid.get_registered_count() > 0:
                        bodies = self.reid.detect_bodies(frame)
                        for body_box in bodies:
                            reid_id, reid_name, reid_sim = self.reid.identify(frame, body_box, timestamp=capture_ts)
                            if reid_id:
                                # Centroid check to avoid spamming the tracker
                                bcx, bcy = (body_box[0] + body_box[2]) // 2, (body_box[1] + body_box[3]) // 2
                                is_tracked = False
                                for tracked in tracked_faces:
                                    tx1, ty1, tx2, ty2 = tracked['box']
                                    if abs(bcx - (tx1+tx2)//2) < (frame.shape[1] * 0.15) and \
                                       abs(bcy - (ty1+ty2)//2) < (frame.shape[0] * 0.15):
                                        if frame_count - tracked['last_seen'] < (fps * 2):
                                            is_tracked = True
                                            tracked['box'] = body_box
                                            tracked['last_seen'] = frame_count
                                            break
                                            
                                if not is_tracked:
                                    person = {
                                        'person_id': reid_id,
                                        'person_name': reid_name,
                                        'similarity': reid_sim,
                                        'bbox': body_box,
                                        'recognition_method': 'body_reid',
                                        'is_masked': False
                                    }
                                    self._log_debug(f"[ReID] No face, but body matched: {reid_name} ({reid_id}) | Sim: {reid_sim:.2f}")
                                    tracked_faces.append({'box': body_box, 'identity': reid_id, 'last_seen': frame_count})
                                    
                                    event_type = self.event_type.lower() if self.event_type else 'monitor'
                                    if not self.event_type:
                                        try:
                                            from config.cam_config_manager import CamConfigManager
                                            conf = CamConfigManager.load_config()
                                            cams_list = conf.get('cams', [])
                                            if self.assigned_cam_index < len(cams_list):
                                                role = cams_list[self.assigned_cam_index].get('role', 'monitor').lower()
                                                event_type = 'in' if role == 'entrance' else ('out' if role == 'exit' else 'monitor')
                                            else:
                                                event_type = 'monitor'
                                        except Exception:
                                            if 'cam_0' in filename: event_type = 'in'
                                            elif 'cam_1' in filename: event_type = 'out'
                                            else: event_type = 'in' if self.assigned_cam_index % 2 == 0 else 'out'
                                            
                                    status, snapshot_path = self.handle_attendance(person, frame, capture_ts, event_type)
                                    if status is False:
                                        db_operation_failed = True
                                        
                                    if reid_id not in clip_detections:
                                        clip_detections[reid_id] = {
                                            'type': 'match',
                                            'event_type': event_type.upper(),
                                            'id': reid_id,
                                            'name': reid_name,
                                            'sim': reid_sim,
                                            'timestamp': datetime.fromtimestamp(capture_ts).strftime('%H:%M:%S'),
                                            'worker': self.worker_id,
                                            'snapshot_path': snapshot_path,
                                            'is_masked': False,
                                            'recognition_method': 'body_reid'
                                        }
                
            except Exception as e:
                import traceback
                self._log_debug(f"Frame {frame_count} Error: {e}\n{traceback.format_exc()}")

        cap.release()
        del cap # Force release
        
        # [NEW] Emit Summarized results to UI after CLIP completion
        if self.result_queue and clip_detections:
            self._log_debug(f"Emitting summary for clip: {len(clip_detections)} unique persons found.")
            for pid, packet in clip_detections.items():
                self.result_queue.put(packet)

        # Final DB update
        try:
            filename = os.path.basename(video_path)
            self.db.mark_video_as_processed(filename)
        except: pass
        
        if db_operation_failed:
            self._log_debug("DB Write Failed during processing. Preserving file for retry.")
            return False # Signal RETRY
        
        end_proc_time = time.time()
        duration = end_proc_time - start_proc_time
        
        # Cleanup Phase: DELETE processed file to save disk space
        for attempt in range(3):
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    
                self._log_debug(f"Finished & Deleted: {filename}")
                self._log_debug(f"   ↳ Processed {frame_count} frames in {duration:.2f}s ({frame_count/duration if duration > 0 else 0:.1f} FPS)")
                break
            except Exception as e:
                self._log_debug(f"Delete Attempt {attempt+1} Failed: {e}")
                time.sleep(1)
            
        return True

    def process_snapshot(self, filepath):
        if not os.path.exists(filepath):
            return True
            
        filename = os.path.basename(filepath)
        self._log_debug(f"Processing snapshot: {filename}")
        
        # 1. Parse timestamp from filename
        # Expected format: cam_{cam_index}_{timestamp}.jpg
        capture_ts = time.time()
        try:
            import re
            m = re.search(r'_(\d+)\.', filename)
            if m:
                raw_ts = float(m.group(1))
                if raw_ts > 1e11:
                    capture_ts = raw_ts / 1000.0
                else:
                    capture_ts = raw_ts
        except Exception as e:
            self._log_debug(f"Snapshot timestamp parse error: {e}")
            
        # 2. Read image
        frame = cv2.imread(filepath)
        if frame is None:
            self._log_debug(f"CRITICAL: Failed to read image: {filepath}")
            try: os.remove(filepath)
            except: pass
            return True
            
        db_operation_failed = False
        
        try:
            # Resize image for faster CPU inference
            max_dim = 640
            h, w = frame.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                resized_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                resized_frame = frame
                scale = 1.0

            # Stage 2: Heavy Recognition (Direct InsightFace — V5 pipeline, bypass detect_faces)
            faces = self.face_handler.app.get(resized_frame)
            
            # Scale coordinates back
            if scale != 1.0:
                for face in faces:
                    face.bbox = face.bbox / scale
                    if hasattr(face, 'kps') and face.kps is not None:
                        face.kps = face.kps / scale
            
            if len(faces) > 0:
                # --- V5 PIPELINE: recognize_multiple_faces handles mask detection internally ---
                recognized = self.face_handler.recognize_multiple_faces(faces, frame)
                
                for person in recognized:
                    pid = person.get('person_id')
                    sim = person.get('similarity', 0.0)
                    box = person.get('bbox', [0,0,0,0])
                    is_masked = person.get('is_masked', False)
                    rec_method = person.get('recognition_method')
                    
                    # Try ReID fallback if face is not recognized
                    if not pid or pid == 'UNKNOWN':
                        if self.reid is not None and self.reid.get_registered_count() > 0:
                            reid_id, reid_name, reid_sim = self.reid.identify(frame, box, timestamp=capture_ts)
                            if reid_id:
                                person['person_id'] = reid_id
                                person['person_name'] = reid_name
                                person['similarity'] = reid_sim
                                person['recognition_method'] = 'body_reid'
                                pid = reid_id
                                sim = reid_sim
                                rec_method = 'body_reid'
                                self._log_debug(f"[ReID] Face unrecognized, but ReID body match: {reid_name} ({reid_id}) | Sim: {reid_sim:.2f}")
                                
                    if pid and not pid.startswith('UNKNOWN'):
                        if self.reid is not None and person.get('recognition_method') != 'body_reid':
                            self.reid.register_body(pid, person.get('person_name'), frame, box, timestamp=capture_ts)
                            self._autosave_reid_cache()
                    
                    # Update DB and generate snapshot
                    success, snapshot_path = self.handle_attendance(person, frame, capture_ts, self.event_type.lower() if self.event_type else 'monitor')
                    if success is False:
                        db_operation_failed = True
                        
                    if self.result_queue:
                        name = person.get('person_name') or 'UNKNOWN'
                        self.result_queue.put({
                            'type': 'match',
                            'event_type': self.event_type.upper() if self.event_type else 'MONITOR',
                            'id': pid or 'UNKNOWN',
                            'name': name,
                            'sim': sim,
                            'timestamp': datetime.fromtimestamp(capture_ts).strftime('%H:%M:%S'),
                            'worker': self.worker_id,
                            'bbox': box.tolist() if hasattr(box, 'tolist') else list(box),
                            'snapshot_path': snapshot_path,
                            'is_masked': is_masked,
                            'recognition_method': rec_method
                        })
            else:
                self._log_debug(f"Snapshot {filename} contained no faces. Checking for bodies...")
                if self.reid is not None and self.reid.get_registered_count() > 0:
                    bodies = self.reid.detect_bodies(frame)
                    self._log_debug(f"Detected {len(bodies)} bodies in snapshot.")
                    for body_box in bodies:
                        reid_id, reid_name, reid_sim = self.reid.identify(frame, body_box, timestamp=capture_ts)
                        if reid_id:
                            person = {
                                'person_id': reid_id,
                                'person_name': reid_name,
                                'similarity': reid_sim,
                                'bbox': body_box,
                                'recognition_method': 'body_reid',
                                'is_masked': False
                            }
                            self._log_debug(f"[ReID] Snapshot body matched: {reid_name} ({reid_id}) | Sim: {reid_sim:.2f}")
                            
                            success, snapshot_path = self.handle_attendance(person, frame, capture_ts, self.event_type.lower() if self.event_type else 'monitor')
                            if success is False:
                                db_operation_failed = True
                                
                            if self.result_queue:
                                self.result_queue.put({
                                    'type': 'match',
                                    'event_type': self.event_type.upper() if self.event_type else 'MONITOR',
                                    'id': reid_id,
                                    'name': reid_name,
                                    'sim': reid_sim,
                                    'timestamp': datetime.fromtimestamp(capture_ts).strftime('%H:%M:%S'),
                                    'worker': self.worker_id,
                                    'bbox': body_box if isinstance(body_box, list) else list(body_box),
                                    'snapshot_path': snapshot_path,
                                    'is_masked': False,
                                    'recognition_method': 'body_reid'
                                })
                else:
                    self._log_debug(f"Snapshot {filename} contained no faces (Triage false positive) and ReID is inactive/empty.")
        except Exception as e:
            self._log_debug(f"Error processing snapshot: {e}")
            import traceback
            self._log_debug(traceback.format_exc())
            
        if db_operation_failed:
            # Network or database is down. Keep file in pending_snapshots so it can be retried!
            self._log_debug(f"DB/Network failure. Keeping snapshot in queue for retry: {filename}")
            return False
            
        # Clean up / delete the pending snapshot file on success
        try:
            os.remove(filepath)
        except Exception as e:
            self._log_debug(f"Failed to remove pending snapshot {filename}: {e}")
            
        return True

    def handle_attendance(self, person, frame, capture_ts, event_type='in'):
        pid = person['person_id']
        name = person['person_name']
        sim = person['similarity']
        
        if not pid: 
            self._log_debug(f"Face detected but Unknown (Sim: {sim:.2f})")
            # Don't return early! Proceed to snapshot saving section.
        else:
            self._log_debug(f"Recognized: {name} ({pid}) | Sim: {sim:.2f}")
        
        # Use Capture Time for Cooldown Check
        last_marked = self.cooldowns.get(pid, 0)
        
        # COOLDOWN CHECK: Enforce cooldown based on EVENT TIME
        if capture_ts - last_marked < self.cooldown_seconds:
            # Skip DB hit and snapshot if within cooldown
            return True, None

        # Immediate Database Attempt
        try:
            # Convert timestamp to datetime for display/logging
            capture_dt = datetime.fromtimestamp(capture_ts)
            
            # --- LOITERING PROTECTION [NEW] ---
            # Update IN time
            if event_type == 'in' and pid:
                self.last_in_time[pid] = capture_ts
            
            # Suppress OUT if too soon after IN
            if event_type == 'out' and pid:
                 from config import config as cfg_module # [FIX] Import here for access
                 last_in = self.last_in_time.get(pid, 0)
                 threshold = getattr(cfg_module, 'LOITERING_THRESHOLD', 60)
                 if (capture_ts - last_in) < threshold:
                     self._log_debug(f"Loitering suppressed for {name} (Time since IN: {capture_ts - last_in:.1f}s)")
                     return True, None # Skip Attendance Update

            # 1. Log Raw Event (Ensures 'logs' table is populated)
            # This is critical for API mode where mark_attendance doesn't auto-log
            if pid:
                self.db.log_raw_detection(pid, name, timestamp=capture_dt, event_type=event_type)

            # 2. Update DB with ACTUAL CAPTURE TIME and Event Type
            # In V1 (API Mode), self.db IS the API Client, so this Call sends the request directly.
            success, msg = (True, "OK")
            if pid:
                success, msg = self.db.mark_attendance(pid, timestamp=capture_dt, event_type=event_type)
            
            if not success and pid:
                self._log_debug(f"Mark Attendance Failed: {msg}")
                # For API mode, we might want to retry? But let's just log failure for now.
                return False, None # Signal Failure
            
            # 2. Update Cooldown with ACTUAL CAPTURE TIME
            if pid:
                self.cooldowns[pid] = capture_ts
            
            # 3. Save Snapshot in Categorized Folders
            from config.config import get_config
            config = get_config()
            threshold = config.get('similarity_threshold', 0.4)
            
            snapshot_path = None
            if sim >= threshold:
                category = event_type.lower() # 'in' or 'out'
                if not pid or pid == 'UNKNOWN':
                    category = 'unknown'
                    
                target_dir = os.path.join(self.snapshots_dir, category)
                os.makedirs(target_dir, exist_ok=True)
                filename = f"{pid}_{int(capture_ts)}.jpg"
                snapshot_path = os.path.join(target_dir, filename)
                
                # [FIX] Robust write with error checking
                success = cv2.imwrite(snapshot_path, frame)
                if success:
                    self._log_debug(f"Saved {category.upper()} snapshot: {filename}")
                else:
                    self._log_debug(f"CRITICAL: Failed to save snapshot to {snapshot_path}. Check permissions/disk space.")

                # AUDIBLE ALERT: Beep twice
                try:
                    try:
                        import winsound
                        winsound.Beep(1000, 200) # 1000Hz, 200ms
                        time.sleep(0.2)
                        winsound.Beep(1000, 200)
                    except ImportError:
                        # Linux/Mac Fallback
                        print('\a')
                        time.sleep(0.2)
                        print('\a')
                except: pass
                
            return True, snapshot_path
                
        except Exception as e:
            # Duplicate entry or DB connection error
            self._log_debug(f"DB Entry Failed: {e}")
            return False, None
