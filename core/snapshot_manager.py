import os
import cv2
import time
import uuid
import queue
import threading
from datetime import datetime
from config.config import (
    BASE_DIR,
    SNAPSHOT_PADDING,
    SNAPSHOT_QUEUE_TIMEOUT,
    MAX_RETRY
)
from core.frame_buffer import FrameBufferManager
from core.track_manager import TrackManager

class SnapshotTask:
    """
    Snapshot Task Payload.
    Carries the frame matrix directly to avoid FrameBuffer timing issues.
    """
    def __init__(self, track_id, camera_id, frame_id, bounding_box, direction="monitor", frame_timestamp=None, frame_matrix=None):
        self.task_id = uuid.uuid4().hex
        self.track_id = track_id
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.frame_matrix = frame_matrix  # Direct frame reference — no buffer lookup needed
        self.bounding_box = list(bounding_box) if bounding_box else []
        self.direction = direction
        self.frame_timestamp = frame_timestamp or time.time()
        self.enqueue_timestamp = time.time()
        self.retry_count = 0
        self.snapshot_uuid = uuid.uuid4().hex[:6]


class SnapshotWorker(threading.Thread):
    """
    Dedicated Single Snapshot Worker Thread per camera stream slot.
    Operates in complete isolation: Camera 1 processing NEVER blocks Camera 2.
    """
    def __init__(self, camera_id, task_queue, ai_task_queues, logger=None):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.task_queue = task_queue
        self.ai_task_queues = ai_task_queues
        self.logger = logger
        
        self.is_running = True
        self.state = "IDLE"
        self.last_active_time = time.time()
        self.pending_dir = os.path.join(BASE_DIR, "data", "pending_snapshots")
        os.makedirs(self.pending_dir, exist_ok=True)

    def log(self, msg):
        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp_str}] [SnapshotWorker-{self.camera_id}] {msg}"
        print(formatted, flush=True)
        if self.logger:
            try: self.logger(formatted)
            except: pass

    def crop_person_fixed(self, frame, bbox):
        """Fixed 35% bounding box padding crop for optimal face detection & alignment."""
        if frame is None or frame.size == 0 or not bbox or len(bbox) < 4:
            return frame
            
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = bbox
        bw = bx2 - bx1
        bh = by2 - by1
        
        # Generous 35% Padding to ensure face and head are fully captured
        pad_w = int(bw * 0.35)
        pad_h = int(bh * 0.35)
        
        x1 = max(0, int(bx1 - pad_w))
        y1 = max(0, int(by1 - pad_h))
        x2 = min(w, int(bx2 + pad_w))
        y2 = min(h, int(by2 + pad_h))
        
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else frame

    def run(self):
        self.log(f"SnapshotWorker dedicated to Camera {self.camera_id} ONLINE.")
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1.0)
            except queue.Empty:
                self.state = "IDLE"
                continue

            self.state = "BUSY"
            self.last_active_time = time.time()
            tm = TrackManager()

            try:
                # 1. Use frame matrix passed directly in SnapshotTask (most reliable)
                import numpy as np
                frame_matrix = task.frame_matrix
                entry = None

                # Fallback: try FrameBuffer if task.frame_matrix not set
                if frame_matrix is None or (hasattr(frame_matrix, 'size') and frame_matrix.size == 0):
                    self.log(f"[SnapshotWorker] task.frame_matrix not set for Track {task.track_id}, falling back to FrameBuffer...")
                    fbm = FrameBufferManager()
                    entry = fbm.get_frame(task.camera_id, task.frame_id)
                    if entry:
                        entry.add_ref()
                        frame_matrix = entry.frame_matrix
                    else:
                        buf = fbm.get_buffer(task.camera_id)
                        with buf.lock:
                            if buf.ordered_ids:
                                latest_id = buf.ordered_ids[-1]
                                entry = buf.frames.get(latest_id)
                                if entry:
                                    entry.add_ref()
                                    frame_matrix = entry.frame_matrix

                if frame_matrix is None or (hasattr(frame_matrix, 'size') and frame_matrix.size == 0):
                    self.log(f"CRITICAL: No frame available for Track {task.track_id} on Cam {task.camera_id}. Skipping.")
                    tm.notify_snapshot_failed(task.camera_id, task.track_id, error="Missing frame")
                    self.task_queue.task_done()
                    continue

                # 2. Save WHOLE IMAGE (full frame matrix) per user directive — NO ROI cropping!
                ts = int(task.frame_timestamp * 1000)
                filename = f"cam_{task.camera_id}_track_{task.track_id}_{task.direction}_{ts}_{task.snapshot_uuid}.jpg"
                filepath = os.path.join(self.pending_dir, filename)

                success = cv2.imwrite(filepath, frame_matrix)
                
                # Release FrameEntry reference count
                if entry:
                    entry.release_ref()

                if success:
                    crop_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
                    print(f"\n=======================================================", flush=True)
                    print(f"[SNAPSHOT CAPTURED] Camera {task.camera_id + 1} | Track {task.track_id} | Dir: {task.direction.upper()}", flush=True)
                    print(f"[SNAPSHOT FILE] Saved -> {filepath} ({crop_size//1024} KB)", flush=True)
                    print(f"=======================================================\n", flush=True)
                    self.log(f"Saved Crop Snapshot: {filename} ({crop_size//1024} KB)")
                    
                    # Notify TrackManager FSM: SNAPSHOTTED
                    tm.notify_snapshot_created(task.camera_id, task.track_id, filepath, crop_size=crop_size)

                    # 4. Dispatch to AI Worker Queue for this camera (modulus mapped for shared queue pools)
                    dispatched = False
                    if self.ai_task_queues and len(self.ai_task_queues) > 0:
                        q_idx = task.camera_id % len(self.ai_task_queues)
                        q = self.ai_task_queues[q_idx]
                        try:
                            q.put(filepath, timeout=1.0)
                            dispatched = True
                        except Exception as q_err:
                            self.log(f"Queue push warning on Cam {task.camera_id}: {q_err}")
                            
                    if dispatched:
                        tm.notify_queue_success(task.camera_id, task.track_id)
                else:
                    self.log(f"CRITICAL: cv2.imwrite failed for {filename}")
                    tm.notify_snapshot_failed(task.camera_id, task.track_id, error="cv2.imwrite error")

            except Exception as e:
                self.log(f"Worker Error on Track {task.track_id}: {e}")
                tm.notify_snapshot_failed(task.camera_id, task.track_id, error=str(e))

            self.task_queue.task_done()
            self.state = "IDLE"


class SnapshotSupervisor(threading.Thread):
    """
    Automated Worker Supervisor.
    Monitors all dedicated camera SnapshotWorkers every 1.0s.
    Detects crashes/failures and automatically restarts worker threads.
    """
    def __init__(self, pipeline_manager):
        super().__init__(daemon=True)
        self.pm = pipeline_manager
        self.is_running = True

    def run(self):
        while self.is_running:
            time.sleep(1.0)
            with self.pm.lock:
                for cam_id, worker in list(self.pm.workers.items()):
                    if worker is None or not worker.is_alive() or worker.state == "FAILED":
                        print(f"[SnapshotSupervisor] ALERT: Worker for Camera {cam_id} is DEAD/FAILED. Auto-restarting...", flush=True)
                        self.pm.start_worker_for_camera(cam_id)


class SnapshotPipelineManager:
    """
    Singleton Manager for Per-Camera Isolated Snapshot Pipelines.
    Owns per-camera queues, dedicated SnapshotWorkers, and SnapshotSupervisor.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SnapshotPipelineManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, ai_task_queues=None, logger=None):
        if self._initialized:
            if ai_task_queues is not None:
                self.ai_task_queues = ai_task_queues
            return
            
        self._initialized = True
        self.ai_task_queues = ai_task_queues or []
        self.logger = logger
        
        self.queues = {}   # cam_id -> Queue
        self.workers = {}  # cam_id -> SnapshotWorker
        self.dispatched_files = set()  # Set of pending snapshot paths already queued
        self.lock = threading.RLock()  # Reentrant lock — allows nested acquisition
        
        self.supervisor = SnapshotSupervisor(self)
        self.supervisor.start()

    def get_queue(self, cam_id):
        with self.lock:
            if cam_id not in self.queues:
                self.queues[cam_id] = queue.Queue(maxsize=100)
            return self.queues[cam_id]

    def _get_queue_nolock(self, cam_id):
        """Get or create queue without acquiring lock (caller must hold lock)."""
        if cam_id not in self.queues:
            self.queues[cam_id] = queue.Queue(maxsize=100)
        return self.queues[cam_id]

    def start_worker_for_camera(self, cam_id):
        with self.lock:
            q = self._get_queue_nolock(cam_id)
            old_worker = self.workers.get(cam_id)
            if old_worker and old_worker.is_alive():
                old_worker.is_running = False
                
            worker = SnapshotWorker(cam_id, q, self.ai_task_queues, logger=self.logger)
            self.workers[cam_id] = worker
            worker.start()
            print(f"[SnapshotPipelineManager] Started dedicated SnapshotWorker for Camera {cam_id}.", flush=True)
            return worker

    def enqueue_task(self, cam_id, task: SnapshotTask):
        """
        Zero-Drop Bounded Queueing.
        Attempts enqueueing with timeout retries. Never silently drops events.
        """
        q = self.get_queue(cam_id)
        
        # Ensure worker thread is running
        with self.lock:
            if cam_id not in self.workers or not self.workers[cam_id].is_alive():
                self.start_worker_for_camera(cam_id)

        # Retry enqueueing with timeout to prevent silent drops
        for attempt in range(3):
            try:
                q.put(task, timeout=SNAPSHOT_QUEUE_TIMEOUT)
                return True
            except queue.Full:
                time.sleep(0.05)
                
        print(f"[SnapshotPipelineManager] CRITICAL: Queue for Camera {cam_id} full after 3 retries. Enqueuing blocking...", flush=True)
        try:
            q.put(task, block=True, timeout=3.0)
            return True
        except:
            return False

    def enqueue_snapshot(self, camera_id, frame_matrix, track=None):
        """
        Backward-compatible snapshot enqueueing method.
        Pushes frame matrix to ring FrameBuffer and enqueues SnapshotTask.
        """
        if frame_matrix is None or frame_matrix.size == 0 or track is None:
            return False
            
        fbm = FrameBufferManager()
        if not hasattr(self, '_legacy_frame_counter'):
            self._legacy_frame_counter = 0
        with self.lock:
            self._legacy_frame_counter += 1
            fid = self._legacy_frame_counter
            
        fbm.push_frame(camera_id, fid, frame_matrix)
        
        direction = getattr(track, 'direction', 'entrance')
        if direction not in ['entrance', 'exit', 'in', 'out']:
            direction = 'exit' if camera_id % 2 == 1 else 'entrance'
            
        task = SnapshotTask(
            track_id=getattr(track, 'track_id', 101),
            camera_id=camera_id,
            frame_id=fid,
            bounding_box=getattr(track, 'bounding_box', []),
            direction=direction,
            frame_timestamp=time.time()
        )
        return self.enqueue_task(camera_id, task)

    def scan_pending_snapshots(self):
        """Scans pending_snapshots directory and pushes un-processed files to AI queues."""
        pending_dir = os.path.join(BASE_DIR, "data", "pending_snapshots")
        if not os.path.exists(pending_dir):
            return
            
        try:
            valid_exts = ('.jpg', '.jpeg', '.png')
            files = [os.path.join(pending_dir, f) for f in os.listdir(pending_dir) if f.lower().endswith(valid_exts)]
            if not files:
                return
                
            files.sort(key=lambda f: os.path.getmtime(f))
            now = time.time()
            
            with self.lock:
                for filepath in files[:50]:
                    if (now - os.path.getmtime(filepath)) < 0.5:
                        continue
                        
                    filename = os.path.basename(filepath)
                    import re
                    c_match = re.search(r'cam_(\d+)_', filename)
                    cam_idx = int(c_match.group(1)) if c_match else 0
                    
                    if self.ai_task_queues and len(self.ai_task_queues) > 0:
                        q_idx = cam_idx % len(self.ai_task_queues)
                        q = self.ai_task_queues[q_idx]
                        try:
                            q.put_nowait(filepath)
                        except Exception:
                            pass
        except Exception:
            pass

    def is_alive(self):
        """Returns True if supervisor or any dedicated camera worker thread is alive."""
        with self.lock:
            if self.supervisor and self.supervisor.is_alive():
                return True
            return any(w and w.is_alive() for w in self.workers.values())

    def start(self):
        """Starts supervisor and ensures dedicated camera workers are active."""
        with self.lock:
            if self.supervisor is None or not self.supervisor.is_alive():
                self.supervisor = SnapshotSupervisor(self)
                self.supervisor.start()


# Compatibility wrapper functions for legacy callers
def get_snapshot_manager():
    return SnapshotPipelineManager()

SnapshotManager = SnapshotPipelineManager
