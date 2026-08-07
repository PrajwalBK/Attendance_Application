import os
import time
import threading
from enum import Enum
from datetime import datetime
from config.config import (
    TRACK_TIMEOUT_SECONDS,
    QUEUE_TIMEOUT_SECONDS,
    PROCESSING_TIMEOUT_SECONDS,
    MAX_RETRY
)

class TrackState(Enum):
    NEW = "NEW"
    SNAPSHOT_PENDING = "SNAPSHOT_PENDING"
    SNAPSHOTTED = "SNAPSHOTTED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    MATCHED = "MATCHED"
    ATTENDANCE_PENDING = "ATTENDANCE_PENDING"
    ATTENDANCE_DONE = "ATTENDANCE_DONE"
    FAILED = "FAILED"
    FAILED_PERMANENT = "FAILED_PERMANENT"
    EXPIRED = "EXPIRED"

class TrackVisibility(Enum):
    VISIBLE = "VISIBLE"
    LOST = "LOST"

class Track:
    """
    Pure Spatial & Tracking State Container.
    Maintains track_id, spatial coordinates, movement vector, visibility, 
    and state machine flags. Does NOT store AI employee names or DB IDs.
    """
    def __init__(self, track_id, camera_id, bounding_box, now):
        self.track_id = track_id
        self.camera_id = camera_id
        self.bounding_box = list(bounding_box)  # [bx1, by1, bx2, by2]
        
        cx = (bounding_box[0] + bounding_box[2]) / 2.0
        cy = (bounding_box[1] + bounding_box[3]) / 2.0
        self.centroid = (cx, cy)
        self.velocity = (0.0, 0.0)
        self.direction = "monitor"
        
        self.first_seen = now
        self.last_seen = now
        self.state_change_time = now
        
        self.state = TrackState.NEW
        self.visibility = TrackVisibility.VISIBLE
        self.retry_count = 0
        self.frame_counter = 1
        self.last_snapshot_time = 0.0  # Enables instant & continuous snapshot interval capturing
        
        self.crossed_entry_line = False
        self.crossed_exit_line = False
        
        self.history = [(cx, cy)]
        self.last_frame = None
        self.snapshot_path = None

    def update_spatial(self, box, now, frame=None):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        if len(self.history) >= 1:
            prev_cx, prev_cy = self.history[-1]
            dt = max(0.001, now - self.last_seen)
            self.velocity = ((cx - prev_cx) / dt, (cy - prev_cy) / dt)
            
        self.bounding_box = list(box)
        self.centroid = (cx, cy)
        self.last_seen = now
        self.visibility = TrackVisibility.VISIBLE
        self.frame_counter += 1
        
        self.history.append((cx, cy))
        if len(self.history) > 30:
            self.history.pop(0)
            
        if frame is not None:
            self.last_frame = frame


class TrackManager:
    """
    Centralized Track Manager.
    SOLE OWNER of TrackState and FSM transitions.
    Manages track lifecycles, latency metrics, and thread-safe notifications.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrackManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, logger=None):
        if self._initialized:
            return
        self._initialized = True
        self.logger = logger
        self.tracks = {}  # (cam_id, track_id) -> Track
        self.lock = threading.Lock()
        
        # Performance & Runtime Metrics
        self.metrics = {
            'tracks_created': 0,
            'tracks_expired': 0,
            'snapshots_saved': 0,
            'recognition_success': 0,
            'recognition_failed': 0,
            'attendance_success': 0,
            'attendance_failed': 0,
            'avg_queue_time': 0.0,
            'avg_recognition_time': 0.0,
            'avg_attendance_time': 0.0
        }
        
        # Start background watchdog thread
        self.watchdog = TrackWatchdog(self)
        self.watchdog.start()

    def log(self, msg):
        timestamp_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp_str}] [TrackManager] {msg}"
        if self.logger:
            try: self.logger(formatted)
            except: print(formatted, flush=True)
        else:
            print(formatted, flush=True)

    def transition_state(self, track: Track, new_state: TrackState, reason: str = ""):
        old_state = track.state
        now = time.time()
        elapsed = now - track.state_change_time
        track.state = new_state
        track.state_change_time = now
        
        reason_str = f" ({reason})" if reason else ""
        self.log(f"[Track {track.track_id} | Cam {track.camera_id}] State: {old_state.value} -> {new_state.value}{reason_str} | Latency: {elapsed*1000:.1f}ms")

    def get_track(self, cam_id, track_id):
        with self.lock:
            return self.tracks.get((cam_id, track_id))

    def register_track(self, track: Track):
        with self.lock:
            self.tracks[(track.camera_id, track.track_id)] = track
            self.metrics['tracks_created'] += 1
            self.log(f"Registered Track {track.track_id} on Cam {track.camera_id}")

    # --- FSM EVENT NOTIFICATIONS ---

    def notify_snapshot_requested(self, cam_id, track_id):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state == TrackState.NEW:
                self.transition_state(track, TrackState.SNAPSHOT_PENDING, "Snapshot requested")
                return True
        return False

    def notify_snapshot_created(self, cam_id, track_id, snapshot_path, crop_size=0):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state == TrackState.SNAPSHOT_PENDING:
                track.snapshot_path = snapshot_path
                self.metrics['snapshots_saved'] += 1
                self.transition_state(track, TrackState.SNAPSHOTTED, f"Crop saved: {crop_size//1024}KB")
                return True
        return False

    def notify_snapshot_failed(self, cam_id, track_id, error=""):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state == TrackState.SNAPSHOT_PENDING:
                track.retry_count += 1
                if track.retry_count >= MAX_RETRY:
                    self.transition_state(track, TrackState.FAILED_PERMANENT, f"Snapshot failed {track.retry_count}x: {error}")
                else:
                    self.transition_state(track, TrackState.NEW, f"Snapshot failed (Retry {track.retry_count}/{MAX_RETRY}): {error}")
                return True
        return False

    def notify_queue_success(self, cam_id, track_id):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state in (TrackState.SNAPSHOTTED, TrackState.FAILED):
                self.transition_state(track, TrackState.QUEUED, "Pushed to AI Priority Queue")
                return True
        return False

    def notify_processing_started(self, cam_id, track_id):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state == TrackState.QUEUED:
                self.transition_state(track, TrackState.PROCESSING, "AI Worker picked task")
                return True
        return False

    def notify_recognition_result(self, cam_id, track_id, is_match, sim=0.0, pid=None, name=None, processing_time=0.0):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track:
                # Update running average recognition time
                if processing_time > 0:
                    current_avg = self.metrics['avg_recognition_time']
                    self.metrics['avg_recognition_time'] = (current_avg * 0.9) + (processing_time * 0.1) if current_avg > 0 else processing_time
                    
                if is_match:
                    self.metrics['recognition_success'] += 1
                    self.transition_state(track, TrackState.MATCHED, f"Match: {name} ({pid}) Sim: {sim:.2f}")
                else:
                    self.metrics['recognition_failed'] += 1
                    track.retry_count += 1
                    if track.retry_count >= MAX_RETRY:
                        self.transition_state(track, TrackState.FAILED_PERMANENT, f"No face / Low sim ({sim:.2f}) {track.retry_count}x")
                    else:
                        self.transition_state(track, TrackState.FAILED, f"Low sim ({sim:.2f}) Retry {track.retry_count}/{MAX_RETRY}")
                return True
        return False

    def notify_attendance_requested(self, cam_id, track_id):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state == TrackState.MATCHED:
                self.transition_state(track, TrackState.ATTENDANCE_PENDING, "Attendance DB write initiated")
                return True
        return False

    def notify_attendance_result(self, cam_id, track_id, success, error=""):
        with self.lock:
            track = self.tracks.get((cam_id, track_id))
            if track and track.state in (TrackState.ATTENDANCE_PENDING, TrackState.MATCHED):
                if success:
                    self.metrics['attendance_success'] += 1
                    self.transition_state(track, TrackState.ATTENDANCE_DONE, "Logged to SQLite & API")
                else:
                    self.metrics['attendance_failed'] += 1
                    self.transition_state(track, TrackState.MATCHED, f"Attendance log retry: {error}")
                return True
        return False

    def expire_stale_tracks(self, now, timeout_seconds=TRACK_TIMEOUT_SECONDS):
        with self.lock:
            expired_keys = []
            for key, track in self.tracks.items():
                if now - track.last_seen >= timeout_seconds:
                    track.visibility = TrackVisibility.LOST
                    if track.state not in (TrackState.PROCESSING, TrackState.ATTENDANCE_PENDING):
                        self.transition_state(track, TrackState.EXPIRED, "Absent timeout")
                        expired_keys.append(key)
            for k in expired_keys:
                del self.tracks[k]
                self.metrics['tracks_expired'] += 1


class TrackWatchdog(threading.Thread):
    """
    Background Watchdog Thread.
    Audits active tracks every 1.0 second. Resets stuck QUEUED (>5s) or 
    PROCESSING (>10s) tracks to prevent worker locks or stuck states.
    """
    def __init__(self, track_manager):
        super().__init__(daemon=True)
        self.tm = track_manager
        self.is_running = True

    def run(self):
        while self.is_running:
            time.sleep(1.0)
            now = time.time()
            with self.tm.lock:
                for key, track in list(self.tm.tracks.items()):
                    elapsed = now - track.state_change_time
                    
                    # Watchdog Check 1: Stuck in QUEUED state (> 5s)
                    if track.state == TrackState.QUEUED and elapsed > QUEUE_TIMEOUT_SECONDS:
                        self.tm.log(f"[WATCHDOG] Track {track.track_id} stuck in QUEUED for {elapsed:.1f}s. Resetting to SNAPSHOTTED.")
                        self.tm.transition_state(track, TrackState.SNAPSHOTTED, "Watchdog timeout queue reset")
                        
                    # Watchdog Check 2: Stuck in PROCESSING state (> 10s)
                    elif track.state == TrackState.PROCESSING and elapsed > PROCESSING_TIMEOUT_SECONDS:
                        track.retry_count += 1
                        if track.retry_count >= MAX_RETRY:
                            self.tm.log(f"[WATCHDOG] Track {track.track_id} stuck in PROCESSING for {elapsed:.1f}s. Transitioning to FAILED_PERMANENT.")
                            self.tm.transition_state(track, TrackState.FAILED_PERMANENT, "Watchdog processing timeout")
                        else:
                            self.tm.log(f"[WATCHDOG] Track {track.track_id} stuck in PROCESSING for {elapsed:.1f}s. Resetting to QUEUED.")
                            self.tm.transition_state(track, TrackState.QUEUED, "Watchdog processing reset")
