import time
import threading
from config.config import FRAME_BUFFER_SECONDS

class FrameEntry:
    """
    Reference-Counted Frame Entry.
    Maintains frame matrix, timestamp, and active reference count.
    A frame is never purged while reference_count > 0.
    """
    def __init__(self, frame_id, camera_id, frame_matrix, timestamp=None):
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.frame_matrix = frame_matrix
        self.timestamp = timestamp or time.time()
        self.reference_count = 0
        self.lock = threading.Lock()

    def add_ref(self):
        with self.lock:
            self.reference_count += 1
            return self.reference_count

    def release_ref(self):
        with self.lock:
            self.reference_count = max(0, self.reference_count - 1)
            return self.reference_count


class FrameBuffer:
    """
    Thread-Safe Reference-Counted Circular FrameBuffer per camera stream.
    Calculates capacity dynamically: capacity = camera_fps * FRAME_BUFFER_SECONDS.
    Purges oldest frames ONLY when reference_count == 0.
    """
    def __init__(self, camera_id, fps=30.0):
        self.camera_id = camera_id
        self.fps = max(10.0, float(fps))
        self.capacity = int(max(30, self.fps * FRAME_BUFFER_SECONDS))
        self.frames = {}  # frame_id -> FrameEntry
        self.ordered_ids = []
        self.lock = threading.Lock()

    def update_fps(self, fps):
        if fps and fps > 5:
            with self.lock:
                self.fps = float(fps)
                self.capacity = int(max(30, self.fps * FRAME_BUFFER_SECONDS))

    def push_frame(self, frame_id, frame_matrix, timestamp=None):
        if frame_matrix is None or frame_matrix.size == 0:
            return None
            
        entry = FrameEntry(frame_id, self.camera_id, frame_matrix, timestamp)
        
        with self.lock:
            self.frames[frame_id] = entry
            self.ordered_ids.append(frame_id)
            
            # Circular Purge: Remove oldest frames when over capacity IF reference_count == 0
            while len(self.ordered_ids) > self.capacity:
                oldest_id = self.ordered_ids[0]
                oldest_entry = self.frames.get(oldest_id)
                
                if oldest_entry and oldest_entry.reference_count == 0:
                    self.ordered_ids.pop(0)
                    del self.frames[oldest_id]
                else:
                    # Oldest frame is currently pinned by a SnapshotTask! Break purge loop.
                    break
                    
        return entry

    def get_frame(self, frame_id):
        with self.lock:
            return self.frames.get(frame_id)


class FrameBufferManager:
    """
    Singleton Manager for per-camera FrameBuffers.
    Provides global thread-safe access to ring buffers.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FrameBufferManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.buffers = {}
        self.lock = threading.Lock()

    def get_buffer(self, camera_id):
        with self.lock:
            if camera_id not in self.buffers:
                self.buffers[camera_id] = FrameBuffer(camera_id)
            return self.buffers[camera_id]

    def push_frame(self, camera_id, frame_id, frame_matrix, timestamp=None):
        buf = self.get_buffer(camera_id)
        return buf.push_frame(frame_id, frame_matrix, timestamp)

    def get_frame(self, camera_id, frame_id):
        buf = self.get_buffer(camera_id)
        return buf.get_frame(frame_id)
