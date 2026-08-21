import cv2
import threading
import time
import socket
import os

# [ZERO LATENCY RTSP] Disable FFMPEG internal buffer queues to get true real-time feeds
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|framedrop;1"

class ThreadedCamera:
    def __init__(self, src=0):
        # Ensure self.src_main is ALWAYS Main-Stream (5 MP) and self.src_sub is ALWAYS Sub-Stream (low-CPU)
        input_src = src.strip() if isinstance(src, str) else src
        if isinstance(input_src, str) and "://" in input_src:
            import re
            # Derive main-stream (high-res 5 MP for snapshots)
            self.src_main = input_src
            if "subtype=1" in self.src_main:
                self.src_main = self.src_main.replace("subtype=1", "subtype=0")
            elif "/sub" in self.src_main:
                self.src_main = self.src_main.replace("/sub", "/main")
            elif "stream=1" in self.src_main:
                self.src_main = self.src_main.replace("stream=1", "stream=0")
            elif re.search(r'Channels/(\d+)02', self.src_main):
                self.src_main = re.sub(r'Channels/(\d+)02', r'Channels/\101', self.src_main)

            # Derive sub-stream (low-res for smooth 30 FPS GUI preview)
            self.src_sub = input_src
            if "subtype=0" in self.src_sub:
                self.src_sub = self.src_sub.replace("subtype=0", "subtype=1")
            elif "/main" in self.src_sub:
                self.src_sub = self.src_sub.replace("/main", "/sub")
            elif "stream=0" in self.src_sub:
                self.src_sub = self.src_sub.replace("stream=0", "stream=1")
            elif re.search(r'Channels/(\d+)01', self.src_sub):
                self.src_sub = re.sub(r'Channels/(\d+)01', r'Channels/\102', self.src_sub)
        else:
            self.src_main = input_src
            self.src_sub = input_src

        # [FORCE TIMEOUT] Inject hard timeout only if not already present
        for attr in ['src_main', 'src_sub']:
            val = getattr(self, attr, None)
            if isinstance(val, str) and val.startswith("rtsp://"):
                if "timeout=" not in val:
                    sep = "&" if "?" in val else "?"
                    setattr(self, attr, f"{val}{sep}timeout=5000000")

        # Maintain backward compatibility field
        self.src = self.src_sub
        
        # [NEW] Detect platform for backend optimization
        import platform
        self.is_windows = platform.system() == "Windows"

        # [BULLETPROOF GUARD] Perform 1.5s TCP Handshake before opening
        self.is_alive = True
        if isinstance(self.src_sub, str) and "://" in self.src_sub:
            try:
                # Extract IP and Port for handshake
                parts = self.src_sub.split("@")[-1].split("/")[0].split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 554
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.5)
                if sock.connect_ex((host, port)) != 0:
                    self.is_alive = False
                sock.close()
            except: self.is_alive = False
            
        self.status = False
        self.frame = None
        self.has_frame = False
        self.stopped = False
        self.last_frame_time = time.time()
        self.frame_consumed = True
        
        self.capture_sub = None
        self.capture_main = None
        self.main_lock = threading.Lock()

        if self.is_alive:
            # 1. Open Sub-stream (continuous decoding)
            if isinstance(self.src_sub, str) and "://" in self.src_sub:
                print(f"[CAMERA] Opening Sub-stream: {self.src_sub}")
                self.capture_sub = cv2.VideoCapture(self.src_sub, cv2.CAP_FFMPEG)
                self.capture_sub.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            else:
                # Local Webcam: Try DSHOW first, then fallback to default
                print(f"[CAMERA] Probing Local Webcam Index: {self.src_sub}")
                if self.is_windows:
                    self.capture_sub = cv2.VideoCapture(self.src_sub, cv2.CAP_DSHOW)
                    if not self.capture_sub or not self.capture_sub.isOpened():
                        print(f"[CAMERA] DSHOW failed for index {self.src_sub}, falling back to default...")
                        if self.capture_sub: self.capture_sub.release()
                        self.capture_sub = cv2.VideoCapture(self.src_sub)
                else:
                    self.capture_sub = cv2.VideoCapture(self.src_sub)
            
            # Set sub-stream properties
            if self.capture_sub and self.capture_sub.isOpened():
                self.capture_sub.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not isinstance(self.src_sub, str):
                    self.capture_sub.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture_sub.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[CAMERA] Sub-stream CAM_{self.src_sub} OPENED SUCCESSFULLY")
            else:
                print(f"[CAMERA ERROR] FAILED to open sub-stream source: {self.src_sub}")
                
            # 2. Open Main-stream (deferred to update_main background thread to avoid blocking startup)
            if self.src_main != self.src_sub and isinstance(self.src_main, str) and "://" in self.src_main:
                print(f"[CAMERA] Main-stream connection deferred to background thread: {self.src_main}")
        else:
            print(f"[HANDSHAKE] Failed for {self.src_sub}. Skipping to avoid 30s hang.")

        # Backward compatibility field reference
        self.capture = self.capture_sub

        self.thread_sub = threading.Thread(target=self.update_sub, args=())
        self.thread_sub.daemon = True
        self.thread_sub.start()

        self.thread_main = threading.Thread(target=self.update_main, args=())
        self.thread_main.daemon = True
        self.thread_main.start()

    def update_sub(self):
        while not self.stopped:
            # 1. Reconnect Sub-stream if disconnected/stale
            if not self.capture_sub or not self.capture_sub.isOpened() or (time.time() - self.last_frame_time > 10.0):
                if not self.stopped:
                    try:
                        alive = True
                        if isinstance(self.src_sub, str) and "://" in self.src_sub:
                            alive = False
                            parts = self.src_sub.split("@")[-1].split("/")[0].split(":")
                            host = parts[0]
                            port = int(parts[1]) if len(parts) > 1 else 554
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(0.5)
                            alive = (sock.connect_ex((host, port)) == 0)
                            sock.close()
                        
                        if alive:
                            if self.capture_sub: self.capture_sub.release()
                            if isinstance(self.src_sub, str) and "://" in self.src_sub:
                                self.capture_sub = cv2.VideoCapture(self.src_sub, cv2.CAP_FFMPEG)
                                self.capture_sub.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                            else:
                                if self.is_windows:
                                    self.capture_sub = cv2.VideoCapture(self.src_sub, cv2.CAP_DSHOW)
                                else:
                                    self.capture_sub = cv2.VideoCapture(self.src_sub)
                            
                            if self.capture_sub and self.capture_sub.isOpened():
                                self.capture_sub.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                if not isinstance(self.src_sub, str):
                                    self.capture_sub.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                    self.capture_sub.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.capture = self.capture_sub # Keep reference updated
                                self.last_frame_time = time.time()
                        
                        if not self.capture_sub or not self.capture_sub.isOpened():
                            time.sleep(2.0)
                    except: pass

            # 3. Read continuously from sub-stream (instant non-blocking frame update)
            if self.capture_sub and self.capture_sub.isOpened():
                try:
                    status, frame = self.capture_sub.read()
                    if status and frame is not None and frame.size > 0:
                        self.status = True
                        self.frame = frame
                        self.has_frame = True
                        self.last_frame_time = time.time()
                    else:
                        self.status = False
                        time.sleep(0.005)
                except Exception as e:
                    print(f"[CAMERA ERROR] Exception during sub-stream read: {e}")
                    self.status = False
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def update_main(self):
        while not self.stopped:
            # 2. Reconnect Main-stream if it was opened but closed
            if self.src_main != self.src_sub and (not self.capture_main or not self.capture_main.isOpened()):
                if not self.stopped:
                    try:
                        alive = True
                        if isinstance(self.src_main, str) and "://" in self.src_main:
                            alive = False
                            parts = self.src_main.split("@")[-1].split("/")[0].split(":")
                            host = parts[0]
                            port = int(parts[1]) if len(parts) > 1 else 554
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(0.5)
                            alive = (sock.connect_ex((host, port)) == 0)
                            sock.close()
                        if alive:
                            with self.main_lock:
                                if self.capture_main: self.capture_main.release()
                                self.capture_main = cv2.VideoCapture(self.src_main, cv2.CAP_FFMPEG)
                                self.capture_main.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                                if self.capture_main and self.capture_main.isOpened():
                                    self.capture_main.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except: pass

            # 4. Grab continuously from main-stream (without decoding) to clear network buffers
            # Decoding is only done on-demand in read_high_res() when a snapshot is triggered
            if self.capture_main and self.capture_main.isOpened():
                try:
                    with self.main_lock:
                        self.capture_main.grab()
                except Exception as e:
                    print(f"[CAMERA ERROR] Exception during main-stream grab: {e}")

            time.sleep(0.03)

    def read(self):
        if self.frame is None:
            return False, None
        is_stale = (time.time() - self.last_frame_time > 10.0)
        return not is_stale, self.frame

    def read_high_res(self):
        # On-demand grab+retrieve from main-stream (1080p/5MP high-resolution snapshot frame)
        if self.capture_main and self.capture_main.isOpened():
            try:
                with self.main_lock:
                    if self.capture_main.grab():
                        status, frame = self.capture_main.retrieve()
                        if status and frame is not None and frame.size > 0:
                            return True, frame
            except Exception as e:
                print(f"[CAMERA ERROR] Exception during main-stream retrieve: {e}")
        # Fallback to current frame
        if self.frame is not None and self.frame.size > 0:
            return True, self.frame.copy()
        return False, None

    def isOpened(self):
        return self.capture_sub is not None and self.capture_sub.isOpened() and not self.stopped

    def release(self):
        self.stopped = True
        try:
            self.thread_sub.join(timeout=1.0)
            self.thread_main.join(timeout=1.0)
        except: pass
        if self.capture_sub:
            self.capture_sub.release()
            self.capture_sub = None
        if self.capture_main:
            with self.main_lock:
                self.capture_main.release()
                self.capture_main = None
        self.capture = None

    def get(self, propId):
        if self.capture_sub and self.capture_sub.isOpened():
            return self.capture_sub.get(propId)
        return None
