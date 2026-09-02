import cv2
import threading
import time
import socket
import os

# [ZERO LATENCY RTSP] Disable FFMPEG internal buffer queues and set 5s socket timeout
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|framedrop;1|stimeout;5000000"

class ThreadedCamera:
    def __init__(self, src=0):
        # Ensure self.src_main is ALWAYS Main-Stream (5 MP) and self.src_sub is ALWAYS Sub-Stream (low-CPU)
        input_src = src.strip() if isinstance(src, str) else src
        if isinstance(input_src, str) and "://" in input_src:
            import re
            # Derive main-stream (high-res for snapshots)
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

        # Maintain backward compatibility field
        self.src = self.src_sub
        
        # Detect platform for backend optimization
        import platform
        self.is_windows = platform.system() == "Windows"

        # Pre-check host reachability with graceful fallback
        self.is_alive = True
        if isinstance(self.src_sub, str) and "://" in self.src_sub:
            try:
                no_proto = self.src_sub.split("://", 1)[-1]
                host_part = no_proto.rsplit("@", 1)[-1] if "@" in no_proto else no_proto
                host_port = host_part.split("/")[0]
                if ":" in host_port:
                    host, port_str = host_port.split(":", 1)
                    port = int(port_str)
                else:
                    host = host_port
                    port = 554
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                res = sock.connect_ex((host, port))
                sock.close()
                if res != 0:
                    print(f"[CAMERA] Warning: Initial TCP ping to {host}:{port} returned {res}, attempting OpenCV connection anyway...")
            except Exception as e:
                print(f"[CAMERA] Host check note: {e}")
            
        self.status = False
        self.frame = None
        self.frame_lock = threading.Lock()
        self.has_frame = False
        self.stopped = False
        self.last_frame_time = time.time()
        self.frame_consumed = True
        
        self.capture_sub = None
        self.capture = None

        if self.is_alive:
            # Open Sub-stream (low-res H.264 for smooth zero-lag 30 FPS GUI preview)
            if isinstance(self.src_sub, str) and "://" in self.src_sub:
                print(f"[CAMERA] Opening Sub-stream: {self.src_sub}")
                self.capture_sub = cv2.VideoCapture(self.src_sub, cv2.CAP_FFMPEG)
                self.capture_sub.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            else:
                print(f"[CAMERA] Probing Local Webcam Index: {self.src_sub}")
                if self.is_windows:
                    self.capture_sub = cv2.VideoCapture(self.src_sub, cv2.CAP_DSHOW)
                    if not self.capture_sub or not self.capture_sub.isOpened():
                        if self.capture_sub: self.capture_sub.release()
                        self.capture_sub = cv2.VideoCapture(self.src_sub)
                else:
                    self.capture_sub = cv2.VideoCapture(self.src_sub)
            
            if self.capture_sub and self.capture_sub.isOpened():
                self.capture_sub.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not isinstance(self.src_sub, str):
                    self.capture_sub.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture_sub.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[CAMERA] Sub-stream CAM_{self.src_sub} OPENED SUCCESSFULLY")
            else:
                print(f"[CAMERA ERROR] FAILED to open sub-stream source: {self.src_sub}")

        self.capture = self.capture_sub

        self.thread_sub = threading.Thread(target=self.update_sub, args=())
        self.thread_sub.daemon = True
        self.thread_sub.start()

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
                                self.capture = self.capture_sub
                                self.last_frame_time = time.time()
                        
                        if not self.capture_sub or not self.capture_sub.isOpened():
                            time.sleep(2.0)
                    except: pass

            # 2. Continuous zero-lag frame update
            if self.capture_sub and self.capture_sub.isOpened():
                try:
                    if self.capture_sub.grab():
                        status, frame = self.capture_sub.retrieve()
                        if status and frame is not None and frame.size > 0:
                            with self.frame_lock:
                                self.status = True
                                self.frame = frame
                                self.has_frame = True
                                self.last_frame_time = time.time()
                    else:
                        time.sleep(0.002)
                except Exception:
                    self.status = False
                    time.sleep(0.01)
            else:
                time.sleep(0.05)

    def read(self):
        with self.frame_lock:
            if self.frame is None:
                return False, None
            is_stale = (time.time() - self.last_frame_time > 10.0)
            return not is_stale, self.frame.copy()

    def read_high_res(self):
        with self.frame_lock:
            if self.frame is not None and self.frame.size > 0:
                return True, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.capture_sub is not None and self.capture_sub.isOpened() and not self.stopped

    def release(self):
        self.stopped = True
        try:
            self.thread_sub.join(timeout=1.0)
        except: pass
        if self.capture_sub:
            self.capture_sub.release()
            self.capture_sub = None
        self.capture = None

    def get(self, propId):
        if self.capture_sub and self.capture_sub.isOpened():
            return self.capture_sub.get(propId)
        return None
