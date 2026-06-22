import cv2
import threading
import time
import socket

class ThreadedCamera:
    def __init__(self, src=0):
        # Force string sources to be clean of whitespace/newlines
        self.src = src.strip() if isinstance(src, str) else src
        
        # [FORCE TIMEOUT] Inject hard timeout directly into the RTSP string
        if isinstance(self.src, str) and self.src.startswith("rtsp://"):
            if "?" in self.src: self.src += "&timeout=5000000"
            else: self.src += "?timeout=5000000"
        
        # [NEW] Detect platform for backend optimization
        import platform
        self.is_windows = platform.system() == "Windows"

        # [BULLETPROOF GUARD] Perform 0.5s TCP Handshake before opening
        self.is_alive = True
        if isinstance(self.src, str) and "://" in self.src:
            try:
                # Extract IP and Port for handshake
                parts = self.src.split("@")[-1].split("/")[0].split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 554
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                if sock.connect_ex((host, port)) != 0:
                    self.is_alive = False
                sock.close()
            except: self.is_alive = False

        self.status = False
        self.frame = None
        self.stopped = False
        self.last_frame_time = time.time()

        if self.is_alive:
            # Only talk to video driver if handshake succeeded
            if isinstance(self.src, str) and "://" in self.src:
                print(f"[CAMERA] Opening Network Stream: {self.src}")
                self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            else:
                # Local Webcam: Try DSHOW first, then fallback to default
                print(f"[CAMERA] Probing Local Webcam Index: {self.src}")
                if self.is_windows:
                    self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                    if not self.capture or not self.capture.isOpened():
                        print(f"[CAMERA] DSHOW failed for index {self.src}, falling back to default...")
                        if self.capture: self.capture.release()
                        self.capture = cv2.VideoCapture(self.src)
                else:
                    self.capture = cv2.VideoCapture(self.src)
            
            if self.capture and self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Set some default resolution for webcams if needed
                if not isinstance(self.src, str):
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[CAMERA] CAM_{self.src} OPENED SUCCESSFULLY")
            else:
                print(f"[CAMERA ERROR] FAILED to open source: {self.src}")
        else:
            self.capture = None
            print(f"[HANDSHAKE] Failed for {self.src}. Skipping to avoid 30s hang.")

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            # If never opened or died, try reconnection ONLY if handshake passes
            if not self.capture or not self.capture.isOpened() or (time.time() - self.last_frame_time > 10.0):
                if not self.stopped:
                    # [RE-HANDSHAKE] Check liveness before retrying
                    try:
                        alive = True
                        if isinstance(self.src, str) and "://" in self.src:
                            alive = False
                            parts = self.src.split("@")[-1].split("/")[0].split(":")
                            host = parts[0]
                            port = int(parts[1]) if len(parts) > 1 else 554
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(0.5)
                            alive = (sock.connect_ex((host, port)) == 0)
                            sock.close()
                        
                        if alive:
                            if self.capture: self.capture.release()
                            if isinstance(self.src, str) and "://" in self.src:
                                self.capture = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                                self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                            else:
                                if self.is_windows:
                                    self.capture = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                                else:
                                    self.capture = cv2.VideoCapture(self.src)
                            
                            if self.capture and self.capture.isOpened():
                                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                if not isinstance(self.src, str):
                                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.last_frame_time = time.time() # Reset clock on success
                        
                        # If still not opening after a re-attempt, wait 2s before trying again
                        if not self.capture or not self.capture.isOpened():
                            time.sleep(2.0)
                    except: pass

            if self.capture and self.capture.isOpened():
                # Flush the internal FFMPEG buffer to prevent streaming lag on RTSP network streams
                if isinstance(self.src, str) and "://" in self.src:
                    for _ in range(5):
                        self.capture.grab()

                status, frame = self.capture.read()
                if status:
                    self.status = True
                    self.frame = frame
                    self.last_frame_time = time.time()
                else:
                    self.status = False
                    # On failure, don't clear the frame immediately to allow for flicker-free recon
            
            time.sleep(0.01) # ~100 FPS cap

    def read(self):
        # [NON-BLOCKING] Always return something immediately
        # If no frame ever captured, return False
        if self.frame is None:
            return False, None
            
        # Return status=False only if stale for more than 10 seconds
        is_stale = (time.time() - self.last_frame_time > 10.0)
        return not is_stale, self.frame
    
    def isOpened(self):
        # Return True if the capture is open OR if we are still in the process of (re)connecting
        return (self.capture is not None and self.capture.isOpened()) or (not self.stopped and not self.is_alive)

    def release(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except: pass
        if self.capture:
            self.capture.release()
            self.capture = None

    def get(self, propId):
        if self.capture and self.capture.isOpened():
            return self.capture.get(propId)
        return None
