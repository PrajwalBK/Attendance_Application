import os
import multiprocessing
import threading
import time
import socket
import queue
from PySide6.QtCore import QThread, Signal, QObject
from datetime import datetime

# Import existing workers
from core.multiprocess_handler import AttendanceWorker
from core.recorder import RecorderWorker
from core.camera import ThreadedCamera
from core.face_recognition import resource_path

MAX_CAMS = 8

def _quantize_onnx_model(input_path, output_path):
    if not os.path.exists(output_path) and os.path.exists(input_path):
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            print(f"[QUANTIZER] Performing dynamic INT8 quantization: {input_path} -> {output_path}", flush=True)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QUInt8
            )
            print(f"[QUANTIZER] INT8 Quantization completed successfully: {output_path}", flush=True)
            return True
        except Exception as e:
            print(f"[QUANTIZER] Dynamic INT8 Quantization failed: {e}", flush=True)
    return False

class UIWorker(QObject):
    """Signals for UI updates from the backend."""
    detection_occurred = Signal(dict)
    status_updated = Signal(str)
    stats_updated = Signal(dict)
    camera_error = Signal(int, str)

class BackendController(QThread):
    """Manages the lifecycle of AI workers in a background Qt thread."""
    
    def __init__(self):
        super().__init__()
        self.worker_signals = UIWorker()
        self.is_running = False
        self.stop_requested = False

        # Workers (Up to 8 workers)
        self.workers = [None] * MAX_CAMS
        self.recorder = None
        
        # Queues & Locks for worker communication
        self.result_queue = multiprocessing.Queue()
        self.recording_queue = multiprocessing.Queue(maxsize=240)
        self.task_queues = [multiprocessing.Queue() for _ in range(MAX_CAMS)]
        self.init_lock = multiprocessing.Lock()
        
        # Cameras & Frame Buffer
        self.caps = [None] * MAX_CAMS
        self.are_cameras_active = False
        self.latest_frames = [None] * MAX_CAMS
        
        # Load active count from config
        from config.cam_config_manager import CamConfigManager
        config = CamConfigManager.load_config()
        self.active_cam_count = config.get("active_cam_count", 6)
        
        from config.config import get_config
        app_cfg = get_config()
        self.record_video = app_cfg.get('record_video', False)
        self.process_every_n_frames = app_cfg.get('process_every_n_frames', 1)
        self.is_detection_enabled = False
        
        # Per-camera role, ROI, and Direction Rules: Dynamic initialization from config
        self.cam_roles = ['monitor'] * MAX_CAMS
        self.cam_rois = [[0.0, 0.0, 1.0, 1.0]] * MAX_CAMS
        self.cam_direction_rules = [{"up": "out", "down": "in", "left": "ignore", "right": "ignore"}] * MAX_CAMS
        cams_list = config.get("cams", [])
        for i in range(min(len(self.cam_roles), self.active_cam_count)):
            if i < len(cams_list):
                c = cams_list[i]
                self.cam_roles[i] = c.get("role", "monitor").lower()
                self.cam_rois[i] = c.get("roi", [0.0, 0.0, 1.0, 1.0])
                self.cam_direction_rules[i] = c.get("direction_rules", {"up": "out", "down": "in", "left": "ignore", "right": "ignore"})
            else:
                if i == 0: self.cam_roles[i] = 'entrance'
                elif i == 1: self.cam_roles[i] = 'exit'
                else: self.cam_roles[i] = 'monitor'
        self.frame_lock = threading.Lock()
        
        # Authentication & Remote Sync
        self.auth_token = None
        self.auth_email = None
        self.auth_pass = None
        from config.config import API_BASE_URL
        self.remote_url = API_BASE_URL
        self.api_client = None
        self.last_api_sync = 0
        self.last_offline_sync = 0
        
        # Folder monitoring for offline snapshot detection
        self.queued_snapshots = {}
        try:
            from config.config import BASE_DIR
            import os
            os.makedirs(os.path.join(BASE_DIR, "data", "pending_snapshots"), exist_ok=True)
        except Exception:
            pass

    @property
    def is_detection_running(self):
        """Returns True if at least one AI worker is alive and running."""
        return any(w and w.is_alive() for w in self.workers)

    def reconnect_cloud(self, email, password, url):
        self.auth_email = email
        self.auth_pass = password
        
        # [ROBUSTNESS] Sanitize and preserve local IP ports
        from config.config import sanitize_url
        self.remote_url = sanitize_url(url)
        
        from core.api_client import APIClient
        self.api_client = APIClient(self.remote_url, api_user=self.auth_email, api_password=self.auth_pass)
        
        try:
            self.worker_signals.status_updated.emit(f"Linking to: {self.remote_url}...")
            success, token = self.api_client.login()
            
            if success:
                self.auth_token = token
                self.worker_signals.status_updated.emit("Cloud Connected")
                
                # Broadast to workers
                for q in self.task_queues:
                    if q: q.put({'type': 'update_api', 'url': self.remote_url, 'user': self.auth_email, 'pass': self.auth_pass})
                
                threading.Thread(target=self.sync_remote_faces, daemon=True).start()
                return True, "Cloud Linked Successfully"
            else:
                self.worker_signals.status_updated.emit("Cloud Link Failed")
                # [DIAGNOSTIC] Help the user understand Local vs Cloud 401s
                is_local = any(x in self.remote_url for x in ["192.168.", "10.", "localhost", "127.0.0.1"])
                if "401" in str(token) and is_local:
                    return False, f"Local Server Rejected Credentials.\nHint: Does '{email}' exist on your local backend?"
                return False, f"Link Failed: {token}"
        except Exception as e:
            return False, f"Connection Error: {str(e)}"

    def set_local_db_mode(self):
        """Initializes the backend controller in Direct Local Database mode."""
        from database.database import DatabaseManager
        self.api_client = DatabaseManager()
        self.worker_signals.status_updated.emit("Local DB Connected")

    def set_auth_credentials(self, email, password, token, url=None):
        self.auth_email = email
        self.auth_pass = password
        self.auth_token = token
        
        # [DYNAMIC] If no URL provided, pull the latest from config
        if not url:
            from config.config import refresh_api_config
            _, _, url = refresh_api_config()
            
        if url: self.remote_url = url
        
        # Initialize APIClient for sync
        from core.api_client import APIClient
        self.api_client = APIClient(self.remote_url, api_user=self.auth_email, api_password=self.auth_pass, token=self.auth_token)
        
        # Propagate to workers immediately
        for q in self.task_queues:
            if q: q.put({'type': 'update_api', 'url': self.remote_url, 'user': self.auth_email, 'pass': self.auth_pass})
            
        self.worker_signals.status_updated.emit("Cloud Connected")

    def _log_cctv_error(self, msg):
        """Writes detailed camera failure reports to 'cctv_error.log'."""
        try:
            with open("cctv_error.log", "a") as f:
                f.write(f"[{datetime.now()}] {msg}\n")
        except: pass

    def sync_remote_faces(self):
        """Signals AI workers to fetch and reload face encodings directly from the remote API or local DB."""
        from config.config import get_config
        use_api = get_config().get('use_api', True)
        
        if not use_api:
            try:
                self.worker_signals.status_updated.emit("Syncing Local Database...")
                # Tell all active AI workers to reload their memory directly from the DB
                for q in self.task_queues:
                    if q: q.put({'type': 'reload'})
                
                self.worker_signals.status_updated.emit("Sync Complete")
                return True, "Sync signal sent successfully to AI workers."
            except Exception as e:
                self.worker_signals.status_updated.emit("Sync Error")
                print(f"[SYNC ERROR] {e}")
                return False, f"Sync error: {str(e)}"
                
        if not self.auth_token:
            return False, "Not authenticated with cloud."
            
        try:
            self.worker_signals.status_updated.emit("Syncing Cloud Database...")
            # Tell all active AI workers to reload their memory directly from the API
            for q in self.task_queues:
                if q: q.put({'type': 'reload'})
            
            self.worker_signals.status_updated.emit("Sync Complete")
            return True, "Sync signal sent successfully to AI workers."
        except Exception as e:
            self.worker_signals.status_updated.emit("Sync Error")
            print(f"[SYNC ERROR] {e}")
            return False, f"Sync error: {str(e)}"

    def set_active_count(self, count):
        """Updates the number of active cameras and workers."""
        if 1 <= count <= MAX_CAMS:
            old_count = self.active_cam_count
            self.active_cam_count = count
            print(f"[BACKEND] Scaling: {old_count} -> {count} active slots.")
            
            # Persist to config
            from config.cam_config_manager import CamConfigManager
            conf = CamConfigManager.load_config()
            conf["active_cam_count"] = count
            CamConfigManager.save_config(conf)
            
            # If scaling DOWN, stop extra resources
            if count < old_count:
                for i in range(count, min(old_count, MAX_CAMS)):
                    if self.workers[i]:
                        try:
                            self.workers[i].stop()
                            self.workers[i].terminate()
                        except: pass
                        self.workers[i] = None
                    if self.caps[i]:
                        try: self.caps[i].release()
                        except: pass
                        self.caps[i] = None
                    with self.frame_lock:
                        self.latest_frames[i] = None
            
            # If system is already running, we may need to start new cameras/workers
            if self.are_cameras_active and count > old_count:
                self.worker_signals.status_updated.emit(f"Scaling Up: {count} Cams...")
                # The user will likely trigger start_cameras_from_config anyway, 
                # but this ensures internal state is ready.

    def restart_cameras(self):
        """Stops and immediately restarts cameras using the latest config."""
        self.stop_cameras()
        # Small delay to ensure resources are released before re-opening
        threading.Timer(1.0, self.start_cameras_from_config).start()

    def start_cameras_from_config(self):
        """Loads camera sources from the JSON config and starts them."""
        from config.cam_config_manager import CamConfigManager
        conf = CamConfigManager.load_config()
        cams = conf.get("cams", [])
        
        # [GHOST WIPE] Kill all existing connections before starting new ones
        self.are_cameras_active = False 
        for i in range(MAX_CAMS):
            if self.caps[i]:
                try: self.caps[i].stop()
                except: pass
                self.caps[i] = None
                print(f"[RECOVERY] Killed active thread for CAM_{i+1}")
        
        # [ZERO-TOUCH PROPAGATION] Share IP from Cam 1 to others IMMEDIATELY
        valid_ip = next((c.get("ip") for c in cams if c.get("ip") and "." in str(c.get("ip"))), None)
        
        # [ZERO-TOUCH REACHABILITY] Verify if the saved IP is actually on THIS network
        is_reachable = False
        if valid_ip:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            try:
                # Check for RTSP Port 554 (most reliable NVR indicator)
                if sock.connect_ex((valid_ip, 554)) == 0:
                    is_reachable = True
            except: pass
            finally: sock.close()

        # If reachable, propagate to any empty slots
        if is_reachable and valid_ip:
            needs_save = False
            for cam in cams[:self.active_cam_count]:
                c_ip = str(cam.get("ip", ""))
                if not c_ip or "." not in c_ip or c_ip == "{ip}":
                    print(f"[ZERO-TOUCH] Auto-filling {cam.get('name')} with reachable NVR: {valid_ip}")
                    cam["ip"] = valid_ip
                    needs_save = True
            if needs_save:
                CamConfigManager.save_config(conf)
                # Re-load to ensure absolute data integrity
                conf = CamConfigManager.load_config()
                cams = conf.get("cams", [])
            
        # [MANUAL PRIORITY] If user-entered IP is reachable, skip discovery and GO LIVE
        if is_reachable and valid_ip:
            print(f"[ZERO-TOUCH] Verified manual IP {valid_ip} is active. Bypassing search...")
            # Proceed to start normally...
        
        # [FORCE RE-SCAN] Only trigger discovery if NO IP exists OR if the existing IP is dead
        elif not valid_ip or not is_reachable:
            if not is_reachable and valid_ip:
                print(f"[ZERO-TOUCH] {valid_ip} unreachable on this network. Triggering site-migration...")
                self._log_cctv_error(f"SITE-MIGRATION: Saved IP {valid_ip} is unreachable. Re-scanning...")
            
            if getattr(self, "is_scanning", False): return
            self.is_scanning = True
            
            # [CRITICAL] Stop the original start sequence immediately
            self.are_cameras_active = False 
            
            def auto_setup_task():
                try:
                    self.worker_signals.status_updated.emit("INITIALIZING: Autonomous Discovery...")
                    ips = self.broadcast_network_scan()
                    if ips:
                        ip = ips[0]
                        # [FIX] Force button into 'Starting' state during discovery
                        self.worker_signals.status_updated.emit("STARTING CAMERAS: Site Discovery...")
                        # [DEEP SCAN] Resolve {path} placeholder by probing the NVR
                        self.worker_signals.status_updated.emit("INITIALIZING: Identifying Brand...")
                        # Build a temporary test URL with the found IP and a default channel
                        # Add hard timeout to discovery strings
                        test_url = raw_template.replace("{ip}", ip).replace("{channel}", "1")
                        if "?" in test_url: test_url += "&timeout=5000000"
                        else: test_url += "?timeout=5000000"
                        
                        new_template_path, status = self.discover_brand_path(test_url, channel_hint="1")
                        
                        # [STICKY IP & PATH] Once found, force-update the config permanently
                        latest_conf = CamConfigManager.load_config()
                        if new_template_path:
                            # If we found a specific brand path, use the new template
                            latest_conf["rtsp_template"] = new_template_path
                        
                        # [SITE-MIGRATION] When moving to a new office, update ALL cameras to the new IP
                        # And AUTO-ASSIGN the first 6 active channels found
                        _, active_channels = self.enumerate_nvr_channels(latest_conf["rtsp_template"], existing_conf=latest_conf)
                        
                        current_cams = latest_conf.get("cams", [])
                        for i, c in enumerate(current_cams):
                            if i < self.active_cam_count:
                                old_ip = c.get("ip", "Unknown")
                                c["ip"] = ip
                                # Map to discovered live channels if available
                                if i < len(active_channels):
                                    new_ch = f"Ch {active_channels[i]}"
                                    print(f"[MIGRATION] Re-mapped {c.get('name')}: {old_ip} -> {ip} | {c.get('source')} -> {new_ch}")
                                    c["source"] = new_ch
                                else:
                                    print(f"[MIGRATION] Re-mapped {c.get('name')}: {old_ip} -> {ip} (No live channel found for this slot)")
                        
                        latest_conf["cams"] = current_cams
                        CamConfigManager.save_config(latest_conf)
                        
                        # Trigger immediate restart with new sticky config
                        self.start_cameras_from_config()
                finally: self.is_scanning = False
            import threading
            threading.Thread(target=auto_setup_task, daemon=True).start()
            return

        # [READY TO START] Build full RTSP links via Smart Router
        raw_template = conf.get("rtsp_template", "")
        sources = []
        for i, cam in enumerate(cams):
            # Pass the raw source string (e.g., 'Camera 23' or 'Ch 1') to the smart router
            ch_str = cam.get("source", f"Camera {i+1}")
            sources.append(ch_str)
            
            # [FIX] Respect the JSON config role, so "monitor" cams are skipped for recording
            if i < MAX_CAMS:
                self.cam_roles[i] = cam.get("role", "monitor").lower()

        print(f"[STARTUP] Powering on {len([s for s in sources if s])} camera streams (staggered)...")
        # [STAGGERED START] Avoid overwhelming the NVR by spacing out connections
        self.stop_cameras()
        for i, src in enumerate(sources[:self.active_cam_count]):
            if src is not None and src != "":
                self.start_single_camera(i, src)
                time.sleep(0.5) # 500ms gap
        
        self.are_cameras_active = True
        
        # Start the Recorder immediately so cameras don't miss footage
        from config.config import get_config
        self.record_video = get_config().get('record_video', False)
        if self.record_video and not self.recorder:
            from core.recorder import RecorderWorker
            rec_dir = get_config().get('temp_recordings_dir', 'data/temp_recordings')
            self.recorder = RecorderWorker(self.recording_queue, self.task_queues, rec_dir)
            self.recorder.start()

    def start_single_camera(self, i, src):
        """Initializes a single camera stream for the given index asynchronously."""
        if src is None or src == "": return
        
        # [NEW] Wrap in a background thread to prevent UI freezes during 2s handshake
        def _async_connect():
            try:
                self._start_single_camera_impl(i, src)
            except Exception as e:
                print(f"[CAMERA ERROR] Async connection failed: {e}")
                
        import threading
        threading.Thread(target=_async_connect, daemon=True).start()

    def _start_single_camera_impl(self, i, src):
        """Internal synchronous implementation of camera initialization."""
        
        from config.cam_config_manager import CamConfigManager
        conf = CamConfigManager.load_config()
        template = conf.get('rtsp_template', '')
        
        # [CLEAN SWITCHING] Stop the old camera in this slot before starting a new one
        # This prevents background threads from leaking and crashing the app.
        if self.caps[i] is not None:
            try:
                print(f"[CLEAN SWITCH] Releasing old handle for Slot {i+1} before reconnecting...")
                self.caps[i].release()
                self.caps[i] = None
            except Exception as e:
                print(f"[CLEAN SWITCH] Error releasing slot {i+1}: {e}")

        # [SMART RESOLUTION] Logic to build the final RTSP URL from source input
        final_src = src
        
        # --- CASE 1: Local Webcam Check ---
        import re
        if isinstance(src, str) and "Webcam" in src:
            digits = re.search(r'(\d+)', src)
            if digits:
                final_src = int(digits.group(1)) # Convert "Webcam 0" -> 0
                print(f"[CAMERA] Resolved to Local Webcam Index: {final_src}")
        
        # --- CASE 1.5: Direct IP Entry ---
        elif isinstance(src, str) and re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', src):
            ip = src
            cams_list = conf.get('cams', [])
            cam_info = cams_list[i] if i < len(cams_list) else {}
            cam_template = cam_info.get('template', template)
            if cam_template:
                final_src = cam_template.replace("{ip}", ip)
                if "{channel}" in final_src: final_src = final_src.replace("{channel}", "1")
                if "{path}" in final_src: final_src = final_src.replace("{path}", "Streaming/Channels/101")
            print(f"[CAMERA] Resolved to Direct IP: {ip}")

        # --- CASE 2: Network Camera Resolution (Channels) ---
        elif isinstance(src, str) and not src.startswith("rtsp://"):
            import re
            digits = re.search(r'(\d+)', src)
            if digits:
                cam_num = int(digits.group(1))
                
                cams_list = conf.get('cams', [])
                discovered_ips = conf.get('discovered_ips', [])
                
                # --- BRANCH A: SWARM MODE (Standalone IPs) ---
                if len(discovered_ips) > 1:
                    # Use specific IP from configuration for this slot
                    if 0 <= i < MAX_CAMS: cam_info = cams_list[i] if i < len(cams_list) else {}
                    ip = cam_info.get('ip') or (discovered_ips[0] if discovered_ips else '192.168.0.1')
                    cam_template = cam_info.get('template', template)
                    
                    if cam_template:
                        final_src = cam_template.replace("{ip}", ip)
                        # Handle leftover placeholders
                        if "{path}" in final_src:
                            final_src = final_src.replace("{path}", "Streaming/Channels/101")
                        final_src = final_src.replace("{channel}", "1")
                
                # --- BRANCH B: NVR MODE (Single IP, Multiple Channels) ---
                else:
                    ip = (cams_list[i].get('ip') or '192.168.0.1') if i < len(cams_list) else '192.168.0.1'
                    cam_template = cams_list[i].get('template', template)
                    if cam_template:
                        final_src = cam_template.replace("{ip}", ip).replace("{channel}", str(cam_num))
                        if "{path}" in final_src:
                            final_src = final_src.replace("{path}", f"Streaming/Channels/{cam_num}01")
        
        # Post-processing for RTSP strings
        if isinstance(final_src, str):
            # Ensure we don't have double slashes if {path} was empty
            final_src = final_src.replace("//", "/").replace("rtsp:/", "rtsp://")
            
            # [FORCE TIMEOUT] Inject hard timeout only if it's an RTSP string
            if final_src.startswith("rtsp"):
                if "?" in final_src: final_src += "&timeout=5000000"
                else: final_src += "?timeout=5000000"
        
        # [BULLETPROOF HANDSHAKE] Only verify reachable for network streams
        is_reachable = True # Assume reachable for webcams (integers)
        host, port = "Local", 0
        
        if isinstance(final_src, str) and "://" in final_src:
            is_reachable = False
            try:
                # Extract IP and Port for handshake
                parts = final_src.split("@")[-1].split("/")[0].split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 554
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                if sock.connect_ex((host, port)) == 0:
                    is_reachable = True
                sock.close()
            except: pass

        if is_reachable:
            self.caps[i] = ThreadedCamera(final_src)
        else:
            self.caps[i] = None
            msg = f"Handshake Failed: {host}:{port}. Skipping to avoid 30s hang."
            print(f"[RECOVERY] {msg}")
            self._log_cctv_error(f"START-ERROR: {msg}")
            self.worker_signals.camera_error.emit(i, f"Offline: {host}")
        
        # Masked logging for terminal
        log_src = str(final_src)
        if '@' in log_src:
            parts = log_src.rsplit('@', 1)
            if ':' in parts[0]:
                cred_parts = parts[0].split(':', 1)
                log_src = f"{cred_parts[0]}:****@{parts[1]}"
        
        print(f"[CAMERA] Connecting CAM_{i+1} to: {log_src}")
        self.worker_signals.status_updated.emit(f"Connecting CAM_{i+1}...")
        
        if self.caps[i] and not self.caps[i].isOpened():
            print(f"[ERROR] CAM_{i+1} failed to open.")
            self.worker_signals.camera_error.emit(i, f"Failed: {log_src}")

        # Dynamically spawn worker if AI detection is active
        if self.is_detection_enabled:
            self._start_single_worker(i)

    def broadcast_network_scan(self, progress_callback=None):
        """Scans the network for available IP cameras on subnets 0 and 1."""
        print("[SCAN] Starting Universal Camera Discovery...")
        import socket
        import threading
        
        found_ips = []
        threads = []
        
        def check_host(ip, port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0) # Increased for Wi-Fi stability
                if sock.connect_ex((ip, port)) == 0:
                    found_ips.append(ip)
                sock.close()
            except: pass

        # Scan the requested range 192.168.0.1 - 254
        # Optimization: Focus primarily on subnet 0 as requested, then subnet 1 as fallback
        for subnet in ["0", "1"]:
            for i in range(1, 255):
                ip = f"192.168.{subnet}.{i}"
                for port in [554, 8000]: # Standard RTSP and ONVIF/SDK ports
                    t = threading.Thread(target=check_host, args=(ip, port))
                    threads.append(t)
                    t.start()
                    
                    # Control concurrency to prevent socket exhaustion
                    if len(threads) >= 150:
                        for thread in threads: thread.join()
                        threads = []
                        if progress_callback: 
                            # Calculate percentage based on current IP/Subnet
                            progress = int(((int(subnet) * 255 + i) / 510) * 100)
                            progress_callback(progress)

        for thread in threads: 
            thread.join(timeout=0.1) # Be clean but fast
        
        # Deduplicate and sort
        unique_ips = sorted(list(set(found_ips)))
        print(f"[SCAN] Discovery complete. Found {len(unique_ips)} potential hosts: {unique_ips}")
        return unique_ips

    def discover_brand_path(self, raw_url, channel_hint="1", brand=None):
        """Tries common RTSP paths and ports to identify the camera brand automatically.
        
        Returns: (template_url, status) where status is 'OK', 'UNAUTHORIZED', or 'NOT_FOUND'.
        """
        self.worker_signals.status_updated.emit(f"Scanning Infrastructure (CH {channel_hint})...")
        import cv2
        import os
        import socket
        import urllib.parse
        
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000|stimeout;3000000"
        
        # ── Step 1: Parse the URL properly ──────────────────────────────
        try:
            last_at = raw_url.rfind('@')
            if last_at == -1:
                return None, "NOT_FOUND"
            
            cred_part = raw_url[:last_at]
            host_part = raw_url[last_at + 1:]
            
            cred_no_proto = cred_part.replace("rtsp://", "")
            colon_idx = cred_no_proto.index(':')
            user = cred_no_proto[:colon_idx]
            raw_pass = cred_no_proto[colon_idx + 1:]
            # [FIX] Prevent double-encoding. If the pass was already saved with %40 (@),
            # we must unquote it before quoting again for the RTSP link.
            clean_pass = urllib.parse.unquote(raw_pass)
            safe_pass = urllib.parse.quote(clean_pass, safe='')
            
            host_port_part = host_part.split('/')[0]
            if ':' in host_port_part:
                ip, orig_port = host_port_part.split(':', 1)
            else:
                ip = host_port_part
                orig_port = "554"
        except Exception as e:
            print(f"[DEEP-SCAN] URL parse error: {e}")
            return None, "NOT_FOUND"
 
        # ── Step 2: Define what to probe ────────────────────────────────
        ports = list(dict.fromkeys([orig_port, "554", "8554", "8000"]))
        
        test_paths = [
            ("CP Plus/Dahua",  f"cam/realmonitor?channel={channel_hint}&subtype=0"),
            ("Hikvision",      f"Streaming/Channels/{channel_hint}01"),
            ("Generic",        f"ch{channel_hint}/main"),
            ("Uniview",        f"video{channel_hint}"),
            ("Honeywell",      f"h264"),
            ("XMeye",          f"live/ch{channel_hint}"),
            ("ONVIF",          f"onvif{channel_hint}"),
        ]
        
        if brand:
            brand_norm = str(brand).lower()
            if "hikvision" in brand_norm:
                test_paths = [t for t in test_paths if "hik" in t[0].lower()]
            elif "cp plus" in brand_norm or "dahua" in brand_norm:
                test_paths = [t for t in test_paths if "cp" in t[0].lower() or "dah" in t[0].lower()]
            elif "eyematic" in brand_norm:
                test_paths = [t for t in test_paths if "generic" in t[0].lower()]
        
        best_guess = None
        
        print(f"[DEEP-SCAN] Target: {ip} | User: {user} | Ports: {ports}")
        
        # ── Step 3: Probe all combinations ──────────────────────────────
        for port in ports:
            # [FIX] Fast TCP pre-check: skip port entirely if nothing is listening.
            # This prevents the 30-second FFmpeg TCP connect timeout on closed ports.
            port_int = int(port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 seconds max
            try:
                result = sock.connect_ex((ip, port_int))
                sock.close()
                if result != 0:
                    print(f"[DEEP-SCAN] Port {port}: NOT OPEN (skipped)")
                    continue
                print(f"[DEEP-SCAN] Port {port}: OPEN — probing paths...")
            except:
                sock.close()
                print(f"[DEEP-SCAN] Port {port}: UNREACHABLE (skipped)")
                continue
            
            for brand_name, path in test_paths:
                test_url = f"rtsp://{user}:{safe_pass}@{ip}:{port}/{path}"
                print(f"  [{brand_name}] {path}...", end=" ")
                
                cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                opened = cap.isOpened()
                if opened:
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        # SAFE GENERIC PATH: Only replace the known placeholder in the suffix
                        generic_path = path.replace(channel_hint, "{channel}")
                        print(f"SUCCESS [OK]")
                        print(f"[DEEP-SCAN] Verified: {brand_name} on port {port}")
                        # Build a CLEAN template by combining the parts we verified
                        clean_template = f"rtsp://{user}:{safe_pass}@{ip}:{port}/{generic_path}"
                        return clean_template, "OK"
                    else:
                        print("Opened but no frames")
                else:
                    cap.release()
                    print("Rejected (401)")
                    if best_guess is None:
                        # Store the generic path suffix for later reconstruction
                        generic_path = path.replace(channel_hint, "{channel}")
                        best_guess = (generic_path, brand_name, port)
            
        # ── Step 4: Return best guess or failure ────────────────────────
        if best_guess:
            path_suffix, brand_name, port = best_guess
            # Build a CLEAN template by combining the parts we verified
            # This prevents nested 'double-URLs'
            clean_template = f"rtsp://{user}:{safe_pass}@{ip}:{port}/{path_suffix}"
            print(f"[DEEP-SCAN] No verified stream. Best guess: {brand_name} on port {port} (likely wrong password)")
            self.worker_signals.status_updated.emit(f"Auth Failed ({brand_name})")
            return clean_template, "UNAUTHORIZED"
        
        print(f"[DEEP-SCAN] No camera found at {ip}")
        return None, "NOT_FOUND"

    def enumerate_nvr_channels(self, template, max_channels=32, existing_conf=None):
        """Probes an NVR to find exactly how many channels are active.
        
        Returns: The count of active channels identified.
        """
        import cv2
        import os
        from config.cam_config_manager import CamConfigManager
        
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;2000000|stimeout;2000000"
        
        # [STABILITY] Dahua/CP-Plus NVRs need a moment to stabilize between probes
        time.sleep(1.0)
        
        active_count = 0
        self.worker_signals.status_updated.emit("Probing NVR Channels...")
        
        # We test channels sequentially. Most NVRs are 4, 8, 16, or 32.
        # We stop if we hit 3 consecutive empty channels to save time.
        consecutive_empty = 0
        highest_found = 0
        
        active_ids = []
        for ch in range(1, max_channels + 1):
            # [FIX] Ensure {path} is resolved during probe if still present
            current_template = template
            if "{path}" in current_template:
                current_template = current_template.replace("{path}", "cam/realmonitor?channel={channel}&subtype=0")
            
            test_url = current_template.replace("{channel}", str(ch))
            
            print(f"[CH-PROBE] Testing Channel {ch}...", end=" ")
            
            is_active = False
            # [BULLETPROOF HANDSHAKE] Only probe with CV2 if the port answers in 2s
            try:
                # Extract IP and Port from the template if possible, or use channel hint
                ip_part = test_url.split("@")[-1].split("/")[0].split(":")[0]
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                handshake_ok = (sock.connect_ex((ip_part, 554)) == 0)
                sock.close()
            except: handshake_ok = False

            if not handshake_ok:
                print("No Handshake (2s)")
            else:
                # [RETRY LOGIC] Try up to 2 times if the NVR is busy
                for attempt in range(2):
                    cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        cap.release()
                        if ret:
                            is_active = True
                            break
                    cap.release()
                    if attempt == 0:
                        time.sleep(0.5) # Quick breather before retry
            
            if is_active:
                print("LIVE ✓")
                active_ids.append(ch)
                highest_found = ch
                consecutive_empty = 0
            else:
                print("Offline")
                consecutive_empty += 1
            
            # [THROTTLING PROTECTION] Don't hammer the NVR API too fast
            time.sleep(0.3)
            
            if consecutive_empty >= 3 and ch > self.active_cam_count: # Always test at least as many as active
                break

        final_count = max(self.active_cam_count, highest_found) # Default to at least active count for UI spacing
        print(f"[CH-PROBE] Enumeration complete. Identified {final_count} active channels: {active_ids}")
        
        # Save to config for UI persistence
        conf = existing_conf if existing_conf else CamConfigManager.load_config()
        conf["num_channels"] = final_count
        CamConfigManager.save_config(conf)
        
        return final_count, active_ids

    def start_cameras(self, *sources):
        """Starts multiple cameras with a staggered delay to avoid NVR overload."""
        # Ensure we have exactly MAX_CAMS sources (pad with None if needed)
        full_sources = list(sources)
        while len(full_sources) < MAX_CAMS:
            full_sources.append(None)
            
        # Offload camera connection to a background thread to prevent GUI freezing
        threading.Thread(target=self._start_cameras_async, args=tuple(full_sources[:MAX_CAMS]), daemon=True).start()

    def _start_cameras_async(self, *sources):
        """Internal async loop to space out camera connections."""
        self.worker_signals.status_updated.emit("Connecting Cameras...")
        self.stop_cameras()
        
        for i, src in enumerate(sources):
            if i >= self.active_cam_count: break
            if src is not None and src != "":
                self.start_single_camera(i, src)
                time.sleep(0.5) # Stagger connections

        self.are_cameras_active = True
        
        # Start the Recorder immediately
        if self.record_video:
            self.start_recorder()

        active_count = sum(1 for c in self.caps if c is not None)
        self.worker_signals.status_updated.emit(f"CAMERAS ONLINE ({active_count} active)")

    def start_recorder(self):
        """Starts the background recorder process with access to all 6 camera task queues."""
        if not self.record_video:
            return
        if self.recorder and self.recorder.is_alive():
            return
            
        from config.config import get_config
        from core.recorder import RecorderWorker
        rec_dir = get_config().get('temp_recordings_dir', 'data/temp_recordings')
        
        # [ROBUST] Pass ALL 6 task queues to the recorder now. 
        # This allows it to handle handoffs for any camera swapped in at runtime.
        self.recorder = RecorderWorker(self.recording_queue, self.task_queues, rec_dir)
        self.recorder.start()
        print("[BACKEND] Video Recorder started successfully.")

    def stop_cameras(self):
        """Immediately halts all camera operations and resets UI states."""
        self.are_cameras_active = False
        self.worker_signals.status_updated.emit("CAMERAS OFFLINE")
        
        # Clear latest frames immediately to prevent "ghost" images on UI
        with self.frame_lock:
            self.latest_frames = [None] * MAX_CAMS
            
        # Stop Recorder
        if self.recorder:
            try:
                self.recorder.stop()
                if self.recorder.is_alive():
                    self.recorder.join(timeout=1.0)
                    if self.recorder.is_alive():
                        self.recorder.terminate()
            except: pass
            self.recorder = None
            
        # Release all camera handles
        for i in range(MAX_CAMS):
            if self.caps[i]:
                try:
                    self.caps[i].release()
                except: pass
                    
                self.caps[i] = None
                
        self.worker_signals.status_updated.emit("Cameras Offline")

    def register_user(self, name, department, person_id):
        if not self.caps[0]:
            return False, "Camera 1 not active."
            
        ret, frame = self.caps[0].read()
        if not ret or frame is None:
            return False, "Failed to capture frame."
            
        try:
            from core.face_recognition import FaceRecognitionHandler
            from config.config import get_config
            cfg = get_config()
            
            use_api = cfg.get('use_api', True)
            
            # Since we are removing database.database, we use APIClient or DatabaseManager.
            # If self.api_client is not initialized, try to initialize it.
            if not self.api_client:
                if use_api:
                    from core.api_client import APIClient
                    self.api_client = APIClient(
                        cfg.get('api_base_url'),
                        api_user=self.auth_email,
                        api_password=self.auth_pass,
                        token=self.auth_token
                    )
                else:
                    from database.database import DatabaseManager
                    self.api_client = DatabaseManager()
            
            face_handler = FaceRecognitionHandler(self.api_client)
            # Extract BOTH standard (512-dim) and mask (128-dim) encodings — V5 pipeline
            std_encoding, mask_encoding, msg = face_handler.extract_face_encodings(frame)
            if std_encoding is None: return False, msg
            
            if use_api:
                success, db_msg = self.api_client.add_person(
                    person_id, name, std_encoding, department=department,
                    mask_face_encoding=mask_encoding
                )
            else:
                success, db_msg = self.api_client.add_person(
                    person_id, name, std_encoding, 
                    mask_face_encoding=mask_encoding, department=department
                )

            if success:
                # Update active AI workers instantly
                for w in self.workers:
                    if w and hasattr(w, 'face_handler'):
                        w.face_handler.reload_face_encodings()
                
                msg = f"Successfully Registered: {name} (ID: {person_id})"
                return True, msg
            else:
                return False, f"Registration Error: {db_msg}"
        except Exception as e:
            return False, str(e)

    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / float(union + 1e-6)

    def _triage_detect(self, frame):
        if not hasattr(self, '_triage_backend'):
            from config.config import get_config
            conf = get_config()
            self._triage_backend = conf.get('triage_detection_backend', 'opencv_dnn')
            
        if self._triage_backend == 'yolov8':
            return self._triage_detect_yolo(frame)
        else:
            return self._triage_detect_opencv(frame)

    def _triage_detect_yolo(self, frame):
        if not hasattr(self, '_yolo_session'):
            self._yolo_session = None
        if not hasattr(self, '_yolo_body_session'):
            self._yolo_body_session = None
        if not hasattr(self, '_yolo_load_failed'):
            self._yolo_load_failed = False
        if not hasattr(self, '_yolo_body_load_failed'):
            self._yolo_body_load_failed = False

        import os
        from config.config import YOLO_MODEL_PATH, YOLO_BODY_MODEL_PATH, BASE_DIR, ASSET_DIR
        
        yolo_filename = os.path.basename(YOLO_MODEL_PATH)
        yolo_user_path = os.path.join(BASE_DIR, "data", "models", yolo_filename)
        yolo_bundle_path = os.path.join(ASSET_DIR, "data", "models", yolo_filename)
        
        yolo_path = yolo_user_path
        if not getattr(self, '_yolo_downloading', False):
            # Prioritize clean bundled model if present
            if os.path.exists(yolo_bundle_path) and os.path.getsize(yolo_bundle_path) >= 2000000:
                yolo_path = yolo_bundle_path
            elif os.path.exists(yolo_user_path) and os.path.getsize(yolo_user_path) < 2000000:
                try:
                    os.remove(yolo_user_path)
                except:
                    pass

        yolo_body_filename = os.path.basename(YOLO_BODY_MODEL_PATH)
        yolo_body_user_path = os.path.join(BASE_DIR, "data", "models", yolo_body_filename)
        yolo_body_bundle_path = os.path.join(ASSET_DIR, "data", "models", yolo_body_filename)
        
        yolo_body_path = yolo_body_user_path
        if not getattr(self, '_yolo_body_downloading', False):
            # Prioritize clean bundled model if present
            if os.path.exists(yolo_body_bundle_path) and os.path.getsize(yolo_body_bundle_path) >= 3000000:
                yolo_body_path = yolo_body_bundle_path
            elif os.path.exists(yolo_body_user_path) and os.path.getsize(yolo_body_user_path) < 3000000:
                try:
                    os.remove(yolo_body_user_path)
                except:
                    pass

        # Use full precision FP32 models for maximum detection accuracy (prevents missed snapshots)
        yolo_path = yolo_bundle_path if os.path.exists(yolo_bundle_path) else yolo_user_path
        yolo_body_path = yolo_body_bundle_path if os.path.exists(yolo_body_bundle_path) else yolo_body_user_path

        # 1. Initialize Face Session
        if not getattr(self, '_yolo_load_failed', False):
            if not hasattr(self, '_yolo_session') or self._yolo_session is None:
                if not os.path.exists(yolo_path):
                    if not getattr(self, '_yolo_downloading', False) and not getattr(self, '_yolo_download_failed', False):
                        self._yolo_downloading = True
                        def download_yolo():
                            try:
                                import urllib.request
                                print(f"[Backend] YOLOv8 model not found at {yolo_fp32}. Downloading in background...", flush=True)
                                os.makedirs(os.path.dirname(yolo_fp32), exist_ok=True)
                                url = 'https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx'
                                urllib.request.urlretrieve(url, yolo_fp32)
                                print("[Backend] YOLOv8 model downloaded successfully. Compiling INT8...", flush=True)
                                _quantize_onnx_model(yolo_fp32, yolo_int8)
                            except Exception as e:
                                print(f"[Backend] Error downloading YOLOv8: {e}", flush=True)
                                self._yolo_download_failed = True
                            finally:
                                self._yolo_downloading = False
                        import threading
                        threading.Thread(target=download_yolo, daemon=True).start()
                else:
                    if not getattr(self, '_yolo_downloading', False):
                        import onnxruntime as ort
                        from config.config import EXECUTION_PROVIDERS, ORT_INTRA_OP_NUM_THREADS, ORT_INTER_OP_NUM_THREADS
                        try:
                            opts = ort.SessionOptions()
                            opts.intra_op_num_threads = ORT_INTRA_OP_NUM_THREADS
                            opts.inter_op_num_threads = ORT_INTER_OP_NUM_THREADS
                            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            self._yolo_session = ort.InferenceSession(yolo_path, sess_options=opts, providers=EXECUTION_PROVIDERS)
                            print(f"[Backend] YOLOv8 Face Session loaded: {yolo_path}", flush=True)
                        except Exception as e:
                            print(f"[Backend] Error loading YOLOv8 session: {e}", flush=True)
                            self._yolo_session = None
                            self._yolo_load_failed = True

        # 2. Initialize Body Session
        if not getattr(self, '_yolo_body_load_failed', False):
            if not hasattr(self, '_yolo_body_session') or self._yolo_body_session is None:
                if not os.path.exists(yolo_body_path):
                    if not getattr(self, '_yolo_body_downloading', False) and not getattr(self, '_yolo_body_download_failed', False):
                        self._yolo_body_downloading = True
                        def download_yolo_body():
                            try:
                                import urllib.request
                                print(f"[Backend] YOLOv8 body model not found at {yolo_body_fp32}. Downloading in background...", flush=True)
                                os.makedirs(os.path.dirname(yolo_body_fp32), exist_ok=True)
                                url = 'https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx'
                                urllib.request.urlretrieve(url, yolo_body_fp32)
                                print("[Backend] YOLOv8 body model downloaded successfully. Compiling INT8...", flush=True)
                                _quantize_onnx_model(yolo_body_fp32, yolo_body_int8)
                            except Exception as e:
                                print(f"[Backend] Error downloading YOLOv8 body: {e}", flush=True)
                                self._yolo_body_download_failed = True
                            finally:
                                self._yolo_body_downloading = False
                        import threading
                        threading.Thread(target=download_yolo_body, daemon=True).start()
                else:
                    if not getattr(self, '_yolo_body_downloading', False):
                        import onnxruntime as ort
                        from config.config import EXECUTION_PROVIDERS, ORT_INTRA_OP_NUM_THREADS, ORT_INTER_OP_NUM_THREADS
                        try:
                            opts = ort.SessionOptions()
                            opts.intra_op_num_threads = ORT_INTRA_OP_NUM_THREADS
                            opts.inter_op_num_threads = ORT_INTER_OP_NUM_THREADS
                            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                            self._yolo_body_session = ort.InferenceSession(yolo_body_path, sess_options=opts, providers=EXECUTION_PROVIDERS)
                            print(f"[Backend] YOLOv8 Body Session loaded: {yolo_body_path}", flush=True)
                        except Exception as e:
                            print(f"[Backend] Error loading YOLOv8 body session: {e}", flush=True)
                            self._yolo_body_session = None
                            self._yolo_body_load_failed = True
                            self._yolo_body_session = None
                            self._yolo_body_load_failed = True
                
        if self._yolo_session is None and self._yolo_body_session is None:
            return []
            
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        input_size = 640
        resized = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)
        input_tensor = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0
        
        all_detections = [] # list of (box, score, is_face)
        
        # 1. Face Detection
        # 1. Face Detection (Re-enabled alongside Body Detection to guarantee complete snapshots)
        if self._yolo_session is not None:
            try:
                outputs = self._yolo_session.run(None, {'images': input_tensor})
                predictions = outputs[0][0].T # (8400, 5)
                
                from config.config import get_config
                conf_thresh = get_config().get('yolo_confidence_threshold', 0.25)
                keep_idx = predictions[:, 4] > conf_thresh
                filtered = predictions[keep_idx]
                
                for pred in filtered:
                    cx, cy, nw, nh, score = pred
                    x_scale = w / input_size
                    y_scale = h / input_size
                    
                    x1 = (cx - nw / 2) * x_scale
                    y1 = (cy - nh / 2) * y_scale
                    x2 = (cx + nw / 2) * x_scale
                    y2 = (cy + nh / 2) * y_scale
                    
                    all_detections.append(([x1, y1, x2, y2], float(score), True))
            except Exception as e:
                print(f"[Backend] YOLO Face inference error: {e}", flush=True)

        # 2. Body/Person Detection
        # [CPU-OPTIMIZATION] Re-enabled body detection at native 640x640 resolution to detect people far away in the doorway
        if self._yolo_body_session is not None:
            try:
                outputs_body = self._yolo_body_session.run(None, {'images': input_tensor})
                predictions_body = outputs_body[0][0].T # (8400, 84)
                
                from config.config import get_config
                body_conf_thresh = get_config().get('yolo_body_confidence_threshold', 0.35)
                
                keep_idx_body = predictions_body[:, 4] > body_conf_thresh
                filtered_body = predictions_body[keep_idx_body]
                
                for pred in filtered_body:
                    cx, cy, nw, nh = pred[0:4]
                    score = pred[4] # class 0 (person)
                    x_scale = w / input_size
                    y_scale = h / input_size
                    
                    x1 = (cx - nw / 2) * x_scale
                    y1 = (cy - nh / 2) * y_scale
                    x2 = (cx + nw / 2) * x_scale
                    y2 = (cy + nh / 2) * y_scale
                    
                    all_detections.append(([x1, y1, x2, y2], float(score), False))
            except Exception as e:
                print(f"[Backend] YOLO Body inference error: {e}", flush=True)
                
        if len(all_detections) == 0:
            return []
            
        # NMS
        boxes = np.array([d[0] for d in all_detections])
        scores = np.array([d[1] for d in all_detections])
        
        keep = []
        if len(boxes) > 0:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]
            
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
                iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
                
                inds = np.where(iou <= 0.45)[0]
                order = order[inds + 1]
                
        # Separate faces and bodies among the kept indices
        kept_faces = []
        kept_bodies = []
        for idx in keep:
            det = all_detections[idx]
            if det[2]: # is_face
                kept_faces.append(det)
            else:
                kept_bodies.append(det)
                
        # Prioritize body boxes over face boxes for tracking stability.
        # If a face is inside a body box, we track the body box (which is larger and more stable).
        # We only keep the face box if it does not belong to any detected body.
        final_boxes = []
        for body in kept_bodies:
            final_boxes.append(body[0])
            
        for face in kept_faces:
            face_box = face[0]
            fx1, fy1, fx2, fy2 = face_box
            fcx = (fx1 + fx2) / 2
            fcy = (fy1 + fy2) / 2
            
            inside_body = False
            for body in kept_bodies:
                body_box = body[0]
                bx1, by1, bx2, by2 = body_box
                if (bx1 <= fcx <= bx2) and (by1 <= fcy <= by2):
                    inside_body = True
                    break
                    
            if not inside_body:
                final_boxes.append(face_box)
                
        triage_boxes = []
        for box in final_boxes:
            bx1 = max(0, int(box[0]))
            by1 = max(0, int(box[1]))
            bx2 = min(w, int(box[2]))
            by2 = min(h, int(box[3]))
            triage_boxes.append([bx1, by1, bx2, by2])
            
        return triage_boxes

    def _triage_detect_opencv(self, frame):
        if not hasattr(self, '_triage_net'):
            import os
            import cv2
            proto = resource_path(os.path.join("data", "models", "deploy.prototxt"))
            model = resource_path(os.path.join("data", "models", "res10_300x300_ssd_iter_140000.caffemodel"))
            if os.path.exists(proto) and os.path.exists(model):
                self._triage_net = cv2.dnn.readNetFromCaffe(proto, model)
            else:
                self._triage_net = None
                
        all_detections = [] # list of (box, score, is_face)
        (h, w) = frame.shape[:2]
        
        # 1. Face Detection
        if self._triage_net is not None:
            try:
                import cv2
                import numpy as np
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self._triage_net.setInput(blob)
                detections = self._triage_net.forward()
                
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.3:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        all_detections.append((box.astype(int).tolist(), float(confidence), True))
            except Exception as e:
                print(f"[Backend] OpenCV DNN face detection error: {e}", flush=True)

        # 2. Body/Person Detection using HOG fallback
        try:
            if not hasattr(self, '_triage_hog'):
                import cv2
                self._triage_hog = cv2.HOGDescriptor()
                self._triage_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Resize frame for faster HOG detection if large
            scale = 1.0
            if max(h, w) > 400:
                scale = 400 / max(h, w)
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                small_frame = frame
                
            boxes, weights = self._triage_hog.detectMultiScale(
                small_frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            
            for idx, (bx, by, bw, bh) in enumerate(boxes):
                x1 = int(bx / scale)
                y1 = int(by / scale)
                x2 = int((bx + bw) / scale)
                y2 = int((by + bh) / scale)
                weight = float(weights[idx]) if idx < len(weights) else 0.5
                all_detections.append(([x1, y1, x2, y2], weight, False))
        except Exception as e:
            print(f"[Backend] OpenCV HOG body detection error: {e}", flush=True)

        if len(all_detections) == 0:
            return []
            
        # NMS
        import numpy as np
        boxes_arr = np.array([d[0] for d in all_detections])
        scores_arr = np.array([d[1] for d in all_detections])
        
        keep = []
        if len(boxes_arr) > 0:
            x1 = boxes_arr[:, 0]
            y1 = boxes_arr[:, 1]
            x2 = boxes_arr[:, 2]
            y2 = boxes_arr[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores_arr.argsort()[::-1]
            
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                
                intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
                iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
                
                inds = np.where(iou <= 0.45)[0]
                order = order[inds + 1]
                
        kept_faces = []
        kept_bodies = []
        for idx in keep:
            det = all_detections[idx]
            if det[2]: # is_face
                kept_faces.append(det)
            else:
                kept_bodies.append(det)
                
        # Filter out bodies that contain a face
        final_boxes = []
        for face in kept_faces:
            final_boxes.append(face[0])
            
        for body in kept_bodies:
            body_box = body[0]
            bx1, by1, bx2, by2 = body_box
            
            has_face = False
            for face in kept_faces:
                face_box = face[0]
                fx1, fy1, fx2, fy2 = face_box
                fcx = (fx1 + fx2) / 2
                fcy = (fy1 + fy2) / 2
                if (bx1 <= fcx <= bx2) and (by1 <= fcy <= by2):
                    has_face = True
                    break
                    
            if not has_face:
                final_boxes.append(body_box)
                
        triage_boxes = []
        for box in final_boxes:
            bx1 = max(0, int(box[0]))
            by1 = max(0, int(box[1]))
            bx2 = min(w, int(box[2]))
            by2 = min(h, int(box[3]))
            triage_boxes.append([bx1, by1, bx2, by2])
            
        return triage_boxes

    def _save_and_queue_snapshot(self, cam_index, frame, track=None):
        """Dispatch snapshot saving asynchronously using the frame copy pre-captured at detection time."""
        import threading
        track_copy = track.copy() if track else None
        frame_copy = frame.copy() if frame is not None else None
        
        threading.Thread(
            target=self._save_and_queue_snapshot_async,
            args=(cam_index, frame_copy, track_copy),
            daemon=True
        ).start()


    def _save_and_queue_snapshot_async(self, cam_index, frame, track=None):
        try:
            import os
            import cv2
            import time
            from config.config import BASE_DIR
            
            pending_dir = os.path.join(BASE_DIR, "data", "pending_snapshots")
            os.makedirs(pending_dir, exist_ok=True)
            
            role = self.cam_roles[cam_index]
            
            # [MOTION VECTOR ANALYSIS] Classify IN/OUT dynamically if history exists
            # We now support vector analysis for histories >= 2 to detect direction faster.
            if track and 'history' in track and len(track['history']) >= 2:
                history = track['history']
                first_cx, first_cy = history[0]
                last_cx, last_cy = history[-1]
                dx = last_cx - first_cx
                dy = last_cy - first_cy
                
                rules = self.cam_direction_rules[cam_index] if cam_index < len(self.cam_direction_rules) else {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}
                thresh = 5.0
                
                detected_direction = None
                # Prioritize the axis of dominant movement (highest displacement) to increase accuracy
                if abs(dx) >= abs(dy):
                    # Horizontal is dominant
                    if dx > thresh and rules.get("right") != "ignore":
                        detected_direction = rules["right"]
                    elif dx < -thresh and rules.get("left") != "ignore":
                        detected_direction = rules["left"]
                    
                    # Fallback to vertical if horizontal didn't match any active rule
                    if not detected_direction:
                        if dy > thresh and rules.get("down") != "ignore":
                            detected_direction = rules["down"]
                        elif dy < -thresh and rules.get("up") != "ignore":
                            detected_direction = rules["up"]
                else:
                    # Vertical is dominant
                    if dy > thresh and rules.get("down") != "ignore":
                        detected_direction = rules["down"]
                    elif dy < -thresh and rules.get("up") != "ignore":
                        detected_direction = rules["up"]
                        
                    # Fallback to horizontal if vertical didn't match any active rule
                    if not detected_direction:
                        if dx > thresh and rules.get("right") != "ignore":
                            detected_direction = rules["right"]
                        elif dx < -thresh and rules.get("left") != "ignore":
                            detected_direction = rules["left"]
                
                if detected_direction:
                    # role = detected_direction
                    print(f"[MOTION ENGINE] CAM_{cam_index+1} Track movement dx={dx:.1f}, dy={dy:.1f} -> Dynamic Direction: {detected_direction.upper()} (Ignored: using configured default role: {role.upper()})", flush=True)
                    
            ts = int(time.time() * 1000)
            filename = f"cam_{cam_index}_{role}_{ts}.jpg"
            filepath = os.path.join(pending_dir, filename)
            
            from config.config import get_config
            use_high_res = get_config().get('use_high_res_snapshots', False)
            
            high_res_frame = None
            if use_high_res:
                # Asynchronously retrieve the high-resolution frame in the background thread
                # so we don't block the main detection loop!
                try:
                    cap = self.caps[cam_index]
                    if cap and hasattr(cap, 'read_high_res'):
                        ret_hr, hr_frame = cap.read_high_res()
                        if ret_hr and hr_frame is not None:
                            high_res_frame = hr_frame.copy()
                except Exception as hr_err:
                    print(f"[Backend Warning] Async high-res retrieve failed: {hr_err}", flush=True)
                
            if high_res_frame is None:
                # Use the sub-stream frame passed from the main loop (instant, no network/decoding delay)
                high_res_frame = frame
                
            self._log_debug(f"Attempting to write snapshot: {filename}")
            if cv2.imwrite(filepath, high_res_frame):
                self._log_debug(f"Successfully Saved Snapshot: {filename}")
                
                # Check if the AI worker is active and process is alive
                worker = self.workers[cam_index]
                if worker and worker.is_alive():
                    try:
                        self.task_queues[cam_index].put_nowait(filepath)
                        abs_path = os.path.abspath(filepath)
                        mtime = os.path.getmtime(filepath)
                        self.queued_snapshots[abs_path] = mtime
                        self._log_debug(f"Queued Snapshot directly to active Worker {cam_index + 1}: {filename}")
                        print(f"[Backend] Saved & Queued Snapshot: {filename}", flush=True)
                    except Exception as queue_err:
                        self._log_debug(f"Failed to queue snapshot directly (Queue full?): {queue_err}")
                else:
                    self._log_debug(f"Saved snapshot to disk (AI Offline): {filename}")
                    print(f"[Backend] Saved Snapshot (AI Offline): {filename}", flush=True)
            else:
                self._log_debug(f"Failed to write snapshot file: {filepath}")
        except Exception as e:
            self._log_debug(f"Error saving/queueing snapshot: {e}")
            print(f"[Backend] Error saving/queueing snapshot: {e}", flush=True)

    def scan_pending_snapshots(self):
        try:
            import os
            import re
            from config.config import BASE_DIR
            
            pending_dir = os.path.join(BASE_DIR, "data", "pending_snapshots")
            if not os.path.exists(pending_dir):
                return
                
            # Clean up the queued_snapshots dictionary for files that no longer exist
            self.queued_snapshots = {p: mt for p, mt in self.queued_snapshots.items() if os.path.exists(p)}
            
            valid_exts = ('.jpg', '.jpeg', '.png')
            for fname in os.listdir(pending_dir):
                if not fname.lower().endswith(valid_exts):
                    continue
                    
                filepath = os.path.join(pending_dir, fname)
                abs_path = os.path.abspath(filepath)
                
                # Check modification time
                try:
                    mtime = os.path.getmtime(filepath)
                except Exception:
                    continue
                
                # Skip if already queued, modification time is unchanged, AND target worker is still alive
                if abs_path in self.queued_snapshots:
                    entry = self.queued_snapshots[abs_path]
                    saved_mtime, queued_worker_idx = entry if isinstance(entry, tuple) else (entry, None)
                    
                    if saved_mtime == mtime:
                        # Determine if the target worker is still alive
                        target_idx = queued_worker_idx
                        if target_idx is None:
                            match = re.search(r'cam_(\d+)_', fname)
                            target_idx = int(match.group(1)) if match else 0
                            
                        if 0 <= target_idx < MAX_CAMS:
                            w = self.workers[target_idx]
                            if w and w.is_alive():
                                continue
                    
                # Determine camera index (defaults to 0, which is 'entrance'/IN)
                cam_index = 0
                match = re.search(r'cam_(\d+)_', fname)
                if match:
                    cam_index = int(match.group(1))
                    
                if 0 <= cam_index < MAX_CAMS:
                    worker = self.workers[cam_index]
                    if worker and worker.is_alive():
                        # Queue snapshot to the specific worker process
                        self.task_queues[cam_index].put_nowait(filepath)
                        self.queued_snapshots[abs_path] = (mtime, cam_index)
                        self._log_debug(f"Queued snapshot to Worker {cam_index + 1}: {fname}")
                    else:
                        # Find ANY active worker to process the pending snapshot
                        active_worker_index = -1
                        for idx, w in enumerate(self.workers):
                            if w and w.is_alive():
                                active_worker_index = idx
                                break
                        
                        if active_worker_index != -1:
                            self.task_queues[active_worker_index].put_nowait(filepath)
                            self.queued_snapshots[abs_path] = (mtime, active_worker_index)
                            self._log_debug(f"Queued snapshot for CAM_{cam_index+1} (Monitor/Offline) to active Worker {active_worker_index + 1}: {fname}")
        except Exception as e:
            self._log_debug(f"Error scanning pending snapshots: {e}")

    def sync_offline_records(self):
        """Attempts to synchronize cached offline attendance records and raw logs with the remote server."""
        from config.config import get_config
        use_api = get_config().get('use_api', True)
        if not use_api or not self.api_client:
            return

        try:
            from database.offline_storage import OfflineStorage
            storage = OfflineStorage()
            
            # 1. Sync Raw Logs
            raw_logs = storage.get_pending_raw_logs()
            if raw_logs:
                print(f"[SYNC] Found {len(raw_logs)} offline raw logs to synchronize.", flush=True)
            for row in raw_logs:
                row_id, pid, name, timestamp_str, snapshot_path, event_type = row
                try:
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    # log_raw_detection doesn't return success flag, but throws exception on network failure
                    self.api_client.log_raw_detection(pid, name, timestamp=dt, snapshot_path=snapshot_path, event_type=event_type)
                    storage.delete_raw_log(row_id)
                    print(f"[SYNC] Successfully synced raw log for {name} ({pid})", flush=True)
                except Exception as e:
                    print(f"[SYNC ERROR] Failed to sync raw log for {name} ({pid}): {e}", flush=True)
                    return # Stop syncing on connection failure

            # 2. Sync Attendance Records
            attendance_records = storage.get_pending_attendance()
            if attendance_records:
                print(f"[SYNC] Found {len(attendance_records)} offline attendance records to synchronize.", flush=True)
            for row in attendance_records:
                row_id, pid, timestamp_str, snapshot_path, event_type = row
                try:
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    success, msg = self.api_client.mark_attendance(pid, timestamp=dt, snapshot_path=snapshot_path, event_type=event_type)
                    if success:
                        storage.delete_attendance(row_id)
                        print(f"[SYNC] Successfully synced attendance for PI: {pid}", flush=True)
                    else:
                        print(f"[SYNC ERROR] Server rejected attendance for PI {pid}: {msg}", flush=True)
                        if "API Error" in msg:
                            import re
                            m = re.search(r'API Error (\d+):', msg)
                            if m:
                                status_code = int(m.group(1))
                                if status_code >= 500 or status_code in (401, 403, 429):
                                    print(f"[SYNC] Temporary error ({status_code}). Keeping record in offline storage for retry.")
                                    return
                            elif any(x in msg for x in ("500", "502", "503", "504")) or "<html" in msg.lower():
                                print("[SYNC] Gateway/Server HTML error detected. Keeping record in offline storage for retry.")
                                return
                            # Server replied and explicitly rejected (e.g. 400 Bad Request). Delete to prevent getting stuck.
                            storage.delete_attendance(row_id)
                except Exception as e:
                    print(f"[SYNC ERROR] Failed to sync attendance for PI {pid}: {e}", flush=True)
                    return # Stop syncing on connection failure
                    
        except Exception as e:
            print(f"[SYNC ERROR] General error during offline sync: {e}", flush=True)

    def run(self):
        self.is_running = True
        self.stop_requested = False
        
        self.worker_signals.status_updated.emit("Backend Ready")
        
        # Initialize frame counters and tracked faces for all cams dynamically
        self.frame_counters = [0] * MAX_CAMS
        self.tracked_faces = {k: [] for k in range(MAX_CAMS)}
        self.last_folder_scan = 0
        self._cameras_active_since = 0  # Timestamp when cameras first became active
        self._ai_auto_start_attempted = False  # Guard to only auto-start once
        
        while not self.stop_requested:
            # 1. Main Capture Loop (Centralized Heartbeat)
            if self.are_cameras_active:
                # First, read and store all camera frames (takes 0ms)
                active_cams = []
                for i in range(self.active_cam_count):
                    if i < MAX_CAMS and self.caps[i] and self.caps[i].isOpened():
                        ret, frame = self.caps[i].read()
                        if ret and frame is not None:
                            self.latest_frames[i] = frame
                            
                            # Feed the Recorder ONLY if camera is not in "Monitor" mode and recording is enabled
                            if self.record_video and self.recorder and not self.recording_queue.full():
                                if str(self.cam_roles[i]).lower() != 'monitor':
                                    try:
                                        self.recording_queue.put_nowait((i, frame, time.time()))
                                        if int(time.time()) % 10 == 0 and i == 0:
                                            print(f"[BACKEND] Dispatching frames to recorder...")
                                    except: pass
                                    
                            if str(self.cam_roles[i]).lower() != 'monitor':
                                self.frame_counters[i] += 1
                                if self.frame_counters[i] % self.process_every_n_frames == 0:
                                    active_cams.append((i, frame))

                # Run triage detections concurrently in the ThreadPoolExecutor (releases GIL)
                triage_results = {}
                if active_cams:
                    if not hasattr(self, '_triage_executor'):
                        from concurrent.futures import ThreadPoolExecutor
                        self._triage_executor = ThreadPoolExecutor(max_workers=MAX_CAMS)
                    
                    def run_detect(cam_idx, cam_frame):
                        try:
                            return cam_idx, self._triage_detect(cam_frame)
                        except Exception as e:
                            print(f"[Backend Error] Parallel triage failed on Cam {cam_idx}: {e}", flush=True)
                            return cam_idx, []
                            
                    futures = [self._triage_executor.submit(run_detect, idx, f) for idx, f in active_cams]
                    for fut in futures:
                        try:
                            idx, boxes = fut.result()
                            triage_results[idx] = boxes
                        except Exception as e:
                            print(f"[Backend Error] Future resolution error: {e}", flush=True)

                # Process results and update trackers sequentially in the main backend thread
                for i, frame in active_cams:
                    if i in triage_results:
                        boxes = triage_results[i]
                        try:
                            # 2. Update tracker and determine if snapshot is needed
                            now = time.time()
                            current_tracks = self.tracked_faces[i]
                            
                            # Clean up old tracks (not seen in last 1.5s).
                            # Fallback: if a track is discarded but was never snapshotted, capture it now!
                            for t in current_tracks:
                                if now - t['last_seen'] >= 1.5:
                                    if t.get('last_snapshot', 0) == 0:
                                        snapshot_frame = t.get('first_snapshot_frame')
                                        if snapshot_frame is None:
                                            snapshot_frame = t.get('last_frame', frame)
                                        self._save_and_queue_snapshot(i, snapshot_frame, track=t)
                            current_tracks = [t for t in current_tracks if now - t['last_seen'] < 1.5]
                            
                            for box in boxes:
                                # Compute box center
                                bx1, by1, bx2, by2 = box
                                cx = (bx1 + bx2) / 2.0
                                cy = (by1 + by2) / 2.0

                                # Find best matching track
                                best_track = None
                                best_iou = 0.0
                                for track in current_tracks:
                                    iou = self._calculate_iou(box, track['box'])
                                    if iou > 0.1 and iou > best_iou:  # Lowered IoU threshold from 0.3 to 0.1 for high-speed tracking
                                        best_iou = iou
                                        best_track = track
                                
                                # Centroid distance fallback for fast-moving targets when overlap (IoU) is zero
                                if not best_track:
                                    best_dist = float('inf')
                                    # Allow matching within 40% of the maximum frame dimension for fast targets
                                    max_allowable_dist = max(frame.shape[1], frame.shape[0]) * 0.40
                                    for track in current_tracks:
                                        # Only match via centroid if the track was seen very recently (within 0.25s)
                                        # to prevent new entries from matching old/stale tracks of different people
                                        if now - track['last_seen'] < 0.25:
                                            tb = track['box']
                                            tcx = (tb[0] + tb[2]) / 2.0
                                            tcy = (tb[1] + tb[3]) / 2.0
                                            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                                            if dist < max_allowable_dist and dist < best_dist:
                                                best_dist = dist
                                                best_track = track

                                if best_track:
                                    # Update existing track
                                    best_track['box'] = box
                                    best_track['last_seen'] = now
                                    best_track['last_frame'] = frame.copy()
                                    if 'history' not in best_track:
                                        best_track['history'] = []
                                    best_track['history'].append((cx, cy))
                                    if len(best_track['history']) > 30:
                                        best_track['history'].pop(0)
                                        
                                    # Trigger first snapshot as soon as we have enough history (>= 2 points) to decide direction,
                                    # or trigger subsequent snapshots every 0.4s.
                                    is_first_snapshot = (best_track.get('last_snapshot', 0) == 0)
                                    has_enough_history = (len(best_track['history']) >= 2)
                                    if (is_first_snapshot and has_enough_history) or (not is_first_snapshot and now - best_track.get('last_snapshot', 0) >= 0.4):
                                        best_track['last_snapshot'] = now
                                        self._save_and_queue_snapshot(i, frame.copy(), track=best_track)
                                else:
                                    # New track! Start with last_snapshot = 0 to wait for the 2nd history point in the next frame (33ms delay)
                                    new_track = {
                                        'box': box,
                                        'last_seen': now,
                                        'last_snapshot': 0,
                                        'history': [(cx, cy)],
                                        'last_frame': frame.copy(),
                                        'first_snapshot_frame': None
                                    }
                                    current_tracks.append(new_track)
                                    
                            self.tracked_faces[i] = current_tracks
                        except Exception as e:
                            print(f"[BACKEND ERROR] Triage/Queue Error on Cam {i}: {e}", flush=True)

            # [SELF-HEALING] Auto-start AI workers if cameras are active but workers aren't running
            # This guarantees workers start even if the QTimer-based start_detection fails
            now = time.time()
            if self.are_cameras_active:
                if self._cameras_active_since == 0:
                    self._cameras_active_since = now
                    self._ai_auto_start_attempted = False
                
                # After 5 seconds of cameras being active, check if AI workers are running
                if not self._ai_auto_start_attempted and (now - self._cameras_active_since) >= 5.0:
                    active_workers = sum(1 for w in self.workers if w and w.is_alive())
                    if active_workers == 0:
                        print(f"[SELF-HEALING] Cameras active for {now - self._cameras_active_since:.1f}s but 0 AI workers running. Auto-starting detection...", flush=True)
                        self._ai_auto_start_attempted = True
                        self.is_detection_enabled = True
                        if not hasattr(self, 'init_lock') or self.init_lock is None:
                            self.init_lock = multiprocessing.Lock()
                        threading.Thread(target=self._start_detection_async, daemon=True).start()
                    else:
                        self._ai_auto_start_attempted = True  # Workers are already running, no need to check again
            else:
                self._cameras_active_since = 0
                self._ai_auto_start_attempted = False

            # 2. Folder Scanning for Offline Snapshot Detection
            if now - self.last_folder_scan >= 2.0:
                self.last_folder_scan = now
                if self.is_detection_enabled:
                    self.scan_pending_snapshots()
            
            # Offline Cache Synchronization Check
            if now - self.last_offline_sync >= 15.0:
                self.last_offline_sync = now
                if self.is_detection_enabled and self.api_client:
                    threading.Thread(target=self.sync_offline_records, daemon=True).start()
            
            # 2. Process AI Results
            try:
                while True:
                    try:
                        result = self.result_queue.get_nowait()
                        
                        try:
                            with open("backend_queue.log", "a") as f:
                                f.write(f"[{datetime.now()}] RCVD: {result.get('type')} | Worker: {result.get('worker')}\n")
                        except: pass
                        
                        if result.get('type') == 'stats':
                            total = result.get('total', 0)
                            self.worker_signals.stats_updated.emit(result)
                            
                            # [WARNING] Alert when recognition is blind
                            if total == 0:
                                warning = {
                                    'type': 'warning',
                                    'message': "AI IS BLIND: 0 Registered faces found. Recognition will not occur.",
                                    'worker': result.get('worker')
                                }
                                self.worker_signals.detection_occurred.emit(warning)
                                print(f"[CRITICAL] Worker {result.get('worker')} is blind (0 faces).")
                        else:
                            self.worker_signals.detection_occurred.emit(result)
                    except Exception:
                        break
            except Exception as e:
                print(f"[BACKEND ERROR] {e}")

            time.sleep(0.05)
            
        self.cleanup()
        self.is_running = False
        self.worker_signals.status_updated.emit("System Offline")

    def start_detection(self):
        """Spawns a thread to initialize AI Workers for any active cameras missing one."""
        print(f"[BACKEND] start_detection() called. is_detection_enabled={self.is_detection_enabled}, active_cams={self.active_cam_count}", flush=True)
        if getattr(self, '_is_starting_detection', False):
            print("[BACKEND] Detection is already starting. Skipping duplicate start.", flush=True)
            return
        if self.is_detection_enabled and any(w and w.is_alive() for w in self.workers):
            print("[BACKEND] Detection already running. Skipping duplicate start.", flush=True)
            return
        self.is_detection_enabled = True
        self._is_starting_detection = True
        if not hasattr(self, 'init_lock') or self.init_lock is None:
            self.init_lock = multiprocessing.Lock()
        threading.Thread(target=self._start_detection_async, daemon=True).start()

    def _start_single_worker(self, i):
        """Helper to start/restart a worker dynamically for a single active camera slot."""
        if not self.is_detection_enabled:
            return
        if i >= self.active_cam_count:
            return
            
        role = self.cam_roles[i]
        if str(role).lower() == 'monitor':
            return
            
        # Only start if worker slot is empty or not alive
        if self.workers[i] is None or not self.workers[i].is_alive():
            if not hasattr(self, 'init_lock') or self.init_lock is None:
                self.init_lock = multiprocessing.Lock()
                
            event_type = 'IN' if role == 'entrance' else ('OUT' if role == 'exit' else None)
            
            self.workers[i] = AttendanceWorker(
                self.task_queues[i], self.result_queue,
                worker_id=i + 1, assigned_cam_index=i,
                api_user=self.auth_email, api_pass=self.auth_pass,
                init_lock=self.init_lock,
                event_type=event_type,
                other_queues=self.task_queues
            )
            if self.caps[i] is not None:
                print(f"[BACKEND] Started AI Worker {i+1} dynamically for Camera {i} | Role: {role.upper()}")
            else:
                print(f"[BACKEND] Started Backlog Assistant {i+1} dynamically for Camera {i} | Role: {role.upper()}")
                
            self.workers[i].start()

    def _start_detection_async(self):
        try:
            print(f"[BACKEND] _start_detection_async: Starting AI workers for {self.active_cam_count} cameras...", flush=True)
            self.worker_signals.status_updated.emit("Starting AI (Staggered Startup)...")

            # [FULL-FORCE] Always ensure all active slots have an active worker
            for i in range(self.active_cam_count):
                # Only start if worker slot is empty or the process has died
                if self.workers[i] is None or not self.workers[i].is_alive():
                    role = self.cam_roles[i]
                    event_type = 'IN' if role == 'entrance' else ('OUT' if role == 'exit' else None)
                    
                    print(f"[BACKEND] Creating AttendanceWorker {i+1} (role={role}, event_type={event_type}, auth={self.auth_email is not None})...", flush=True)
                    self.workers[i] = AttendanceWorker(
                        self.task_queues[i], self.result_queue,
                        worker_id=i + 1, assigned_cam_index=i,
                        api_user=self.auth_email, api_pass=self.auth_pass,
                        init_lock=self.init_lock,
                        event_type=event_type,
                        other_queues=self.task_queues
                    )
                    if self.caps[i] is not None:
                        print(f"[BACKEND] Started AI Worker {i+1} for Camera {i} | Role: {role.upper()}", flush=True)
                    else:
                        # [BACKLOG ASSISTANT] Slot is empty? Use it to process pending recordings faster
                        print(f"[BACKEND] Started Backlog Assistant {i+1} for Camera {i} | Role: {role.upper()}", flush=True)
                    
                    self.workers[i].start()
                    print(f"[BACKEND] Worker {i+1} process started (pid={self.workers[i].pid})", flush=True)

            active_workers = sum(1 for w in self.workers if w and w.is_alive())
            print(f"[BACKEND] All workers launched. Active: {active_workers}/{self.active_cam_count}", flush=True)
            self.worker_signals.status_updated.emit(f"Detection Active ({active_workers} AI workers)")
        except Exception as e:
            print(f"CRITICAL: AI Startup Failed! {e}")
            import traceback
            traceback.print_exc()
            self.worker_signals.status_updated.emit("System Offline (Error)")
        finally:
            self._is_starting_detection = False

    def stop_detection(self, wait=False):
        self.is_detection_enabled = False
        for i, w in enumerate(self.workers):
            if w:
                try:
                    w.stop()
                    # Wait briefly then kill if still alive
                    if w.is_alive():
                        w.join(timeout=1.0)
                        if w.is_alive():
                            w.terminate()
                except: pass
                self.workers[i] = None
                
        # Note: Recorder is explicitly NOT stopped here. It runs as long as cameras run.
        self.worker_signals.status_updated.emit("Detection Stopped")

    def stop_system(self):
        self.stop_requested = True
        self.stop_detection(wait=True)
        self.stop_cameras()
        
        # [HANG FIX] Cancel Queue join threads to prevent Python from blocking indefinitely on exit
        try:
            self.result_queue.cancel_join_thread()
            self.recording_queue.cancel_join_thread()
            for q in self.task_queues:
                if q: q.cancel_join_thread()
        except: pass

    def cleanup(self):
        self.stop_cameras()
        
    def set_cam_role(self, i, role):
        """Update the role of a camera slot (Entrance, Exit, Monitor) at runtime."""
        if 0 <= i < MAX_CAMS:
            self.cam_roles[i] = role
            print(f"[BACKEND] Set CAM_{i+1} Role: {role.upper()}")
            
            # [PERSISTENCE] Save role to config
            try:
                from config.cam_config_manager import CamConfigManager
                conf = CamConfigManager.load_config()
                if i < len(conf.get('cams', [])):
                    conf['cams'][i]['role'] = role
                    CamConfigManager.save_config(conf)
            except Exception as e:
                print(f"[BACKEND] Error saving config role: {e}")

            # If AI is running, update the corresponding worker's behavior
            if self.workers[i]:
                event_type = 'IN' if role == 'entrance' else ('OUT' if role == 'exit' else None)
                # MUST inform the worker process
                if self.task_queues[i]:
                    try:
                        self.task_queues[i].put({'type': 'update_role', 'event_type': event_type})
                    except: pass
                
    def set_cam_roi(self, i, roi):
        """Update and persist the Region of Interest (ROI) for a camera slot."""
        if 0 <= i < MAX_CAMS:
            self.cam_rois[i] = roi
            print(f"[BACKEND] Set CAM_{i+1} ROI: {roi}")
            
            # [PERSISTENCE] Save ROI to config
            try:
                from config.cam_config_manager import CamConfigManager
                conf = CamConfigManager.load_config()
                if i < len(conf.get('cams', [])):
                    conf['cams'][i]['roi'] = roi
                    CamConfigManager.save_config(conf)
            except Exception as e:
                print(f"[BACKEND] Error saving config ROI: {e}")
                
    def swap_camera_source(self, i, source_text):
        """Stop the existing camera in slot `i` and immediately start the new requested source."""
        if not (0 <= i < MAX_CAMS): return
        
        print(f"[HOT-SWAP] Changing CAM_{i+1} Source to: {source_text}")
        
        # PERSISTENCE: Update the configuration so it survives restarts
        try:
            from config.cam_config_manager import CamConfigManager
            conf = CamConfigManager.load_config()
            if i < len(conf['cams']):
                # Clean whitespace
                source_text = source_text.strip()
                conf['cams'][i]['source'] = source_text
                
                # [NEW] If the source looks like an IP, update the 'ip' field too for Swarm Mode liveness
                import re
                if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', source_text):
                    print(f"[HOT-SWAP] Manual IP Override detected for Slot {i+1}: {source_text}")
                    conf['cams'][i]['ip'] = source_text
                
                CamConfigManager.save_config(conf)
        except Exception as e:
            print(f"[HOT-SWAP] Config save error: {e}")

        # [CLEAN SWITCHING] delegating to start_single_camera which now handles the release()
        # This occurs asynchronously in a background thread.
        self.start_single_camera(i, source_text)
        
        # [STATE MANAGEMENT] Ensure the system is "Powered On"
        self.are_cameras_active = True
        self.worker_signals.status_updated.emit(f"Swapping CAM_{i+1}...")
        
        # [RECORDER SYNC] Ensure the recorder is running if the system is active
        self.start_recorder()
        
        # [HANG FIX] Cancel Queue join threads to prevent Python from blocking indefinitely on exit
        try:
            self.result_queue.cancel_join_thread()
            self.recording_queue.cancel_join_thread()
            for q in self.task_queues:
                if q: q.cancel_join_thread()
        except: pass

    def _log_debug(self, msg):
        try:
            import os
            from config.config import BASE_DIR
            pending_dir = os.path.join(BASE_DIR, "logs")
            os.makedirs(pending_dir, exist_ok=True)
            log_path = os.path.join(pending_dir, "backend_debug.log")
            with open(log_path, "a") as f:
                f.write(f"[{datetime.now()}] {msg}\n")
        except Exception as err:
            print(f"Log Error: {err}", flush=True)
        print(f"[Backend] {msg}", flush=True)
