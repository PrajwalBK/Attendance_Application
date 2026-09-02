"""
go2rtc WebRTC Streaming Service for Vision Attendance System.

Manages the lifecycle of the go2rtc binary, configures RTSP-to-WebRTC streams,
and handles SDP offer/answer exchanges for ultra-low latency (<100ms) browser streaming.
"""

import os
import sys
import time
import json
import logging
import platform
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import zipfile
import io
from typing import Dict, Any, Optional

logger = logging.getLogger("webrtc_service")
logging.basicConfig(level=logging.INFO)

class WebRTCService:
    def __init__(self, api_port: int = 1984, webrtc_port: int = 8555, rtsp_port: int = 8554):
        self.api_port = api_port
        self.webrtc_port = webrtc_port
        self.rtsp_port = rtsp_port
        self.base_url = f"http://127.0.0.1:{self.api_port}"
        
        # Base directory resolution
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.bin_dir = os.path.join(self.base_dir, "bin")
        self.config_dir = os.path.join(self.base_dir, "config")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        
        os.makedirs(self.bin_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.binary_path = self._resolve_binary_path()
        self.config_path = os.path.join(self.config_dir, "go2rtc.yaml")
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None
        
        # In-memory stream tracking
        self.registered_streams: Dict[str, str] = {}

    def _resolve_binary_path(self) -> str:
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "windows":
            return os.path.join(self.bin_dir, "go2rtc.exe")
        elif system == "linux":
            if "arm" in machine or "aarch64" in machine:
                return os.path.join(self.bin_dir, "go2rtc_linux_arm64")
            return os.path.join(self.bin_dir, "go2rtc_linux_amd64")
        elif system == "darwin":
            if "arm" in machine:
                return os.path.join(self.bin_dir, "go2rtc_mac_arm64")
            return os.path.join(self.bin_dir, "go2rtc_mac_amd64")
        return os.path.join(self.bin_dir, "go2rtc")

    def ensure_binary(self) -> bool:
        """Download go2rtc executable if not present on the current host system."""
        if os.path.exists(self.binary_path):
            return True
            
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        url = None
        is_zip = False
        
        if system == "windows":
            url = "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_win64.zip"
            is_zip = True
        elif system == "linux":
            if "arm" in machine or "aarch64" in machine:
                url = "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_arm64"
            else:
                url = "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_amd64"
        elif system == "darwin":
            if "arm" in machine:
                url = "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_mac_arm64"
            else:
                url = "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_mac_amd64"
                
        if not url:
            logger.error(f"Unsupported OS/Architecture for go2rtc: {system} {machine}")
            return False
            
        logger.info(f"Downloading go2rtc from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if is_zip:
                    with zipfile.ZipFile(io.BytesIO(data)) as z:
                        z.extractall(self.bin_dir)
                else:
                    with open(self.binary_path, "wb") as f:
                        f.write(data)
                    os.chmod(self.binary_path, 0o755)
            logger.info(f"Successfully downloaded go2rtc to {self.binary_path}")
            return os.path.exists(self.binary_path)
        except Exception as e:
            logger.error(f"Failed to auto-download go2rtc: {e}")
            return False

    def _generate_default_config(self):
        """Generate a clean go2rtc YAML configuration if not present."""
        yaml_content = f"""# Auto-generated go2rtc configuration for Vision Attendance System
log:
  level: info

api:
  listen: "127.0.0.1:{self.api_port}"

rtsp:
  listen: "127.0.0.1:{self.rtsp_port}"

webrtc:
  listen: "0.0.0.0:{self.webrtc_port}"

streams: {{}}
"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

    def start(self) -> bool:
        """Start the go2rtc background process."""
        if self.is_running():
            logger.info("go2rtc is already running.")
            return True
            
        if not self.ensure_binary():
            logger.error("Cannot start go2rtc: Binary is missing.")
            return False
            
        if not os.path.exists(self.config_path):
            self._generate_default_config()

        log_path = os.path.join(self.logs_dir, "go2rtc.log")
        self.log_file = open(log_path, "a", encoding="utf-8")
        
        cmd = [self.binary_path, "-config", self.config_path]
        logger.info(f"Starting go2rtc process: {' '.join(cmd)}")
        
        try:
            creationflags = 0
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0x08000000

            self.process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags
            )
            
            # Wait up to 3 seconds for server to be responsive
            for _ in range(15):
                time.sleep(0.2)
                if self.is_running():
                    logger.info(f"go2rtc started successfully on port {self.api_port} (WebRTC: {self.webrtc_port})")
                    return True
                    
            logger.warning("go2rtc process started, but health check is pending.")
            return True
        except Exception as e:
            logger.error(f"Failed to start go2rtc process: {e}")
            return False

    def stop(self):
        """Stop the go2rtc process."""
        if self.process:
            logger.info("Stopping go2rtc process...")
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
            
        if self.log_file and not self.log_file.closed:
            self.log_file.close()

    def is_running(self) -> bool:
        """Check if go2rtc API endpoint is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/streams", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def add_stream(self, stream_name: str, src: str) -> bool:
        """
        Dynamically add or update a stream source in go2rtc.
        
        Args:
            stream_name: Unique stream identifier, e.g., 'cam0', 'cam1'.
            src: RTSP URL (e.g., 'rtsp://user:pass@ip:554/ch01/0') or source string.
        """
        if not src:
            return False
            
        src_clean = str(src).strip()
        
        # go2rtc PUT /api/streams API accepts JSON: {"name": stream_name, "src": src_clean}
        url = f"{self.base_url}/api/streams"
        payload = json.dumps({"name": stream_name, "src": src_clean}).encode("utf-8")
        
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="PUT"
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status in (200, 201):
                    self.registered_streams[stream_name] = src_clean
                    logger.info(f"[WebRTC] Registered stream '{stream_name}' -> {src_clean}")
                    return True
        except Exception as e:
            logger.error(f"[WebRTC] Failed to register stream '{stream_name}': {e}")
        return False

    def remove_stream(self, stream_name: str) -> bool:
        """Remove a stream from go2rtc."""
        encoded_src = urllib.parse.quote(stream_name)
        url = f"{self.base_url}/api/streams?src={encoded_src}"
        try:
            req = urllib.request.Request(url, method="DELETE")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                self.registered_streams.pop(stream_name, None)
                logger.info(f"[WebRTC] Removed stream '{stream_name}'")
                return True
        except Exception as e:
            logger.warning(f"[WebRTC] Error removing stream '{stream_name}': {e}")
        return False

    def exchange_webrtc_sdp(self, stream_name: str, offer_sdp: str) -> Optional[str]:
        """
        Exchange WebRTC SDP offer with go2rtc and return the SDP answer.
        
        Args:
            stream_name: Registered stream name ('cam0', 'cam1').
            offer_sdp: WebRTC Session Description Protocol offer string from browser.
        Returns:
            WebRTC SDP answer string from go2rtc.
        """
        encoded_name = urllib.parse.quote(stream_name)
        url = f"{self.base_url}/api/webrtc?src={encoded_name}"
        
        try:
            data = offer_sdp.encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/sdp", "Accept": "application/sdp"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    answer_sdp = resp.read().decode("utf-8")
                    return answer_sdp
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")
            logger.error(f"[WebRTC] SDP exchange HTTP Error {he.code}: {err_body}")
        except Exception as e:
            logger.error(f"[WebRTC] SDP exchange failed for stream '{stream_name}': {e}")
        return None

    def exchange_whep_sdp(self, stream_name: str, offer_sdp: str) -> Optional[str]:
        """
        Exchange WHEP (WebRTC HTTP Egress Protocol) SDP offer with go2rtc.
        """
        encoded_name = urllib.parse.quote(stream_name)
        url = f"{self.base_url}/api/whep?src={encoded_name}"
        
        try:
            data = offer_sdp.encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/sdp", "Accept": "application/sdp"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status in (200, 201):
                    return resp.read().decode("utf-8")
        except Exception as e:
            logger.error(f"[WebRTC] WHEP exchange failed for '{stream_name}': {e}")
        return None

    def get_streams_status(self) -> Dict[str, Any]:
        """Retrieve active streams and connection statistics from go2rtc."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/streams", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e), "registered": self.registered_streams}


# Global singleton instance
webrtc_service = WebRTCService()
