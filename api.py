import os
import sys
import cv2
import time
import asyncio
import threading
import multiprocessing
import queue
from typing import Dict, Any

from fastapi import FastAPI, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add core backend modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config
from core.camera import ThreadedCamera
from config.cam_config_manager import CamConfigManager

# Pre-load heavy modules in the main thread to avoid ImportLock deadlocks in background threads
import core.face_recognition

app = FastAPI(title="Vision Attendance Local API")

# Allow CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State for Hardware
class SystemState:
    def __init__(self):
        self.is_system_active = False
        self.are_cameras_active = False
        self.caps = [None, None] # Supporting up to 2 cameras based on original UI
        self.latest_frames = [None, None]
        self.frame_lock = threading.Lock()
        
        # Multiprocessing properties
        self.result_queue = None
        self.recording_queue = None
        self.task_queue_1 = None
        self.task_queue_2 = None
        
        self.worker_1 = None
        self.worker_2 = None
        self.recorder_process = None
        
        self.thread_queue = queue.Queue(maxsize=30)
        
        # UI Logs
        self.recent_logs = []
        self.logs_lock = threading.Lock()

state = SystemState()

def _bg_queue_worker():
    """Thread to move frames from UI/API -> Recorder Process"""
    while True:
        try:
            item = state.thread_queue.get()
            if state.recording_queue:
                state.recording_queue.put(item)
        except Exception as e:
            print(f"BG Thread Error: {e}")

threading.Thread(target=_bg_queue_worker, daemon=True).start()

def result_consumer_loop():
    """Consume results from AI workers to prevent memory leaks, log to UI and console"""
    while True:
        if state.result_queue and not state.result_queue.empty():
            try:
                data = state.result_queue.get_nowait()
                print(f"[API] Detection Event: {data['name']} (Sim: {data['sim']:.2f})")
                
                with state.logs_lock:
                    state.recent_logs.insert(0, {
                        'id': data.get('id', 'N/A'),
                        'name': data.get('name', 'Unknown'),
                        'timestamp': data.get('timestamp', time.strftime("%H:%M:%S")),
                        'worker': data.get('worker', 1),
                        'sim': data.get('sim', 0.0)
                    })
                    # Keep only last 50 logs
                    if len(state.recent_logs) > 50:
                        state.recent_logs.pop()
                        
            except queue.Empty:
                pass
            except Exception as e:
                pass
        time.sleep(0.5)

threading.Thread(target=result_consumer_loop, daemon=True).start()

# --- Background Worker Mock/Init (To be fleshed out with actual OpenCV integration) ---
def video_capture_loop():
    """Continuously fetch frames and store them for the MJPEG stream."""
    while True:
        if not state.are_cameras_active:
            time.sleep(1)
            continue
            
        for i in range(2):
            cap = state.caps[i]
            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 1. Add to Recorder Queue (PRODUCER) for AI Processing
                    if state.is_system_active:
                        try:
                            # Put frame into THREAD queue (Fast, In-Process)
                            state.thread_queue.put_nowait((i, frame, time.time()))
                        except queue.Full:
                            pass # Drop frame silently if thread is backed up
                        except Exception:
                            pass
                            
                    # 2. Resize for stream performance (similar to legacy UI)
                    frame_resized = cv2.resize(frame, (640, 480))
                    with state.frame_lock:
                        state.latest_frames[i] = frame_resized.copy()
        time.sleep(0.03) # ~30 FPS

# Start the capture thread
threading.Thread(target=video_capture_loop, daemon=True).start()

# --- Pydantic Models ---
class SystemToggleReq(BaseModel):
    action: str # "start" or "stop"

class CameraToggleReq(BaseModel):
    action: str # "start" or "stop"
    cam1_source: str = None
    cam2_source: str = None

# --- API Endpoints ---

@app.get("/api/local/status")
async def get_status():
    return {
        "is_system_active": state.is_system_active,
        "are_cameras_active": state.are_cameras_active
    }

@app.get("/api/local/logs")
async def get_logs():
    with state.logs_lock:
        return {"logs": list(state.recent_logs)}

@app.get("/api/local/auth-token")
async def get_auth_token():
    """Auto-login to remote API using stored credentials and return a token for the frontend."""
    import requests as req_lib
    conf = get_config()
    base_url = conf.get('api_base_url', 'https://api.visionattendance.com')
    username = conf.get('api_username', '')
    password = conf.get('api_password', '')
    try:
        r = req_lib.post(f"{base_url}/api/auth/login/", json={"email": username, "password": password}, timeout=10)
        if r.status_code == 200:
            token = r.json().get('access_token', '')
            return {"token": token, "status": "ok"}
        return {"token": "", "status": "login_failed", "detail": r.text}
    except Exception as e:
        return {"token": "", "status": "error", "detail": str(e)}

from core.multiprocess_handler import AttendanceWorker
from core.recorder import RecorderWorker

def start_background_workers(state_ref):
    """Launch background workers using threads via composition to cleanly run classes inside the FastAPI event loop."""
    import queue
    import threading
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Generating queues...\n")
    state_ref.result_queue = queue.Queue()
    state_ref.recording_queue = queue.Queue(maxsize=60)
    state_ref.task_queue_1 = queue.Queue()
    state_ref.task_queue_2 = queue.Queue()
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Initializing worker 1...\n")
    state_ref.worker_1 = AttendanceWorker(state_ref.task_queue_1, state_ref.result_queue, worker_id=1, assigned_cam_index=0)
    state_ref.worker_1.stop_event = threading.Event()
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Initializing worker 2...\n")
    state_ref.worker_2 = AttendanceWorker(state_ref.task_queue_2, state_ref.result_queue, worker_id=2, assigned_cam_index=1)
    state_ref.worker_2.stop_event = threading.Event()
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Starting threads...\n")
    
    t1 = threading.Thread(target=state_ref.worker_1.run, daemon=True)
    t2 = threading.Thread(target=state_ref.worker_2.run, daemon=True)
    t1.start()
    t2.start()
    
    from config.config import get_config
    conf = get_config()
    rec_dir = conf.get('temp_recordings_dir', 'data/temp_recordings')
    queue_map = {0: state_ref.task_queue_1, 1: state_ref.task_queue_2}
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Initializing recorder...\n")
    state_ref.recorder_process = RecorderWorker(state_ref.recording_queue, queue_map, rec_dir)
    state_ref.recorder_process.stop_event = threading.Event()
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Starting recorder thread...\n")
    t3 = threading.Thread(target=state_ref.recorder_process.run, daemon=True)
    t3.start()
    
    with open("debug_startup.log", "a") as f: f.write("[DEBUG] Finished start_background_workers!\n")

@app.post("/api/local/system/toggle")
def toggle_system(req: SystemToggleReq):
    with open("debug_startup.log", "a") as f: f.write(f"System toggle requested: {req.action}\n")
    if req.action == "start":
        if state.is_system_active: return {"status": "success", "message": "Already active"}
        
        # Launch workers in a background thread so we don't block the HTTP response
        # ML models take 5-15 seconds to initialize — we return immediately
        state.is_system_active = True
        def _boot():
            with open("debug_startup.log", "a") as f: f.write("[DEBUG] Calling start_background_workers...\n")
            start_background_workers(state)
            with open("debug_startup.log", "a") as f: f.write("[DEBUG] Boot thread finished!\n")
        boot_thread = threading.Thread(target=_boot, daemon=True)
        boot_thread.start()
        
        return {"status": "success", "message": "System starting... (workers initializing in background)"}
        
    elif req.action == "stop":
        state.is_system_active = False
        print("[API] Stopping AI Workers...")
        
        # Stop Workers
        if state.worker_1: state.worker_1.stop()
        if state.worker_2: state.worker_2.stop()
        if state.recorder_process: state.recorder_process.stop()
        
        # We don't necessarily need to stop cameras, but the original code did
        if state.are_cameras_active:
             for i in range(2):
                 if state.caps[i]:
                     state.caps[i].release()
                     state.caps[i] = None
             state.are_cameras_active = False
             
        return {"status": "success", "message": "System stopped."}
    return {"status": "error", "message": "Invalid action."}

def _parse_source(selection_str):
    if not selection_str: return None
    try:
        if "Channel" in selection_str:
            channel_num = selection_str.split(" ")[1]
            cam_conf = CamConfigManager.load_config()
            base_url = cam_conf.get('rtsp_template', '')
            if not base_url: return None
            import re
            return re.sub(r'channel=\d+', f'channel={channel_num}', str(base_url))
        elif "Webcam" in selection_str:
            return int(selection_str.split(" ")[1])
    except:
        pass
    return None

@app.post("/api/local/cameras/toggle")
async def toggle_cameras(req: CameraToggleReq):
    print(f"Camera toggle requested: {req.action}")
    if req.action == "start":
        if not state.is_system_active:
            return {"status": "error", "message": "Start system first."}
            
        src1 = _parse_source(req.cam1_source)
        if src1 is not None:
            print(f"Opening Cam 1: {src1}")
            state.caps[0] = ThreadedCamera(src1)
            
        src2 = _parse_source(req.cam2_source)
        if src2 is not None:
             print(f"Opening Cam 2: {src2}")
             state.caps[1] = ThreadedCamera(src2)
             
        state.are_cameras_active = any(c is not None for c in state.caps)
        return {"status": "success", "message": "Cameras starting."}
        
    elif req.action == "stop":
        state.are_cameras_active = False
        for i in range(2):
             if state.caps[i]:
                 state.caps[i].release()
                 state.caps[i] = None
        return {"status": "success", "message": "Cameras stopped."}

def generate_frames(cam_index: int):
    """Generator for MJPEG stream formatting."""
    while True:
        if not state.are_cameras_active:
            time.sleep(1)
            continue
            
        frame = None
        with state.frame_lock:
             if state.latest_frames[cam_index] is not None:
                 frame = state.latest_frames[cam_index].copy()
                 
        if frame is None:
            # Yield a blank placeholder image or simply wait
            time.sleep(0.1)
            continue
            
        # Optional: Add simple timestamp or info on the frame
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.05) # Cap stream at ~20-30fps

@app.get("/api/local/video_feed/{camera_id}")
async def video_feed(camera_id: int):
    if camera_id not in [0, 1]:
        return {"error": "Invalid camera ID"}
    return StreamingResponse(generate_frames(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    # When running directly, start uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
