import os
import sys

# System Configuration file for Vision Attendance System (CPU Version)

def get_base_dir():
    """Get the absolute path to the base directory of the project."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = get_base_dir()
ASSET_DIR = BASE_DIR
WEBCAM_INDEX = 0

# System Info
APP_NAME = "Vision Attendance System (CPU Optimized)"
VERSION = "6.2.0"
AUTHOR = "Prajwal BK"

# Camera Defaults
DEFAULT_FPS = 15  # Optimized for CPU execution to reduce thread load
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MAX_CAMS = 8

# Hardware Acceleration Profile
ACCELERATION_PROFILE = "CPU_MULTI_THREAD"
DYNAMIC_WORKER_SCALING = True

# Storage Paths
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
RECORDS_DIR = os.path.join(BASE_DIR, "data", "records")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "data", "attendance_snapshots")
DATABASE_DIR = os.path.join(BASE_DIR, "database")

# Ensure required directories exist
for path in [MODELS_DIR, RECORDS_DIR, SNAPSHOTS_DIR, DATABASE_DIR]:
    os.makedirs(path, exist_ok=True)

# YOLOv8 Model Settings
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-face_int8.onnx")
YOLO_BODY_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.onnx")
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_BODY_CONFIDENCE_THRESHOLD = 0.30

# OpenCV DNN Model Settings (Caffe SSD Fallback)
DNN_PROTO_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
DNN_MODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONFIDENCE_THRESHOLD = 0.30

# Video Processor Defaults
PROCESS_EVERY_N_FRAMES = 1
ATTENDANCE_COOLDOWN_SECONDS = 300  # 5 Minutes per person

# RTSP Stream Reconnection Settings
RTSP_RECONNECT_INTERVAL = 3.0
RTSP_BUFFER_SIZE = 1

# Database Configuration (SQLite Fallback)
DB_PATH = os.path.join(DATABASE_DIR, "attendance.db")
OFFLINE_STORAGE_PATH = os.path.join(DATABASE_DIR, "offline_records.json")
FACE_ENCODINGS_PATH = os.path.join(DATABASE_DIR, "face_encodings.pkl")

# Performance Logging
LOG_PERFORMANCE_METRICS = True
METRICS_LOG_PATH = os.path.join(BASE_DIR, "logs", "performance.log")

# Ensure logs dir exists
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# Application Default State
DEFAULT_ACTIVE_CAMS = 2
AUTORUN_DETECTION_ON_START = True

# Debug Mode
DEBUG_MODE = True

# Recording Settings
RECORD_VIDEO = False

# Face Recognition Settings
SIMILARITY_THRESHOLD = 0.40
DETECTION_SIZE = (640, 640) 
FACE_DETECTION_MODEL = 'buffalo_l' 
USE_HIGH_RES_SNAPSHOTS = True

# --- MULTI-PERSON TRACKER & FSM CONFIGURATION ---
TRACK_TIMEOUT_SECONDS = 2.0
QUEUE_TIMEOUT_SECONDS = 5.0
PROCESSING_TIMEOUT_SECONDS = 10.0
MAX_RETRY = 3
MATCHING_IOU_WEIGHT = 0.7
MATCHING_CENTROID_WEIGHT = 0.3

# --- PER-CAMERA ISOLATED SNAPSHOT PIPELINE CONFIGURATION ---
FRAME_BUFFER_SECONDS = 2.0
SNAPSHOT_QUEUE_TIMEOUT = 1.0
SNAPSHOT_PADDING = 0.20

# --- MASK DETECTION SETTINGS ---
MASK_DETECTION_ENABLED = True
MASKED_SIMILARITY_THRESHOLD = 0.35
UPPER_FACE_CROP_RATIO = 0.55
MASK_NOSE_RATIO_THRESHOLD = 0.30
MASK_MOUTH_RATIO_THRESHOLD = 0.22
MASK_MOUTH_SPREAD_THRESHOLD = 0.35

# Execution Providers
import ctypes.util
import onnxruntime as ort

EXECUTION_PROVIDERS = ['CPUExecutionProvider']

print(f"[GPU CHECK] Forced strictly to CPU: {EXECUTION_PROVIDERS}", flush=True)

# ONNX Runtime Thread Limits
if getattr(sys, 'frozen', False):
    ORT_INTRA_OP_NUM_THREADS = 1
    ORT_INTER_OP_NUM_THREADS = 1
else:
    ORT_INTRA_OP_NUM_THREADS = 2
    ORT_INTER_OP_NUM_THREADS = 2

# Background AI Worker Pool Size
NUM_AI_WORKERS = 3

# Detection Backend Configuration (Used for employee recognition from captured snapshots)
FACE_DETECTION_BACKEND = 'insightface'

# Triage Detector Backend Configuration (Used for capturing pending snapshots on live camera streams)
TRIAGE_DETECTION_BACKEND = 'opencv_dnn'

# --- SECURITY & AUTH ---
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# --- API CONFIGURATION ---
def is_api_enabled():
    try:
        from config.auth_manager import AuthManager
        return not AuthManager.load_setting("db_mode", False)
    except Exception:
        return True

USE_API = is_api_enabled()

def sanitize_url(url):
    if not url:
        return url
    url = url.strip().replace(' ', '')
    url = url.replace('http//', 'http://').replace('https//', 'https://')
    if not url.startswith('http://') and not url.startswith('https://'):
        is_local = any(x in url for x in ["192.168.", "localhost", "127.0.0.1", "10."])
        if is_local:
            url = 'http://' + url
        else:
            url = 'https://' + url
    if "visionattendance.com" in url and url.startswith('http://'):
        url = url.replace('http://', 'https://')
    elif "visionattendance.com" in url and not url.startswith('https://'):
        url = 'https://' + url.replace('http://', '').lstrip('/')
    return url.rstrip('/')

# API Base Credentials
API_BASE_URL = 'https://visionattendance.com'
API_USERNAME = None
API_PASSWORD = None

def refresh_api_config():
    """Reloads API credentials and URL from disk at runtime."""
    global API_BASE_URL, API_USERNAME, API_PASSWORD
    try:
        from config.auth_manager import AuthManager
        _email, _pass, _url = AuthManager.load_credentials()
        if _url:
            API_BASE_URL = sanitize_url(_url)
        else:
            API_BASE_URL = "https://visionattendance.com"
        if _email: API_USERNAME = _email
        if _pass: API_PASSWORD = _pass
    except Exception:
        if not API_BASE_URL:
            API_BASE_URL = "https://visionattendance.com"
    return API_BASE_URL, API_USERNAME, API_PASSWORD

refresh_api_config()

def get_config():
    return {
        'face_detection_backend': FACE_DETECTION_BACKEND,
        'triage_detection_backend': TRIAGE_DETECTION_BACKEND,
        'yolo_confidence_threshold': YOLO_CONFIDENCE_THRESHOLD,
        'yolo_body_confidence_threshold': YOLO_BODY_CONFIDENCE_THRESHOLD,
        'dnn_confidence_threshold': DNN_CONFIDENCE_THRESHOLD,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'attendance_cooldown_seconds': ATTENDANCE_COOLDOWN_SECONDS,
        'process_every_n_frames': PROCESS_EVERY_N_FRAMES,
        'num_ai_workers': NUM_AI_WORKERS,
        'detection_size': DETECTION_SIZE,
        'execution_providers': EXECUTION_PROVIDERS,
        'use_api': USE_API,
        'api_base_url': API_BASE_URL,
        'use_high_res_snapshots': USE_HIGH_RES_SNAPSHOTS,
        'mask_detection_enabled': MASK_DETECTION_ENABLED,
        'masked_similarity_threshold': MASKED_SIMILARITY_THRESHOLD,
        'upper_face_crop_ratio': UPPER_FACE_CROP_RATIO,
    }

def validate_config():
    errors = []
    if not 0.0 <= SIMILARITY_THRESHOLD <= 1.0:
        errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    if ATTENDANCE_COOLDOWN_SECONDS < 0:
        errors.append("ATTENDANCE_COOLDOWN_SECONDS must be non-negative")
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    return True

try:
    validate_config()
except ValueError as e:
    print(f"Warning: {e}")