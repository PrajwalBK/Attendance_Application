import os

import sys
def _get_storage_dir():
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        test_file = os.path.join(exe_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return exe_dir
        except Exception:
            local_app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
            fallback_dir = os.path.join(local_app_data, 'VisionAttendance')
            os.makedirs(fallback_dir, exist_ok=True)
            # Bootstrap: copy bundled configurations if they don't exist in LocalAppData yet
            import shutil
            for cfg in ['auth_config.json', 'cam_config.json', 'db_config.json']:
                src = os.path.join(exe_dir, 'config', cfg)
                dst = os.path.join(fallback_dir, 'config', cfg)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                    except: pass
            return fallback_dir
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STORAGE_DIR = _get_storage_dir()

if getattr(sys, 'frozen', False):
    ASSET_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    ASSET_DIR = STORAGE_DIR

BASE_DIR = STORAGE_DIR # Maintain backward compatibility

# --- CONFIGURATION ---

# Database configuration is now handled dynamically via core/db_config_manager.py
# and stored in config/db_config.json

# --- SMTP EMAIL CONFIGURATION ---

SMTP_CONFIG = {
    'smtp_server': 'smtp.gmail.com',      # SMTP server (Gmail example)
    'smtp_port': 587,                      # Port (587 for TLS, 465 for SSL)
    'sender_email': 'your_email@gmail.com',  # Your email address
    'sender_password': 'your_app_password',  # App-specific password
    'use_tls': True,                       # Use TLS encryption
    'admin_email': 'admin@example.com',    # Admin email for notifications
}

# Email Notification Settings
EMAIL_NOTIFICATIONS_ENABLED = False        # Enable/disable email notifications
SEND_ARRIVAL_NOTIFICATIONS = False        # Send email on every arrival (not recommended)
SEND_DEPARTURE_NOTIFICATIONS = False      # Send email on every departure (not recommended)
SEND_ABSENCE_ALERTS = True               # Send alerts for absent employees
SEND_LATE_ARRIVAL_ALERTS = True          # Send alerts for late arrivals
SEND_DAILY_SUMMARY = False               # Send daily summary emails (all attendance data)
DAILY_SUMMARY_TIME = '18:00'             # Time to send daily summary (24-hour format)


# --- FILE PATHS ---
# All paths converted to absolute paths for IIS/Production safety

FACE_ENCODINGS_PATH = os.path.join(BASE_DIR, 'data', 'face_encodings.pkl')
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, 'data', 'unknown_faces')
TEMP_RECORDINGS_DIR = os.path.join(BASE_DIR, 'data', 'temp_recordings')
VERIFIED_LOG_PATH = os.path.join(BASE_DIR, 'data', 'verified_attendance_log.txt')
SNAPSHOTS_DIR = os.path.join(BASE_DIR, 'data', 'attendance_snapshots')
# [PRODUCTION] Move internal read-only assets to ASSET_DIR
DNN_PROTO_PATH = os.path.join(ASSET_DIR, 'data', 'models', 'deploy.prototxt')
DNN_MODEL_PATH = os.path.join(ASSET_DIR, 'data', 'models', 'res10_300x300_ssd_iter_140000.caffemodel')

# YOLOv8-Face Model Paths & Config
_YOLO_FP32_PATH = os.path.join(ASSET_DIR, 'data', 'models', 'yolov8n-face.onnx')
_YOLO_INT8_PATH = os.path.join(ASSET_DIR, 'data', 'models', 'yolov8n-face_int8.onnx')
YOLO_MODEL_PATH = _YOLO_INT8_PATH if os.path.exists(_YOLO_INT8_PATH) else _YOLO_FP32_PATH
YOLO_CONFIDENCE_THRESHOLD = 0.25

PROCESSED_RECORDINGS_DIR = os.path.join(BASE_DIR, 'data', 'processed_recordings')
VIDEO_CHUNK_DURATION = 30  # Seconds per video file
RECORD_VIDEO = False  # Set to True to save video files, False to disable saving video files


# Face Recognition Settings
SIMILARITY_THRESHOLD = 0.4  
DETECTION_SIZE = (1024, 1024) 
FACE_DETECTION_MODEL = 'buffalo_l' 

# --- MASK DETECTION SETTINGS ---
MASK_DETECTION_ENABLED = True             # Master switch for mask-aware recognition
MASKED_SIMILARITY_THRESHOLD = 0.25        # Lower threshold for masked face matching (was 0.35; lowered to accept near-miss scores ~0.29)
UPPER_FACE_CROP_RATIO = 0.55             # Crop top 55% of face bounding box for upper-face embedding
MASK_NOSE_RATIO_THRESHOLD = 0.30          # Landmark heuristic: nose-to-eye ratio
MASK_MOUTH_RATIO_THRESHOLD = 0.22         # Landmark heuristic: mouth drop ratio
MASK_MOUTH_SPREAD_THRESHOLD = 0.35        # Landmark heuristic: mouth width ratio

# --- PERSON RE-ID SETTINGS (OSNet body-appearance fallback) ---
REID_ENABLED              = True          # Master switch — False disables all ReID
REID_MODEL_PATH           = os.path.join(ASSET_DIR, 'data', 'models', 'osnet_ibn_x1_0.onnx')
REID_SIMILARITY_THRESHOLD = 0.72          # Cosine similarity required for a body match
REID_CACHE_DIR            = os.path.join(BASE_DIR,  'data', 'reid_cache')  # Daily cache dir
REID_AUTOSAVE_INTERVAL    = 60            # Seconds between auto-saves of the ReID gallery
REID_TEMPORAL_WINDOW      = 43200         # Keep body templates active for 43200 seconds (12 hours)
                                            # (was 300s — too long given hospital staff often
                                            # wear similar-coloured scrubs; a tighter window
                                            # reduces the chance of matching the wrong person
                                            # who happens to be dressed similarly)

# Execution Providers (GPU/CPU)
EXECUTION_PROVIDERS = [
    'CPUExecutionProvider'
]

# ONNX Runtime Thread Limits
ORT_INTRA_OP_NUM_THREADS = 2
ORT_INTER_OP_NUM_THREADS = 2

# Detection Backend Configuration
# Options: 'insightface' (Default, Accurate) | 'opencv_dnn' (Faster, Less Accurate) | 'yolov8' (High Accuracy, Fast)
FACE_DETECTION_BACKEND = 'yolov8'

# Triage Detector Backend Configuration
# Options: 'opencv_dnn' (Default, super fast CPU SSD) | 'yolov8' (Higher accuracy, recommended for crowded areas)
TRIAGE_DETECTION_BACKEND = 'yolov8' 


# --- SECURITY & AUTH ---
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7" # Generated secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 24 Hours

# --- API CONFIGURATION ---
def is_api_enabled():
    """Returns True if system is in Cloud or Local API mode. Returns False if Direct Database mode is active."""
    try:
        from config.auth_manager import AuthManager
        return not AuthManager.load_setting("db_mode", False)
    except Exception:
        return True

USE_API = is_api_enabled()


def sanitize_url(url):
    """Auto-fix common URL typos so non-technical users don't get cryptic errors."""
    if not url:
        return url
    url = url.strip().replace(' ', '')
    url = url.replace('http//', 'http://').replace('https//', 'https://')
    
    # Handle missing protocol
    if not url.startswith('http://') and not url.startswith('https://'):
        # [ROBUST] If it looks like a local IP or localhost, default to http
        is_local = any(x in url for x in ["192.168.", "localhost", "127.0.0.1", "10."])
        if is_local:
            url = 'http://' + url
        else:
            url = 'https://' + url
            
    # [ROBUST] Force HTTPS for production cloud domain if not specified or incorrectly HTTP
    if "visionattendance.com" in url and url.startswith('http://'):
        url = url.replace('http://', 'https://')
    elif "visionattendance.com" in url and not url.startswith('https://'):
        url = 'https://' + url.replace('http://', '').lstrip('/')
            
    return url.rstrip('/')

# Default Local Settings
API_BASE_URL = 'http://127.0.0.1:8080'
API_USERNAME = None
API_PASSWORD = None

# [ROBUST] Load Persistent Cloud Settings via AuthManager
from config.auth_manager import AuthManager

def refresh_api_config():
    """Reloads API credentials and URL from disk at runtime."""
    global API_BASE_URL, API_USERNAME, API_PASSWORD
    _email, _pass, _url = AuthManager.load_credentials()
    API_BASE_URL = sanitize_url(_url)
    API_USERNAME = _email
    API_PASSWORD = _pass
    return API_BASE_URL, API_USERNAME, API_PASSWORD

# Initial Load
refresh_api_config()

if API_USERNAME:
    print(f"[CONFIG] Loaded Persistent Cloud Link for: {API_USERNAME}")
else:
    print("[CONFIG] No Cloud Credentials saved. Please link your account in the UI.")

# OpenCV DNN Model Paths removed (moved to production block above)
DNN_CONFIDENCE_THRESHOLD = 0.5

# Visual Settings
SHOW_DETECTION_SCORE = True   # Enable "Score" overlay (0.95, etc.)

# ByteTrack Configuration
TRACK_ACTIVATION_THRESHOLD = 0.5  
LOST_TRACK_BUFFER = 30            
FRAME_RATE = 30                   

# Attendance Settings
ATTENDANCE_COOLDOWN_SECONDS = 30   # 30 Seconds Cooldown (Frequent Logging)
LOITERING_THRESHOLD = 90           # 90 Seconds Cooldown to prevent instant re-entry/lingering near door
AUTO_MARK_ARRIVAL = True          
AUTO_MARK_LEAVING = True          
# Auto-Away: If a person hasn't been detected for this many minutes, mark as Away
# This replaces the need for a dedicated exit camera.
AUTO_AWAY_MINUTES = 5             

# Video Processing Settings
# Video Processing Settings
WEBCAM_INDEX = 0               
DISPLAY_LANDMARKS = True          
DISPLAY_FPS = True                
DISPLAY_INFO_PANEL = True

# Performance Optimization
PROCESS_EVERY_N_FRAMES = 4    # Process every 4th frame for high CPU efficiency
RESIZE_FACTOR = 1.0           # No resizing for maximum detail         

# Annotation Settings
BOX_THICKNESS = 2
TEXT_THICKNESS = 1
TEXT_SCALE = 0.5
TEXT_PADDING = 5

# Colors (BGR format)
COLOR_DETECTED = (0, 255, 0)      
COLOR_UNKNOWN = (0, 0, 255)       
COLOR_WARNING = (0, 165, 255)     
COLOR_INFO = (255, 255, 255)      

# CSV Export Settings
CSV_EXPORT_FOLDER = os.path.join(BASE_DIR, 'exports' + os.sep) # End with separator
CSV_DATE_FORMAT = '%Y-%m-%d'
CSV_TIME_FORMAT = '%H:%M:%S'

# Registration Settings
REGISTRATION_CAPTURE_KEY = 'c'   
REGISTRATION_CANCEL_KEY = 'q'    
MIN_FACE_SIZE = 50                
MAX_REGISTRATION_ATTEMPTS = 3     

# Logging Settings
LOG_ATTENDANCE_MARKS = True       
LOG_RECOGNITION_EVENTS = True     
LOG_ERRORS = True                 

def get_config():
    """Return configuration as dictionary"""
    # Ensure we return the latest values if they changed during runtime
    refresh_api_config()
    
    return {
        'mysql_config': None,
        'smtp_config': SMTP_CONFIG,
        'face_encodings_path': FACE_ENCODINGS_PATH,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'detection_size': DETECTION_SIZE,
        'face_detection_model': FACE_DETECTION_MODEL,
        'execution_providers': EXECUTION_PROVIDERS,
        'track_activation_threshold': TRACK_ACTIVATION_THRESHOLD,
        'lost_track_buffer': LOST_TRACK_BUFFER,
        'frame_rate': FRAME_RATE,
        'attendance_cooldown': ATTENDANCE_COOLDOWN_SECONDS,
        'start_cooldown': ATTENDANCE_COOLDOWN_SECONDS, # Alias for clarity
        'auto_away_minutes': AUTO_AWAY_MINUTES,
        'webcam_index': WEBCAM_INDEX,
        'display_landmarks': DISPLAY_LANDMARKS,
        'display_fps': DISPLAY_FPS,
        'process_every_n_frames': PROCESS_EVERY_N_FRAMES,
        'ort_intra_op_num_threads': ORT_INTRA_OP_NUM_THREADS,
        'ort_inter_op_num_threads': ORT_INTER_OP_NUM_THREADS,
        'resize_factor': RESIZE_FACTOR,
        'face_detection_backend': FACE_DETECTION_BACKEND,
        'triage_detection_backend': TRIAGE_DETECTION_BACKEND,
        'dnn_proto_path': DNN_PROTO_PATH,
        'dnn_model_path': DNN_MODEL_PATH,
        'dnn_confidence_threshold': DNN_CONFIDENCE_THRESHOLD,
        'yolo_model_path': YOLO_MODEL_PATH,
        'yolo_confidence_threshold': YOLO_CONFIDENCE_THRESHOLD,
        'show_detection_score': SHOW_DETECTION_SCORE,
        'temp_recordings_dir': TEMP_RECORDINGS_DIR,
        'verified_log_path': VERIFIED_LOG_PATH,
        'snapshots_dir': SNAPSHOTS_DIR,
        'video_chunk_duration': VIDEO_CHUNK_DURATION,
        'processed_recordings_dir': PROCESSED_RECORDINGS_DIR,
        'record_video': RECORD_VIDEO,
        'loitering_threshold': LOITERING_THRESHOLD,
        'use_api': is_api_enabled(),
        'api_base_url': API_BASE_URL,
        'secret_key': SECRET_KEY,
        'algorithm': ALGORITHM,
        'access_token_expire_minutes': ACCESS_TOKEN_EXPIRE_MINUTES,
        'api_username': API_USERNAME,
        'api_password': API_PASSWORD,
        # Mask Detection
        'mask_detection_enabled': MASK_DETECTION_ENABLED,
        'masked_similarity_threshold': MASKED_SIMILARITY_THRESHOLD,
        'upper_face_crop_ratio': UPPER_FACE_CROP_RATIO,
        'mask_nose_ratio_threshold': MASK_NOSE_RATIO_THRESHOLD,
        'mask_mouth_ratio_threshold': MASK_MOUTH_RATIO_THRESHOLD,
        'mask_mouth_spread_threshold': MASK_MOUTH_SPREAD_THRESHOLD,
        # Person ReID
        'reid_enabled': REID_ENABLED,
        'reid_model_path': REID_MODEL_PATH,
        'reid_similarity_threshold': REID_SIMILARITY_THRESHOLD,
        'reid_cache_dir': REID_CACHE_DIR,
        'reid_autosave_interval': REID_AUTOSAVE_INTERVAL,
    }

def validate_config():
    """Validate configuration settings"""
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
    print("Using default values...")