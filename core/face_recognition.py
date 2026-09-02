import cv2
import numpy as np
import pickle
import os
import sys
import types
import time
import base64

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller bundle """
    from config.config import ASSET_DIR
    return os.path.join(ASSET_DIR, relative_path)

def mask_log(msg):
    try:
        from config.config import BASE_DIR
        # Direct write to worker log for filesystem visibility
        for wid in [1, 2, 3, 4]:
            log_path = os.path.join(BASE_DIR, f"worker_{wid}_debug.log")
            if os.path.exists(log_path):
                with open(log_path, "a") as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except: pass
    print(msg, flush=True)

# --- PI FIX: Safer Mock for 3D plotting ---
# Prevents 'Axes3D' crash and ignores matplotlib conflicts
# We use a real ModuleType instead of MagicMock to avoid TypeError in string comparisons
try:
    import mpl_toolkits.mplot3d
except (ImportError, RuntimeError, Exception):
    dummy_m = types.ModuleType("mpl_toolkits.mplot3d")
    class DummyAxes3D:
        name = '3d'
    dummy_m.Axes3D = DummyAxes3D
    sys.modules["mpl_toolkits.mplot3d"] = dummy_m
    
    # Ensure parent exists
    if "mpl_toolkits" not in sys.modules:
        dummy_p = types.ModuleType("mpl_toolkits")
        sys.modules["mpl_toolkits"] = dummy_p
    
    sys.modules["mpl_toolkits"].mplot3d = dummy_m

from insightface.app import FaceAnalysis


from config.config import (
    SIMILARITY_THRESHOLD, FACE_DETECTION_MODEL, DETECTION_SIZE,
    FACE_DETECTION_BACKEND, DNN_PROTO_PATH, DNN_MODEL_PATH, DNN_CONFIDENCE_THRESHOLD,
    MASK_DETECTION_ENABLED, MASKED_SIMILARITY_THRESHOLD, UPPER_FACE_CROP_RATIO,
    MASK_NOSE_RATIO_THRESHOLD, MASK_MOUTH_RATIO_THRESHOLD, MASK_MOUTH_SPREAD_THRESHOLD,
    YOLO_MODEL_PATH, TRIAGE_DETECTION_BACKEND
)

class Face:
    """Generic Face object to standardize results between backends"""
    def __init__(self, bbox, det_score, embedding=None, kps=None):
        self.bbox = bbox # [x1, y1, x2, y2]
        self.det_score = det_score
        self.embedding = embedding
        self.kps = kps

class FaceRecognitionHandler:
    def __init__(self, db_manager, lazy_sync=False): 
        # Configure ONNX Runtime globally for memory-efficient execution
        # Set environment variables BEFORE creating any ONNX sessions
        os.environ['ORT_ARENA_EXTEND_STRATEGY'] = 'kSameAsRequested'
        os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
        
        # Initialize InsightFace
        # Use local root for models if running from EXE
        model_root = resource_path("data/models")
        from config.config import EXECUTION_PROVIDERS
        self.app = FaceAnalysis(
            name=FACE_DETECTION_MODEL, 
            root=model_root,
            providers=EXECUTION_PROVIDERS
        )
        # Determine CUDA context id vs CPU/DirectML context id
        has_cuda = 'CUDAExecutionProvider' in str(EXECUTION_PROVIDERS)
        ctx_id = 0 if has_cuda else -1
        
        # Always prepare InsightFace with standard (640, 640) canvas for 100% face detection & landmark alignment
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self._mfr_app = None
        
        # Initialize Detection Backend
        self.backend = FACE_DETECTION_BACKEND
        self.triage_backend = TRIAGE_DETECTION_BACKEND
        self.net = None
        self._yolo_session = None
        
        from config.config import get_config
        conf = get_config()
        self.yolo_confidence_threshold = conf.get('yolo_confidence_threshold', 0.25)
        
        if self.backend == 'yolov8':
            from config.config import YOLO_MODEL_PATH, EXECUTION_PROVIDERS
            yolo_path = resource_path(os.path.join("data", "models", os.path.basename(YOLO_MODEL_PATH)))
            if not os.path.exists(yolo_path):
                self._download_yolo_model(yolo_path)
            if os.path.exists(yolo_path):
                import onnxruntime as ort
                print(f"Loading YOLOv8-Face Model from {yolo_path}...")
                self._yolo_session = ort.InferenceSession(yolo_path, providers=EXECUTION_PROVIDERS)
                print("YOLOv8-Face loaded successfully.")
            else:
                print("Failed to load YOLOv8-Face. Falling back to InsightFace.")
                self.backend = 'insightface'
        elif self.backend == 'opencv_dnn':
            try:
                print(f"Loading OpenCV DNN Model from {DNN_MODEL_PATH}...")
                self.net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
                print("OpenCV DNN loaded successfully.")
            except Exception as e:
                print(f"Error loading OpenCV DNN: {e}. Fallback to InsightFace.")
                self.backend = 'insightface'

        self.db_manager = db_manager
        self.similarity_threshold = SIMILARITY_THRESHOLD 
        
        # --- MASK DETECTION ---
        # detect_mask() uses HSV skin-color heuristic (V5 pipeline) — no extra model needed.
        # mask_detection_enabled is the only guard required.
        self.mask_detection_enabled = MASK_DETECTION_ENABLED
        self.masked_similarity_threshold = MASKED_SIMILARITY_THRESHOLD
        self.upper_face_crop_ratio = UPPER_FACE_CROP_RATIO
        self.mask_nose_ratio_threshold = MASK_NOSE_RATIO_THRESHOLD
        self.mask_mouth_ratio_threshold = MASK_MOUTH_RATIO_THRESHOLD
        if self.mask_detection_enabled:
            print("[MASK] Mask Detection enabled (HSV skin-color heuristic + buffalo_sc MFR).")


        if not lazy_sync:
            self.registered_faces = self.load_face_encodings()
        else:
            self.registered_faces = {}
            print("[DEBUG] FaceRecognitionHandler: Skipping sync (Lazy Mode).")

        # --- TRIAGE MODEL (Always Loaded for Speed) ---
        self.triage_net = None
        try:
            # Hardcoded paths for the Triage model (Lightweight SSD)
            # These are standard files usually in data/models/
            TRIAGE_PROTO = resource_path(os.path.join("data", "models", "deploy.prototxt"))
            TRIAGE_MODEL = resource_path(os.path.join("data", "models", "res10_300x300_ssd_iter_140000.caffemodel"))
            
            if os.path.exists(TRIAGE_PROTO) and os.path.exists(TRIAGE_MODEL):
                print(f"Loading Triage Model (SSD)...")
                self.triage_net = cv2.dnn.readNetFromCaffe(TRIAGE_PROTO, TRIAGE_MODEL)
                print("Triage Model Ready.")
            else:
                print("Triage Model files not found. Triage stage will be disabled (Passthrough).")
        except Exception as e:
            print(f"Error loading Triage Model: {e}")

    @property
    def mfr_app(self):
        if self._mfr_app is None:
            print("[INFO] Lazy-loading Masked Face Recognition model (buffalo_sc)...")
            model_root = resource_path("data/models")
            from config.config import EXECUTION_PROVIDERS, DETECTION_SIZE
            self._mfr_app = FaceAnalysis(
                name='buffalo_sc',
                root=model_root,
                providers=EXECUTION_PROVIDERS
            )
            from config.config import FACE_DETECTION_BACKEND
            if FACE_DETECTION_BACKEND != 'insightface':
                self._mfr_app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.35)
            else:
                self._mfr_app.prepare(ctx_id=-1, det_size=DETECTION_SIZE, det_thresh=0.35)
        return self._mfr_app

    def _download_yolo_model(self, dest_path):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            url = 'https://huggingface.co/deepghs/yolo-face/resolve/main/yolov8n-face/model.onnx'
            print(f"Downloading YOLOv8-Face model from {url}...")
            urllib.request.urlretrieve(url, dest_path)
            print(f"YOLOv8-Face model downloaded successfully to {dest_path}")
        except Exception as e:
            print(f"Error downloading YOLOv8-Face model: {e}")

    def quick_scan_faces(self, frame):
        """
        Stage 1: Fast Triage.
        Returns True if ANY face is detected (low threshold).
        Returns False if definitely empty.
        """
        if self.triage_backend == 'yolov8':
            if not hasattr(self, '_yolo_session') or self._yolo_session is None:
                from config.config import YOLO_MODEL_PATH, EXECUTION_PROVIDERS
                yolo_path = resource_path(os.path.join("data", "models", os.path.basename(YOLO_MODEL_PATH)))
                if not os.path.exists(yolo_path):
                    self._download_yolo_model(yolo_path)
                if os.path.exists(yolo_path):
                    import onnxruntime as ort
                    self._yolo_session = ort.InferenceSession(yolo_path, providers=EXECUTION_PROVIDERS)
            
            if self._yolo_session is not None:
                # Fast inference at 320x320 for speed
                h, w = frame.shape[:2]
                input_size = 320
                resized = cv2.resize(frame, (input_size, input_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                chw = rgb.transpose(2, 0, 1)
                input_tensor = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0
                
                outputs = self._yolo_session.run(None, {'images': input_tensor})
                predictions = outputs[0][0].T # (2100, 5) if 320x320
                
                # Check if any detection score > yolo_confidence_threshold
                conf_thresh = getattr(self, 'yolo_confidence_threshold', 0.25)
                if np.any(predictions[:, 4] > conf_thresh):
                    return True
                return False

        if self.triage_net is None:
            return True # Fallback: Proceed if no triage model
            
        (h, w) = frame.shape[:2]
        # Resize to 300x300 for speed (Standard SSD Input)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        
        self.triage_net.setInput(blob)
        # [DEBUG] Confirm SSD is running
        # print("SSD RUNNING", end='.', flush=True) 
        detections = self.triage_net.forward()
        
        # Check if any detection > 0.3 (Very permissive threshold)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                return True # Found potential face
                
        return False

    def detect_faces(self, frame):
        """Detect faces in a frame using InsightFace SCRFD."""
        faces = []
        if hasattr(self, 'app') and self.app is not None:
            faces = self.app.get(frame)

        if len(faces) == 0 and getattr(self, 'backend', 'insightface') == 'yolov8':
            faces = self._detect_faces_yolo(frame)
            
        return faces

    def _detect_faces_yolo(self, frame):
        """Detect faces using YOLOv8-Face ONNX model"""
        if not hasattr(self, '_yolo_session') or self._yolo_session is None:
            from config.config import YOLO_MODEL_PATH, EXECUTION_PROVIDERS, ORT_INTRA_OP_NUM_THREADS, ORT_INTER_OP_NUM_THREADS
            yolo_path = resource_path(os.path.join("data", "models", os.path.basename(YOLO_MODEL_PATH)))
            if not os.path.exists(yolo_path):
                self._download_yolo_model(yolo_path)
            
            if os.path.exists(yolo_path):
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = ORT_INTRA_OP_NUM_THREADS
                opts.inter_op_num_threads = ORT_INTER_OP_NUM_THREADS
                self._yolo_session = ort.InferenceSession(yolo_path, sess_options=opts, providers=EXECUTION_PROVIDERS)
            else:
                self._yolo_session = None
                
        if self._yolo_session is None:
            return []
            
        # Preprocessing
        h, w = frame.shape[:2]
        input_size = 640
        resized = cv2.resize(frame, (input_size, input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        chw = rgb.transpose(2, 0, 1)
        input_tensor = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0
        
        outputs = self._yolo_session.run(None, {'images': input_tensor})
        predictions = outputs[0]
        if predictions.ndim == 3:
            predictions = predictions[0]
        # Transpose to (N, C) if in (C, N) format (e.g. 5x8400 -> 8400x5)
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        
        # Filter by confidence (0.25 threshold to detect all faces)
        conf_thresh = getattr(self, 'yolo_confidence_threshold', 0.25)
        keep_idx = predictions[:, 4] > conf_thresh
        filtered = predictions[keep_idx]
        
        if len(filtered) == 0:
            return []
            
        boxes = []
        scores = []
        for pred in filtered:
            cx, cy, nw, nh, score = pred
            
            # Scale coordinates back to original frame
            x_scale = w / input_size
            y_scale = h / input_size
            
            cx *= x_scale
            cy *= y_scale
            nw *= x_scale
            nh *= y_scale
            
            x1 = cx - nw / 2
            y1 = cy - nh / 2
            x2 = cx + nw / 2
            y2 = cy + nh / 2
            
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Apply NMS
        keep = self._yolo_nms(boxes, scores, 0.45)
        
        faces = []
        for idx in keep:
            box = boxes[idx]
            score = scores[idx]
            
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w, int(box[2]))
            y2 = min(h, int(box[3]))
            
            f = Face(bbox=np.array([x1, y1, x2, y2]), det_score=score)
            faces.append(f)
            
        return faces

    def _yolo_nms(self, boxes, scores, iou_threshold):
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def _detect_faces_opencv(self, frame, target_size=(300, 300)):
        """Internal method for OpenCV DNN detection"""
        (h, w) = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, target_size), 1.0,
            target_size, (104.0, 177.0, 123.0))
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Lower default threshold slightly to catch more faces (filtered later by track thresh if needed)
            if confidence > 0.4: 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure within bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Create Face object (Standardized)
                f = Face(bbox=np.array([startX, startY, endX, endY]), det_score=confidence)
                faces.append(f)
                
        return faces
    
    def extract_face_encodings(self, frame):
        """Extract both standard and upper-face crop encodings from a frame."""
        faces = self.app.get(frame)
        h, w = frame.shape[:2]
        resized_frame = None
        
        if len(faces) == 0:
            if h < 300 or w < 300:
                print(f"Small image detected ({w}x{h}), upscaling for better detection...")
                scale = 640 / max(h, w)
                resized_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                faces = self.app.get(resized_frame)
        
        if len(faces) == 0:
            return None, None, "No face detected"
        
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
        std_embedding = faces[0].embedding
        mask_embedding = None
        
        if self.mask_detection_enabled:
            try:
                target_frame = resized_frame if resized_frame is not None else frame
                mask_embedding = self.extract_upper_face_encoding(target_frame, faces[0].bbox)
                if mask_embedding is not None:
                    print("[MASK] Successfully extracted registered upper face crop embedding.")
            except Exception as e:
                print(f"[MASK] Failed to extract upper face crop encoding: {e}")
        
        return std_embedding, mask_embedding, "Face encodings extracted successfully"

    def extract_face_encoding(self, frame):
        """Extract standard face encoding from a frame for backwards compatibility."""
        std_embedding, _, message = self.extract_face_encodings(frame)
        return std_embedding, message

    def extract_face_details_from_crop(self, face_crop):
        """
        Extract embedding and keypoints (kps) from a face crop.
        Handles dynamic upscaling for small crops to improve detection rate.
        Returns: (embedding, kps) or (None, None)
        """
        if face_crop is None or face_crop.size == 0:
            return None, None

        h, w = face_crop.shape[:2]
        faces = self.app.get(face_crop)
        scale = 1.0

        if len(faces) == 0:
            if h < 300 or w < 300:
                scale = 640.0 / max(h, w)
                resized_crop = cv2.resize(face_crop, (0, 0), fx=scale, fy=scale)
                faces = self.app.get(resized_crop)

        if len(faces) == 0:
            return None, None

        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)

        best_face = faces[0]
        embedding = best_face.embedding
        kps = None
        if hasattr(best_face, 'kps') and best_face.kps is not None:
            kps = best_face.kps / scale

        return embedding, kps
    
    def load_face_encodings(self):
        """Load face encodings with Hybrid Fallback (DB + Local File)"""
        from config.config import FACE_ENCODINGS_PATH
        import os
        import pickle
        import threading
        
        encodings = {}
        cache_loaded = False
        
        # 1. Try local cache first for instant startup
        if os.path.exists(FACE_ENCODINGS_PATH):
            try:
                with open(FACE_ENCODINGS_PATH, 'rb') as f:
                    encodings = pickle.load(f)
                print(f"[CACHE] Instantly loaded {len(encodings)} faces from LOCAL CACHE.")
                cache_loaded = True
            except Exception as e:
                print(f"[CACHE] Local cache corrupt: {e}")
                
        if cache_loaded:
            # Spawn background thread to sync from DB and update registered_faces
            if self.db_manager:
                def bg_sync():
                    try:
                        db_encodings = self.db_manager.get_all_face_encodings()
                        if db_encodings is not None:
                            # Strict overwrite with only active organization encodings
                            self.registered_faces = db_encodings.copy()
                            self._rebuild_encoding_matrix()
                            print(f"[DB BACKGROUND] Successfully synced {len(db_encodings)} faces for active organization.")
                            # Update local cache
                            try:
                                with open(FACE_ENCODINGS_PATH, 'wb') as f:
                                    pickle.dump(db_encodings, f)
                            except: pass
                    except Exception as e:
                        print(f"[DB BACKGROUND] Sync failed: {e}")
                
                threading.Thread(target=bg_sync, daemon=True).start()
            return encodings

        # 2. Blocking DB fetch fallback (if no cache exists)
        if self.db_manager:
            try:
                encodings = self.db_manager.get_all_face_encodings()
                if encodings:
                    print(f"[DB] Successfully synced {len(encodings)} faces from server.")
                    # Update local backup for next offline session
                    try:
                        with open(FACE_ENCODINGS_PATH, 'wb') as f:
                            pickle.dump(encodings, f)
                    except: pass
            except Exception as e:
                print(f"[DB] Connection Timeout or Error: {e}")
        
        return encodings
    
    def add_face_encoding(self, person_id, name, face_encoding, mask_face_encoding=None):
        """Add face encodings to the in-memory database"""
        self.registered_faces[person_id] = {
            'name': name,
            'encoding': face_encoding,
            'mask_encoding': mask_face_encoding
        }
        return True
    
    def remove_face_encoding(self, person_id):
        """Remove a face encoding from the database"""
        if person_id in self.registered_faces:
            del self.registered_faces[person_id]
            return True
        return False
    
    def _rebuild_encoding_matrix(self):
        """Rebuild vectorized numpy matrix of registered face encodings for ultra-fast dot-product similarity."""
        self._matrix_pids = []
        self._matrix_names = []
        std_list = []
        
        for person_id, data in self.registered_faces.items():
            enc = data.get('encoding')
            if enc is not None:
                arr = np.asarray(enc, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    std_list.append(arr / norm)
                    self._matrix_pids.append(person_id)
                    self._matrix_names.append(data.get('name', 'UNKNOWN'))
                    
        if std_list:
            self._std_matrix = np.array(std_list, dtype=np.float32) # (N, 512) matrix
        else:
            self._std_matrix = None

    def calculate_similarity(self, encoding1, encoding2):
        """Calculate cosine similarity between two face encodings"""
        if encoding1 is None or encoding2 is None:
            return 0.0
        e1 = np.asarray(encoding1, dtype=np.float32)
        e2 = np.asarray(encoding2, dtype=np.float32)
        if e1.shape != e2.shape:
            return 0.0
        n1 = np.linalg.norm(e1)
        n2 = np.linalg.norm(e2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))
    
    def recognize_face(self, face_encoding, is_masked=False, threshold=None):
        """Recognize a face using ultra-fast vectorized matrix similarity matching (<0.01ms execution)."""
        if face_encoding is None or len(self.registered_faces) == 0:
            return None, None, 0.0
            
        sim_threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Fast path: Vectorized 1-shot matrix dot product if not masked
        if not is_masked:
            if not hasattr(self, '_std_matrix') or self._std_matrix is None or len(self._matrix_pids) != len(self.registered_faces):
                self._rebuild_encoding_matrix()
                
            if getattr(self, '_std_matrix', None) is not None and len(self._matrix_pids) > 0:
                arr = np.asarray(face_encoding, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    unit_enc = arr / norm
                    sims = np.dot(self._std_matrix, unit_enc) # Matrix-vector multiplication (0.01ms)
                    best_idx = int(np.argmax(sims))
                    max_sim = float(sims[best_idx])
                    
                    if max_sim > sim_threshold:
                        return self._matrix_pids[best_idx], self._matrix_names[best_idx], max_sim
                    return None, None, max_sim
        
        # Fallback path for masked or non-matrix entries
        max_similarity = -1.0
        recognized_id = None
        recognized_name = None
        
        for person_id, data in self.registered_faces.items():
            target_encoding = data.get('mask_encoding') if is_masked else data.get('encoding')
            if target_encoding is None:
                continue
            similarity = self.calculate_similarity(face_encoding, target_encoding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > sim_threshold:
                    recognized_id = person_id
                    recognized_name = data['name']
        
        return recognized_id, recognized_name, max_similarity
    
    def recognize_multiple_faces(self, faces, frame=None):
        """Recognize multiple faces in a frame"""
        if self.mask_detection_enabled and frame is not None:
            return self.recognize_multiple_faces_masked(faces, frame)
        
        recognized_faces = []
        
        # Full-frame cache for accurate landmark-aligned embedding extraction
        full_faces_cache = None
        if frame is not None and any(f.embedding is None for f in faces):
            try:
                full_faces_cache = self.app.get(frame)
            except Exception as e:
                print(f"[ERROR] Full-frame InsightFace pass failed: {e}")

        for face in faces:
            # On-demand embedding extraction using full-frame alignment
            if face.embedding is None and frame is not None:
                try:
                    if full_faces_cache:
                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                        best_match = None
                        best_dist = float('inf')
                        for ff in full_faces_cache:
                            bx1, by1, bx2, by2 = ff.bbox
                            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                            dist = (fcx - bcx)**2 + (fcy - bcy)**2
                            if dist < best_dist:
                                best_dist = dist
                                best_match = ff
                        if best_match is not None:
                            face.embedding = best_match.embedding
                            if hasattr(best_match, 'kps'):
                                face.kps = best_match.kps
                    
                    # Direct crop embedding extraction from YOLO face bounding box
                    if face.embedding is None:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        h, w = frame.shape[:2]
                        face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if face_crop.size > 0:
                            crop_faces = self.app.get(face_crop)
                            if crop_faces and len(crop_faces) > 0:
                                face.embedding = crop_faces[0].embedding
                                if hasattr(crop_faces[0], 'kps'):
                                    face.kps = crop_faces[0].kps
                except Exception as e:
                    print(f"[ERROR] On-demand embedding extraction failed: {e}")

            person_id, person_name, similarity = self.recognize_face(face.embedding)
            
            recognized_faces.append({
                'bbox': face.bbox,
                'person_id': person_id,
                'person_name': person_name,
                'similarity': similarity,
                'det_score': face.det_score,
                'landmarks': face.kps if hasattr(face, 'kps') else None
            })
        
        return recognized_faces
    
    def verify_face(self, person_id, face_encoding):
        """Verify if a face encoding matches a specific person"""
        if person_id not in self.registered_faces:
            return False, 0.0
        
        similarity = self.calculate_similarity(
            face_encoding, 
            self.registered_faces[person_id]['encoding']
        )
        
        is_match = similarity > self.similarity_threshold
        return is_match, similarity
    
    def get_registered_count(self):
        """Get the number of registered faces"""
        return len(self.registered_faces)
    
    def get_all_registered_ids(self):
        """Get all registered person IDs"""
        return list(self.registered_faces.keys())
    
    def reload_face_encodings(self):
        """Reload face encodings from database"""
        self.registered_faces = self.load_face_encodings()
        try:
            from config.config import BASE_DIR
            log_path = os.path.join(BASE_DIR, 'data', 'registered_faces_check.log')
            with open(log_path, 'w') as lf:
                lf.write(f"Total registered faces loaded: {len(self.registered_faces)}\n")
                for pid, info in self.registered_faces.items():
                    name = info.get('name', 'N/A')
                    has_std = info.get('encoding') is not None
                    has_mask = info.get('mask_encoding') is not None
                    lf.write(f"  {pid}: {name} | std={has_std} | mask={has_mask}\n")
        except Exception as diag_e:
            print(f"DIAGNOSTIC LOG WRITE FAILED: {diag_e}")
        return len(self.registered_faces)
    
    def update_similarity_threshold(self, new_threshold):
        """Update the similarity threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.similarity_threshold = new_threshold
            return True
        return False

    # =========================================================================
    # MASK-AWARE RECOGNITION METHODS (New — existing methods are NOT modified)
    # =========================================================================

    def detect_mask(self, frame, face):
        """
        V5-identical HSV skin-color heuristic with bounding box fallback.
        Classifies whether a face has a mask by measuring skin-pixel ratio
        in the nose/mouth region. Runs in <1ms — no extra model needed.

        Args:
            frame: BGR frame.
            face:  Face object with .bbox, and optional .kps.

        Returns:
            bool: True if face is masked, False otherwise.
        """
        if not self.mask_detection_enabled:
            return False

        bbox = face.bbox.astype(int)
        bx1, by1, bx2, by2 = bbox[0], bbox[1], bbox[2], bbox[3]
        face_width = bx2 - bx1
        face_height = by2 - by1

        # --- FAST LANDMARK-BASED PRE-CHECK ---
        # If 5-point keypoints are available (left_eye, right_eye, nose, left_mouth, right_mouth),
        # use geometric ratios to quickly rule out unmasked faces BEFORE the HSV analysis.
        # This prevents false positives from shadows, dark skin, or poor lighting.
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 5:
            kps = face.kps.astype(float)
            eye_center_y = (kps[0][1] + kps[1][1]) / 2.0
            nose_y = kps[2][1]
            mouth_center_y = (kps[3][1] + kps[4][1]) / 2.0

            if face_height > 0:
                # Nose-to-eye ratio: if nose is clearly visible and well-separated from eyes,
                # the face is likely not masked (masked faces compress nose visibility)
                nose_eye_ratio = (nose_y - eye_center_y) / face_height
                # Mouth-drop ratio: if mouth landmarks are well below the nose,
                # the lower face is visible (unmasked)
                mouth_drop_ratio = (mouth_center_y - nose_y) / face_height

                # If both nose and mouth are clearly visible, this face is NOT masked
                if nose_eye_ratio > self.mask_nose_ratio_threshold and mouth_drop_ratio > self.mask_mouth_ratio_threshold:
                    return False

        if hasattr(face, 'kps') and face.kps is not None:
            kps  = face.kps.astype(int)
            # Crop lower face (nose-tip → chin), bounded horizontally by cheeks
            x1 = max(0, int(bx1 + face_width * 0.2))
            x2 = min(frame.shape[1], int(bx1 + face_width * 0.8))
            y1 = max(0, int(kps[2][1]))   # Nose-tip keypoint
            y2 = min(frame.shape[0], int(by2))  # Chin
        else:
            # Fallback when keypoints are missing (e.g. YOLOv8 backend)
            # Standard nose-tip is roughly at 50% of face height
            x1 = max(0, int(bx1 + face_width * 0.2))
            x2 = min(frame.shape[1], int(bx1 + face_width * 0.8))
            y1 = max(0, int(by1 + face_height * 0.5))
            y2 = min(frame.shape[0], int(by2))

        if x2 <= x1 or y2 <= y1:
            return False

        lower_face = frame[y1:y2, x1:x2]
        if lower_face.size == 0:
            return False

        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)

        # Skin colour range in HSV (covers all ethnicities)
        lower_skin = np.array([0,  15,  30], dtype=np.uint8)
        upper_skin = np.array([25, 200, 255], dtype=np.uint8)

        skin_mask   = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = cv2.countNonZero(skin_mask)
        total       = lower_face.shape[0] * lower_face.shape[1]

        if total == 0:
            return False

        skin_ratio = skin_pixels / total
        # Unmasked face: skin_ratio > 60%
        # Masked face:   skin_ratio < 35%
        # Tightened threshold from 0.45 -> 0.35 to reduce false positives
        # on unmasked faces with shadows or dark skin tones
        return skin_ratio < 0.35


    def extract_upper_face_encoding(self, frame, face_bbox):
        """
        Extract embedding from the UPPER portion of a face (eyes, eyebrows, forehead).
        This is used as a fallback when a mask is detected and full-face recognition fails.
        
        Args:
            frame: Full BGR frame.
            face_bbox: [x1, y1, x2, y2] bounding box of the detected face.
            
        Returns:
            numpy array or None: The 512-dim embedding from the upper face crop.
        """
        try:
            x1, y1, x2, y2 = map(int, face_bbox)
            h, w = frame.shape[:2]
            
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            face_height = y2 - y1
            face_width = x2 - x1
            
            if face_height < 30 or face_width < 30:
                return None
            
            # Crop upper portion of the face
            upper_y2 = y1 + int(face_height * self.upper_face_crop_ratio)
            upper_crop = frame[y1:upper_y2, x1:x2]
            
            if upper_crop.size == 0:
                return None
            
            # Pad the upper face crop to make it more square-ish for InsightFace.
            # InsightFace expects a roughly face-shaped input. We mirror the upper
            # portion downward to fill the "missing" lower face area.
            crop_h, crop_w = upper_crop.shape[:2]
            # Create a canvas the size of the original face
            padded = np.zeros((face_height, face_width, 3), dtype=np.uint8)
            padded[:crop_h, :crop_w] = upper_crop
            # Mirror the bottom of the upper crop to fill the lower region
            mirror_h = min(crop_h, face_height - crop_h)
            if mirror_h > 0:
                padded[crop_h:crop_h + mirror_h, :crop_w] = upper_crop[crop_h - mirror_h:crop_h, :crop_w][::-1]
            
            # Standardize face size directly to 112x112 (ArcFace native input size)
            padded = cv2.resize(padded, (112, 112))
            
            # Extract features directly from the recognition model, bypassing detection
            embedding = self.app.models['recognition'].get_feat(padded)
            if embedding is None:
                return None
            return embedding[0] if embedding.ndim == 2 else embedding
            
        except Exception as e:
            mask_log(f"[MASK] Upper face encoding error: {e}")
            return None

    def recognize_face_masked_direct(self, upper_face_embedding):
        """
        Compare the live upper-face crop embedding directly against the registered
        database upper-face mask encodings.
        
        Args:
            upper_face_embedding: 512-dim embedding of the live upper face crop.
            
        Returns:
            tuple: (person_id, person_name, similarity)
        """
        if upper_face_embedding is None or len(self.registered_faces) == 0:
            return None, None, 0.0
            
        max_similarity = -1.0
        recognized_id = None
        recognized_name = None
        
        for person_id, data in self.registered_faces.items():
            mask_encoding = data.get('mask_encoding')
            if mask_encoding is None:
                continue
            similarity = self.calculate_similarity(upper_face_embedding, mask_encoding)
            mask_log(f"[MASK] Upper-face similarity to {data.get('name')} ({person_id}): {similarity:.4f}")
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > self.masked_similarity_threshold:
                    recognized_id = person_id
                    recognized_name = data.get('name')
                    
        return recognized_id, recognized_name, max_similarity

    def recognize_face_masked(self, face_embedding, upper_face_embedding=None):
        """
        Attempt to recognize a face using the masked-face strategy:
        
        1. Compare the upper-face embedding against the registered mask encodings.
        2. If that fails (or upper-face embedding is missing), try standard full-face recognition.
        
        Args:
            face_embedding: Standard full-face 512-dim embedding.
            upper_face_embedding: Optional upper-face embedding (from cropped upper face).
            
        Returns:
            tuple: (person_id, person_name, similarity, recognition_method)
                   recognition_method is 'upper_face', 'full_face', or None
        """
        if upper_face_embedding is not None:
            person_id, person_name, similarity = self.recognize_face_masked_direct(upper_face_embedding)
            if person_id is not None:
                return person_id, person_name, similarity, 'upper_face'
            
        # Try standard full-face recognition as fallback
        person_id, person_name, similarity = self.recognize_face(face_embedding)
        if person_id is not None:
            return person_id, person_name, similarity, 'full_face'
            
        return None, None, similarity if similarity > 0 else 0.0, None

    def recognize_multiple_faces_masked(self, faces, frame=None):
        """
        Recognize multiple faces with mask-awareness.
        For each face: standard recognition first, then mask-detection + upper-face fallback.
        
        Args:
            faces: List of InsightFace face objects.
            frame: Full BGR frame (needed for upper-face crop extraction).
            
        Returns:
            list of dicts: Each dict contains bbox, person_id, person_name, similarity,
                          det_score, landmarks, is_masked, recognition_method
        """
        recognized_faces = []
        
        # Full-frame cache for accurate landmark-aligned embedding extraction
        full_faces_cache = None
        if frame is not None and any(f.embedding is None for f in faces):
            try:
                full_faces_cache = self.app.get(frame)
            except Exception as e:
                print(f"[ERROR] Full-frame InsightFace pass failed: {e}")

        for face in faces:
            # On-demand embedding extraction using full-frame alignment
            if face.embedding is None and frame is not None:
                try:
                    if full_faces_cache:
                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        fcx, fcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                        best_match = None
                        best_dist = float('inf')
                        for ff in full_faces_cache:
                            bx1, by1, bx2, by2 = ff.bbox
                            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                            dist = (fcx - bcx)**2 + (fcy - bcy)**2
                            if dist < best_dist:
                                best_dist = dist
                                best_match = ff
                        if best_match is not None:
                            face.embedding = best_match.embedding
                            if hasattr(best_match, 'kps'):
                                face.kps = best_match.kps
                    
                    # Fallback to crop if full-frame match wasn't found
                    if face.embedding is None:
                        x1, y1, x2, y2 = map(int, face.bbox)
                        h, w = frame.shape[:2]
                        face_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if face_crop.size > 0:
                            emb, kps = self.extract_face_details_from_crop(face_crop)
                            if emb is not None: face.embedding = emb
                            if kps is not None: face.kps = kps
                except Exception as e:
                    print(f"[ERROR] On-demand embedding extraction failed: {e}")

            # Standard full-face recognition attempt
            person_id, person_name, similarity = self.recognize_face(face.embedding)
            is_masked = False
            recognition_method = 'full_face' if person_id else None
            
            # If standard recognition FAILED, try mask-aware fallback
            # BUT skip entirely if:
            #  1. similarity is near-zero (face not in database at all — mask won't help)
            #  2. No registered faces have mask_encodings (upper-face matching would return -1.0)
            if person_id is None and self.mask_detection_enabled and frame is not None:
                # Fast exit: if standard similarity is below 0.10, the face isn't registered 
                # in the database at all. Mask detection can't identify someone who doesn't exist.
                if similarity < 0.10:
                    pass  # Skip mask fallback entirely — genuinely unknown face
                else:
                    # Detect if this face has a mask (Corrected signature and single return value)
                    is_masked = self.detect_mask(frame, face)
                    mask_log(f"[MASK] Standard recognition failed. Heuristic is_masked={is_masked}")
                    
                    if is_masked:
                        # 1. Try fast full-face recognition at the lower threshold first (0ms overhead)
                        pid, pname, sim = self.recognize_face(face.embedding, is_masked=False, threshold=self.masked_similarity_threshold)
                        if pid is not None:
                            person_id = pid
                            person_name = pname
                            similarity = sim
                            recognition_method = 'full_face_masked_threshold'
                            mask_log(f"[MASK] Fast match success on full-face standard embedding: {pname} ({pid}) | Sim: {sim:.4f}")
                        else:
                            # 2. Only attempt expensive upper-face extraction if at least one
                            #    registered face actually has a mask_encoding to compare against
                            has_mask_encodings = any(d.get('mask_encoding') is not None for d in self.registered_faces.values())
                            if has_mask_encodings:
                                upper_emb = self.extract_upper_face_encoding(frame, face.bbox)
                                if upper_emb is not None:
                                    mask_log(f"[MASK] Successfully extracted live upper face crop embedding.")
                                    pid, pname, sim = self.recognize_face_masked_direct(upper_emb)
                                    mask_log(f"[MASK] Direct upper-face match result: person_id={pid}, sim={sim:.4f}")
                                    if pid is not None:
                                        person_id = pid
                                        person_name = pname
                                        similarity = sim
                                        recognition_method = 'upper_face'
                                else:
                                    mask_log(f"[MASK] Failed to extract upper face embedding.")
                            else:
                                mask_log(f"[MASK] Skipping upper-face extraction (no mask_encodings registered).")
            
            recognized_faces.append({
                'bbox': face.bbox,
                'person_id': person_id,
                'person_name': person_name,
                'similarity': similarity,
                'det_score': face.det_score,
                'landmarks': face.kps if hasattr(face, 'kps') else None,
                'is_masked': is_masked,
                'recognition_method': recognition_method
            })
        
        return recognized_faces
