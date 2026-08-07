import cv2
import numpy as np
import supervision as sv
import os
import time
from datetime import datetime
from config.config import (
    UNKNOWN_FACES_DIR, SNAPSHOTS_DIR, EXECUTION_PROVIDERS
)

class VideoProcessor:
    def __init__(self, face_handler):
        self.face_handler = face_handler
        
        # Initialize ByteTrack
        # Updated arguments for newer supervision versions
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.5,       
                lost_track_buffer=30,        
                minimum_matching_threshold=0.8,       
                frame_rate=30
            )
        except TypeError:
            # Fallback for older versions
            self.tracker = sv.ByteTrack(
                track_thresh=0.5,       
                track_buffer=30,        
                match_thresh=0.8,       
                frame_rate=30
            )
        
        # Initialize Annotator
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            # text_thickness=1, # Removed for compatibility
            # text_scale=0.5,   # Removed for compatibility
            # text_padding=5    # Removed for compatibility
        )
        
        # Track recognized faces (The Cache)
        self.tracker_id_to_person = {}
        
        # Track logged unknown faces to prevent duplicate logging
        self.logged_unknown_ids = set()
        
        # Ensure unknown faces directory exists
        if not os.path.exists(UNKNOWN_FACES_DIR):
            os.makedirs(UNKNOWN_FACES_DIR)

        if not os.path.exists(SNAPSHOTS_DIR):
            os.makedirs(SNAPSHOTS_DIR)


    
    def clear_cache(self):
        """Forces the processor to forget currently tracked faces"""
        self.tracker_id_to_person = {}
        self.logged_unknown_ids = set()

    def process_frame(self, frame, mark_attendance_callback=None, unknown_person_callback=None):
        """
        Process a single frame for Face Recognition and Tracking.
        Returns: (detections, labels, faces, messages)
        """
        from config.config import PROCESS_EVERY_N_FRAMES, RESIZE_FACTOR, SHOW_DETECTION_SCORE
        
        # Initialize frame counter if not exists
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
            self.last_detections = sv.Detections.empty()
            self.last_faces = []
        
        self.frame_count += 1
        
        # --- PROCESS EVERY Nth FRAME ---
        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0:
            
            # 1. Resize for faster inference
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            
            # 2. Detect faces on small frame
            faces = self.face_handler.detect_faces(small_frame)
            
            # 3. Scale back coordinates to original size
            for face in faces:
                face.bbox = face.bbox / RESIZE_FACTOR
                if hasattr(face, 'kps') and face.kps is not None:
                    face.kps = face.kps / RESIZE_FACTOR
            
            self.last_faces = faces
            
            # 4. Format detections for ByteTrack
            if len(faces) > 0:
                xyxy = np.array([face.bbox for face in faces])
                confidence = np.array([face.det_score for face in faces])
                class_id = np.zeros(len(faces), dtype=int)
                
                detections = sv.Detections(
                    xyxy=xyxy, 
                    confidence=confidence, 
                    class_id=class_id
                )
            else:
                detections = sv.Detections.empty()
            
            # 5. Update tracker
            tracked_detections = self.tracker.update_with_detections(detections)
            self.last_detections = tracked_detections
            
        else:
            # --- SKIP FRAME: USE LAST KNOWN DETECTIONS ---
            faces = self.last_faces
            tracked_detections = self.last_detections

        if tracked_detections.tracker_id is not None:
            tracked_detections.class_id = tracked_detections.tracker_id.astype(int)
        
        labels = []
        messages = []
        
        # Loop through tracked detections
        for i in range(len(tracked_detections)):
            if tracked_detections.tracker_id is None: continue
            
            tracker_id = tracked_detections.tracker_id[i]
            current_bbox = tracked_detections.xyxy[i]
            
            # --- CASE 1: EXISTING TRACK (We already know who this is) ---
            if tracker_id in self.tracker_id_to_person:
                person_id, person_name = self.tracker_id_to_person[tracker_id]
                
                # Get confidence from tracker if available or just use visual cue
                conf = tracked_detections.confidence[i] if hasattr(tracked_detections, 'confidence') and tracked_detections.confidence is not None else 0.0
                score_str = f" {conf:.2f}" if SHOW_DETECTION_SCORE else ""
                
                label = f"{person_name} ({person_id}){score_str}"
                
                # Update attendance (Only on processed frames to save DB calls)
                if self.frame_count % PROCESS_EVERY_N_FRAMES == 0 and mark_attendance_callback:
                    # --- CAPTURE SNAPSHOT (EMPLOYEE) ---
                    snapshot_path = self._save_face_snapshot(frame, current_bbox, person_id, SNAPSHOTS_DIR)
                    
                    success, message = mark_attendance_callback(person_id, person_name, snapshot_path)
                    if success and message and "Tracking" not in message:
                        messages.append(message)

            # --- CASE 2: NEW TRACK (We need to recognize the face) ---
            else:
                # Only run recognition on the processed frame to save resources
                if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                     label = f"Tracking #{tracker_id}"
                else:
                    best_face = None
                    max_iou = 0.0
                    
                    # Find matching face detection
                    for face in faces:
                        iou = self.calculate_iou(current_bbox, face.bbox)
                        if iou > 0.5 and iou > max_iou:
                            max_iou = iou
                            best_face = face

                    if best_face:
                            
                        # --- HYBRID BACKEND SUPPORT ---
                        # If detecting with OpenCV DNN, embedding is None. We must extract it now.
                        if best_face.embedding is None:
                            x1, y1, x2, y2 = map(int, best_face.bbox)
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            face_crop = frame[y1:y2, x1:x2]
                            
                            if face_crop.size > 0:
                                emb, kps = self.face_handler.extract_face_details_from_crop(face_crop)
                                if emb is not None:
                                    best_face.embedding = emb
                                if kps is not None:
                                    best_face.kps = kps + np.array([x1, y1])

                        # Check against the pickle file
                        if best_face.embedding is not None:
                            person_id, person_name, similarity = self.face_handler.recognize_face(best_face.embedding)
                            
                            # --- MASK-AWARE FALLBACK ---
                            # If standard recognition failed, check for mask and retry with upper-face
                            if not person_id and self.face_handler.mask_detection_enabled:
                                is_masked = self.face_handler.detect_mask(frame, best_face)  # V5: (frame, face), returns bool
                                
                                if is_masked:
                                    # 1. Try fast full-face recognition at the lower threshold first (0ms overhead)
                                    m_id, m_name, m_sim = self.face_handler.recognize_face(
                                        best_face.embedding, is_masked=False, threshold=self.face_handler.masked_similarity_threshold
                                    )
                                    if m_id:
                                        person_id = m_id
                                        person_name = m_name
                                        similarity = m_sim
                                        rec_method = 'full_face_masked_threshold'
                                    else:
                                        # 2. Fallback to extracting upper face crop embedding and matching
                                        upper_emb = self.face_handler.extract_upper_face_encoding(frame, best_face.bbox)
                                        if upper_emb is not None:
                                            person_id, person_name, similarity, rec_method = \
                                                self.face_handler.recognize_face_masked(best_face.embedding, upper_emb)
                                        
                                    if person_id:
                                        self.tracker_id_to_person[tracker_id] = (person_id, person_name)
                                        
                                        score_str = f" {best_face.det_score:.2f}" if SHOW_DETECTION_SCORE else ""
                                        label = f"\U0001f637 {person_name} ({person_id}){score_str}"
                                        
                                        if mark_attendance_callback:
                                            snapshot_path = self._save_face_snapshot(frame, current_bbox, person_id, SNAPSHOTS_DIR)
                                            success, message = mark_attendance_callback(person_id, person_name, snapshot_path)
                                            if success and message:
                                                messages.append(message)
                                        
                                        labels.append(label)
                                        continue
                            
                            if person_id:
                                self.tracker_id_to_person[tracker_id] = (person_id, person_name)

                                score_str = f" {best_face.det_score:.2f}" if SHOW_DETECTION_SCORE else ""
                                label = f"{person_name} ({person_id}){score_str}"

                                if mark_attendance_callback:
                                    # --- CAPTURE SNAPSHOT (EMPLOYEE) ---
                                    snapshot_path = self._save_face_snapshot(frame, current_bbox, person_id, SNAPSHOTS_DIR)

                                    success, message = mark_attendance_callback(person_id, person_name, snapshot_path)
                                    if success and message:
                                        messages.append(message)
                            else:
                                score_str = f" {best_face.det_score:.2f}" if SHOW_DETECTION_SCORE else ""
                                label = f"Unknown #{tracker_id}{score_str}"
                        else:
                             label = f"Tracking #{tracker_id} (No Emb)"

                    else:
                        label = f"Tracking #{tracker_id}"
            
            labels.append(label)
        
        return tracked_detections, labels, faces, messages

    def annotate_frame(self, frame, detections, labels, faces):
        """
        Draw bounding boxes, labels, and landmarks on the frame.
        This allows the UI to draw on the *latest* frame using the *latest known* data.
        """
        # Annotate boxes and labels
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            labels=labels
        )
        
        # Draw landmarks
        annotated_frame = self.draw_landmarks(annotated_frame, faces)
        
        return annotated_frame
    
    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def draw_landmarks(self, frame, faces):
        for face in faces:
            if hasattr(face, 'kps') and face.kps is not None:
                for point in face.kps:
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
        return frame
    
    def draw_info_panel(self, frame, info_dict):
        panel_height = 80
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        y_offset = 25
        x_offset = 20
        for key, value in info_dict.items():
            text = f"{key}: {value}"
            cv2.putText(panel, text, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        return np.vstack([panel, frame])
    
    def add_fps_counter(self, frame, fps):
        text = f"FPS: {fps:.1f}"
        cv2.putText(frame, text, (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame
        
    def get_detection_count(self, tracked_detections):
        if tracked_detections.tracker_id is not None:
            return len(tracked_detections.tracker_id)
        return 0


    def _save_face_snapshot(self, frame, bbox, person_id, directory):
        """Helper to save face crop"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{person_id}_{timestamp}.jpg"
            filepath = os.path.join(directory, filename)
            
            x1, y1, x2, y2 = map(int, bbox)
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                cv2.imwrite(filepath, face_crop)
                return filepath
        except Exception as e:
            print(f"Snapshot Error: {e}")
        return None