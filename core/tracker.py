import time
import numpy as np
from config.config import (
    TRACK_TIMEOUT_SECONDS,
    MATCHING_IOU_WEIGHT,
    MATCHING_CENTROID_WEIGHT
)
from core.track_manager import TrackManager, Track, TrackVisibility

class DetectionResult:
    """
    Decoupled Perception Layer Detection Result object.
    Abstracts face/person bounding box detections away from underlying CV engine 
    (OpenCV DNN, YOLO, ONNX, EdgeTPU).
    """
    def __init__(self, bounding_box, confidence=1.0, class_id=0):
        self.bounding_box = list(bounding_box)  # [bx1, by1, bx2, by2]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        
        cx = (bounding_box[0] + bounding_box[2]) / 2.0
        cy = (bounding_box[1] + bounding_box[3]) / 2.0
        self.centroid = (cx, cy)


class MultiPersonTracker:
    """
    Multi-Person Target Tracker per camera stream slot.
    Performs Weighted Matching (0.7 * IoU + 0.3 * CentroidDistance) to eliminate
    track ID switching on fast target movements.
    """
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.next_track_id = 101
        self.track_manager = TrackManager()

    @staticmethod
    def _calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / float(union + 1e-6)

    def _calculate_matching_score(self, box, track_box, frame_w, frame_h):
        """
        Calculates weighted similarity score:
        Score = 0.7 * IoU + 0.3 * (1.0 - CentroidDistance / MaxDistance)
        """
        iou = self._calculate_iou(box, track_box)
        
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        tcx = (track_box[0] + track_box[2]) / 2.0
        tcy = (track_box[1] + track_box[3]) / 2.0
        
        dist = ((cx - tcx)**2 + (cy - tcy)**2)**0.5
        max_dist = max(frame_w, frame_h) * 0.50
        norm_dist_score = max(0.0, 1.0 - (dist / max_dist))
        
        score = (MATCHING_IOU_WEIGHT * iou) + (MATCHING_CENTROID_WEIGHT * norm_dist_score)
        return score, iou, dist

    def update(self, detections, now, frame=None):
        """
        Updates active tracks with frame detections (list of DetectionResult or box lists).
        Returns active Track objects for this camera.
        """
        # Convert raw box lists to DetectionResult objects if needed
        det_objects = []
        for d in detections:
            if isinstance(d, DetectionResult):
                det_objects.append(d)
            elif isinstance(d, (list, tuple)) and len(d) >= 4:
                det_objects.append(DetectionResult(d[:4]))

        # Expire tracks absent longer than TRACK_TIMEOUT_SECONDS
        self.track_manager.expire_stale_tracks(now, timeout_seconds=TRACK_TIMEOUT_SECONDS)

        # Retrieve active tracks for this camera
        with self.track_manager.lock:
            active_tracks = [
                t for (c_id, t_id), t in self.track_manager.tracks.items()
                if c_id == self.camera_id
            ]

        matched_track_ids = set()
        matched_det_indices = set()

        frame_w = frame.shape[1] if frame is not None else 1280
        frame_h = frame.shape[0] if frame is not None else 720

        # Weighted Matching
        for d_idx, det in enumerate(det_objects):
            best_track = None
            best_score = 0.0
            
            for t in active_tracks:
                if t.track_id in matched_track_ids:
                    continue
                score, iou, dist = self._calculate_matching_score(det.bounding_box, t.bounding_box, frame_w, frame_h)
                
                # Match threshold: score >= 0.25 (or IoU > 0.1)
                if (score >= 0.25 or iou > 0.10) and score > best_score:
                    best_score = score
                    best_track = t

            if best_track:
                best_track.update_spatial(det.bounding_box, now, frame)
                matched_track_ids.add(best_track.track_id)
                matched_det_indices.add(d_idx)

        # Register new Track objects for unmatched detections
        for d_idx, det in enumerate(det_objects):
            if d_idx not in matched_det_indices:
                tid = self.next_track_id
                self.next_track_id += 1
                
                new_track = Track(tid, self.camera_id, det.bounding_box, now)
                if frame is not None:
                    new_track.last_frame = frame.copy()
                    
                self.track_manager.register_track(new_track)

        # Return active tracks for this camera
        with self.track_manager.lock:
            return [
                t for (c_id, t_id), t in self.track_manager.tracks.items()
                if c_id == self.camera_id and t.visibility == TrackVisibility.VISIBLE
            ]
