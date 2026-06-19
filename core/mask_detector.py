"""
Mask Detector Module
====================
Lightweight face mask detection using InsightFace's existing 5-point face landmarks.

Strategy:
---------
InsightFace's `buffalo_l` model outputs 5 keypoints for every detected face:
    [0] Left Eye
    [1] Right Eye
    [2] Nose Tip
    [3] Left Mouth Corner
    [4] Right Mouth Corner

When a person wears a mask:
    1. The nose landmark shifts UPWARD (closer to the eye line) because the model
       "guesses" the nose position when the actual nose is occluded.
    2. The mouth landmarks COLLAPSE together and shift upward for the same reason.
    3. The vertical distance between the eye-line and nose/mouth shrinks significantly.

This module measures these geometric ratios and classifies the face as masked/unmasked.

No additional model files, pip packages, or downloads are required.
"""

import numpy as np


class MaskDetector:
    """
    Detects whether a face is wearing a mask based on InsightFace landmark geometry.
    
    Usage:
        detector = MaskDetector()
        is_masked, confidence = detector.detect_mask(face)
    """

    def __init__(self, nose_ratio_threshold=0.30, mouth_ratio_threshold=0.22,
                 mouth_spread_threshold=0.35):
        """
        Args:
            nose_ratio_threshold: Maximum allowed ratio of (eye_to_nose / eye_to_eye)
                                  before classifying as masked. Lower = more sensitive.
            mouth_ratio_threshold: Maximum allowed ratio of (mouth_vertical_spread / face_height)
                                   before classifying as masked.
            mouth_spread_threshold: Maximum allowed ratio of (mouth_width / eye_width)
                                    before classifying as masked. Collapsed mouth = mask.
        """
        self.nose_ratio_threshold = nose_ratio_threshold
        self.mouth_ratio_threshold = mouth_ratio_threshold
        self.mouth_spread_threshold = mouth_spread_threshold

    def detect_mask(self, face):
        """
        Analyze a single InsightFace detection result to determine if a mask is present.
        
        Args:
            face: An InsightFace face object with `.kps` (5x2 numpy array of landmarks)
                  and `.bbox` ([x1, y1, x2, y2]).
                  
        Returns:
            tuple: (is_masked: bool, confidence: float)
                   confidence ranges from 0.0 (definitely no mask) to 1.0 (definitely masked)
        """
        # Guard: If no landmarks available, we can't determine mask status
        if not hasattr(face, 'kps') or face.kps is None:
            return False, 0.0

        kps = face.kps
        if len(kps) < 5:
            return False, 0.0

        # Extract the 5 keypoints
        left_eye = kps[0]      # (x, y)
        right_eye = kps[1]     # (x, y)
        nose = kps[2]          # (x, y)
        left_mouth = kps[3]    # (x, y)
        right_mouth = kps[4]   # (x, y)

        # --- Metric 1: Eye-to-Nose Vertical Ratio ---
        # Normal face: nose is roughly 40-50% of the way down from the eyes.
        # Masked face: model predicts nose closer to eye line (~25-35%).
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
        eye_distance = np.linalg.norm(left_eye - right_eye)

        if eye_distance < 1.0:
            # Eyes too close together (bad detection), skip
            return False, 0.0

        nose_drop = nose[1] - eye_center_y  # Positive = nose is below eyes (normal)
        nose_ratio = nose_drop / eye_distance

        # --- Metric 2: Mouth Vertical Position ---
        # Normal: mouth is well below the nose.
        # Masked: mouth landmarks collapse toward the nose.
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2.0
        bbox = face.bbox
        face_height = bbox[3] - bbox[1]

        if face_height < 1.0:
            return False, 0.0

        # How far mouth is below the eye line, normalized by face height
        mouth_drop_ratio = (mouth_center_y - eye_center_y) / face_height

        # --- Metric 3: Mouth Horizontal Spread ---
        # Normal: mouth corners are spread wide (roughly 60-80% of eye distance).
        # Masked: mouth corners collapse together.
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        mouth_spread_ratio = mouth_width / eye_distance

        # --- Scoring ---
        # Each metric contributes a score. Higher total = more likely masked.
        score = 0.0
        max_score = 3.0

        # Score 1: Nose is abnormally close to eye line
        if nose_ratio < self.nose_ratio_threshold:
            score += 1.0
        elif nose_ratio < self.nose_ratio_threshold * 1.3:
            # Partial score for borderline cases
            score += 0.5

        # Score 2: Mouth is abnormally close to eye line (hasn't dropped far enough)
        if mouth_drop_ratio < self.mouth_ratio_threshold:
            score += 1.0
        elif mouth_drop_ratio < self.mouth_ratio_threshold * 1.3:
            score += 0.5

        # Score 3: Mouth corners have collapsed together
        if mouth_spread_ratio < self.mouth_spread_threshold:
            score += 1.0
        elif mouth_spread_ratio < self.mouth_spread_threshold * 1.3:
            score += 0.5

        confidence = score / max_score
        is_masked = confidence >= 0.5  # At least 2 of 3 metrics triggered

        return is_masked, confidence

    def detect_mask_from_bbox(self, face, frame=None):
        """
        Extended mask detection that also considers color analysis in the lower face region.
        Falls back to landmark-only detection if frame is not provided.
        
        Args:
            face: InsightFace face object with landmarks and bbox.
            frame: Optional BGR frame for color-based analysis of the lower face.
            
        Returns:
            tuple: (is_masked: bool, confidence: float)
        """
        # Landmark-based detection
        is_masked_landmark, landmark_conf = self.detect_mask(face)

        if frame is None:
            return is_masked_landmark, landmark_conf

        # Secondary: Hybrid Color uniformity check on lower face
        # This is executed regardless of landmark result to catch masks on CCTV streams
        # where camera angles skew landmark predictions.
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face_h = y2 - y1
            face_w = x2 - x1
            if face_h < 20 or face_w < 20:
                return is_masked_landmark, landmark_conf

            # Crop lower 45% of face (mask region)
            lower_y1 = y1 + int(face_h * 0.55)
            lower_crop = frame[lower_y1:y2, x1:x2]

            if lower_crop.size == 0:
                return is_masked_landmark, landmark_conf

            # Check color uniformity of mask region
            std_dev_lower = np.mean(np.std(lower_crop, axis=(0, 1)))

            # Crop upper 45% of face (eye & forehead region) for relative baseline comparison
            upper_y2 = y1 + int(face_h * 0.45)
            upper_crop = frame[y1:upper_y2, x1:x2]
            
            if upper_crop.size > 0:
                std_dev_upper = np.mean(np.std(upper_crop, axis=(0, 1)))
            else:
                std_dev_upper = 25.0  # Safe default baseline
                
            # If standard deviation of lower face is low (uniform color)
            # OR the ratio of lower std dev to upper std dev is small, it's a mask.
            is_color_uniform = std_dev_lower < 16.0 or (std_dev_upper > 10.0 and (std_dev_lower / std_dev_upper) < 0.6)
            
            if is_color_uniform:
                # Boost confidence: color evidence strongly suggests mask
                combined_conf = max(landmark_conf, 0.7)
                return True, combined_conf

            # If landmark check says True but color was not uniform (e.g. skin-colored mask, busy pattern),
            # we still respect the landmark check if it was confident.
            if is_masked_landmark:
                return True, landmark_conf

        except Exception as e:
            # Safely log warning and fallback to landmarks
            print(f"[MASK] Warning: Color-ratio check error: {e}")
            pass

        return is_masked_landmark, landmark_conf

    def detect_masks_batch(self, faces, frame=None):
        """
        Process multiple face detections at once.
        
        Args:
            faces: List of InsightFace face objects.
            frame: Optional BGR frame for enhanced detection.
            
        Returns:
            list of tuples: [(is_masked, confidence), ...]
        """
        results = []
        for face in faces:
            if frame is not None:
                result = self.detect_mask_from_bbox(face, frame)
            else:
                result = self.detect_mask(face)
            results.append(result)
        return results
