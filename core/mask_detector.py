"""
Mask Detector Module
====================
Lightweight face mask detection using InsightFace's existing 5-point face landmarks and HSV skin color ratios.

This module measures lower-face skin pixel ratios and landmark geometry to detect masked faces.
"""

import numpy as np
import cv2

class MaskDetector:
    """
    Detects whether a face is wearing a mask based on InsightFace landmark geometry and HSV skin color analysis.
    """

    def __init__(self, nose_ratio_threshold=0.30, mouth_ratio_threshold=0.22,
                 mouth_spread_threshold=0.35):
        self.nose_ratio_threshold = nose_ratio_threshold
        self.mouth_ratio_threshold = mouth_ratio_threshold
        self.mouth_spread_threshold = mouth_spread_threshold

    def detect_mask_hsv(self, face_crop):
        """
        Calculates lower-face HSV skin pixel ratio. Returns (is_masked, confidence).
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0.0
            
        try:
            h, w = face_crop.shape[:2]
            lower_face = face_crop[int(h*0.55):h, 0:w]
            if lower_face.size == 0:
                return False, 0.0
                
            hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_pixels = cv2.countNonZero(mask)
            total_pixels = lower_face.shape[0] * lower_face.shape[1]
            
            if total_pixels == 0:
                return False, 0.0
                
            skin_ratio = skin_pixels / float(total_pixels)
            is_masked = skin_ratio < 0.40
            confidence = 1.0 - skin_ratio
            return is_masked, float(confidence)
        except Exception:
            return False, 0.0
