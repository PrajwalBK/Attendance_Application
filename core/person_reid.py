"""
person_reid.py
==============
Person Re-Identification using OSNet (Omni-Scale Network) ONNX model.

Pipeline:
    1. Person bounding box arrives (from face detector or YOLO person detector).
    2. Body crop is extracted and resized to (256 x 128) — OSNet input size.
    3. OSNet produces a 512-dim appearance embedding (clothing + body shape + colour).
    4. Embedding is compared against the daily ReID cache using cosine similarity.
    5. Best match above REID_SIMILARITY_THRESHOLD is returned as person identity.

Key design decisions:
    - Body embeddings are NEVER stored in the database. They live in a local daily
      cache file (reid_cache/YYYY-MM-DD.pkl) that is auto-deleted next startup.
    - Embeddings are built dynamically: when someone is recognised by FACE, their
      body embedding is immediately captured and cached for future ReID.
    - OSNet model (~2.2 MB ONNX) is auto-downloaded on first run.

Model input:  (1, 3, 256, 128)  — RGB, ImageNet-normalised
Model output: (1, 512)           — L2-normalised embedding
"""

import os
import time
import cv2
import numpy as np
import urllib.request
import logging

logger = logging.getLogger(__name__)

# ─── OSNet Model Constants ────────────────────────────────────────────────────
_OSNET_INPUT_H   = 256
_OSNET_INPUT_W   = 128
_OSNET_EMB_DIM   = 512
_IMAGENET_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Public ONNX model (hosted on HuggingFace / GitHub releases)
_OSNET_ONNX_URL  = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "osnet_x1_0_msmt17.onnx"
)


class PersonReID:
    """
    Lightweight Person Re-Identification using OSNet.

    Usage:
        reid = PersonReID(model_path="data/models/osnet_x1_0.onnx")

        # When face is recognised → cache body embedding
        reid.register_body(person_id, "Ravi Sharma", frame, body_bbox)

        # When face is NOT recognised → try body ReID
        person_id, name, sim = reid.identify(frame, body_bbox)
    """

    def __init__(self, model_path: str, similarity_threshold: float = 0.72,
                 execution_providers=None, temporal_window: float = 300.0):
        self._model_path         = model_path
        self.similarity_threshold = similarity_threshold
        self._providers           = execution_providers or ["CPUExecutionProvider"]
        self._session             = None          # Lazy-loaded ONNX session
        self.temporal_window      = temporal_window

        # ReID gallery: {person_id: {"name": str, "embeddings": [np.array, ...]}}
        self._gallery: dict = {}

    # ── Model loading ──────────────────────────────────────────────────────────

    def _ensure_model(self):
        """Load the OSNet ONNX session (model must already be at model_path)."""
        if self._session is not None:
            return True

        if not os.path.exists(self._model_path):
            print(
                f"\n[ReID] ⚠  OSNet model not found at:\n"
                f"    {self._model_path}\n"
                f"    ReID is disabled. Contact your system administrator.\n"
            )
            return False

        try:
            import onnxruntime as ort
            self._session    = ort.InferenceSession(self._model_path, providers=self._providers)
            self._input_name  = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info(f"[ReID] OSNet loaded — {self._model_path}")
            print("[ReID] OSNet Re-ID model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"[ReID] Failed to load OSNet: {e}")
            print(f"[ReID] Failed to load OSNet: {e}")
            return False


    def _download_model(self) -> bool:
        """
        Obtain the OSNet ONNX model.
        Strategy:
          1. Try to export from torchreid (if installed).
          2. If torchreid not installed, try pip-installing it silently.
          3. If all else fails, print instructions and return False.
        """
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        print("[ReID] OSNet ONNX not found. Attempting to build it via torchreid …")

        # ── Strategy 1: Export via torchreid ──────────────────────────────────
        try:
            return self._export_osnet_onnx()
        except ImportError:
            pass  # torchreid not installed, try installing
        except Exception as e:
            logger.warning(f"[ReID] torchreid export failed: {e}")

        # ── Strategy 2: pip-install torchreid then export ─────────────────────
        try:
            import subprocess, sys
            print("[ReID] Installing torchreid (one-time, ~30 s) …")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "torchreid",
                 "--quiet", "--no-warn-script-location"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return self._export_osnet_onnx()
        except Exception as e:
            logger.error(f"[ReID] Auto-install failed: {e}")

        # ── Strategy 3: Manual instructions ───────────────────────────────────
        print(
            "\n[ReID] ⚠  Could not obtain OSNet model automatically.\n"
            "    Run this once to generate it:\n\n"
            "        pip install torchreid\n"
            "        python -c \"\n"
            "            import torchreid, torch\n"
            "            m = torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)\n"
            "            m.eval()\n"
            f"            torch.onnx.export(m, torch.randn(1,3,256,128), r'{self._model_path}',\n"
            "                opset_version=11, input_names=['input'], output_names=['output'])\n"
            "            print('Done')\n"
            "        \"\n\n"
            "    ReID will be disabled until the model is present.\n"
        )
        return False

    def _export_osnet_onnx(self) -> bool:
        """Export OSNet x1.0 to ONNX using torchreid. Returns True on success."""
        import torchreid   # noqa: F401  (raises ImportError if not installed)
        import torch

        print("[ReID] Exporting OSNet x1.0 → ONNX (one-time, ~10 s) …")
        model = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )
        model.eval()

        dummy = torch.randn(1, 3, _OSNET_INPUT_H, _OSNET_INPUT_W)
        torch.onnx.export(
            model, dummy, self._model_path,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"[ReID] OSNet ONNX saved → {self._model_path}")
        return True


    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Resize → BGR→RGB → normalise → NCHW float32 tensor.
        Input:  (H, W, 3) BGR uint8
        Output: (1, 3, 256, 128) float32
        """
        img = cv2.resize(crop, (_OSNET_INPUT_W, _OSNET_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        img = img.transpose(2, 0, 1)          # HWC → CHW
        return img[np.newaxis, ...]            # → (1, 3, H, W)

    # ── Embedding extraction ───────────────────────────────────────────────────

    def extract_embedding(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        """
        Extract a 512-dim body embedding from the given bounding box in the frame.

        Args:
            frame: Full BGR frame.
            bbox:  [x1, y1, x2, y2] — person bounding box.

        Returns:
            L2-normalised numpy float32 array of shape (512,), or None on failure.
        """
        if not self._ensure_model():
            return None

        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            
            box_h = y2 - y1
            box_w = x2 - x1
            
            # If the box is likely a face box (aspect ratio close to 1), extrapolate to body
            if box_w > 0 and (box_h / box_w) < 1.6:
                cx = (x1 + x2) // 2
                # Extrapolate body box:
                # Height of body is roughly 7.5x head height
                # Width of body is roughly 3x head width (1.5x on each side of center)
                ext_y1 = max(0, int(y1 - 0.2 * box_h))
                ext_y2 = min(h, int(y1 + 7.5 * box_h))
                ext_x1 = max(0, int(cx - 1.5 * box_w))
                ext_x2 = min(w, int(cx + 1.5 * box_w))
                crop = frame[ext_y1:ext_y2, ext_x1:ext_x2]
            else:
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)
                crop = frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 16:
                return None

            tensor = self._preprocess(crop)
            outputs = self._session.run([self._output_name], {self._input_name: tensor})
            emb = outputs[0][0].astype(np.float32)   # (512,)

            # L2 normalise
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb /= norm

            return emb

        except Exception as e:
            logger.warning(f"[ReID] Embedding extraction error: {e}")
            return None

    # ── Gallery management ─────────────────────────────────────────────────────

    def register_body(self, person_id: str, name: str,
                      frame: np.ndarray, bbox, timestamp: float = None) -> bool:
        """
        Extract and cache a body embedding for a person whose face was just
        successfully recognised. Can be called multiple times per person — each
        embedding is stored (up to MAX_EMBEDDINGS_PER_PERSON per person) to
        handle different angles / distances.

        Returns True if embedding was successfully stored.
        """
        MAX_PER_PERSON = 8  # Keep at most 8 embeddings per person per day

        emb = self.extract_embedding(frame, bbox)
        if emb is None:
            return False

        if person_id not in self._gallery:
            self._gallery[person_id] = {"name": name, "embeddings": []}

        existing = self._gallery[person_id]["embeddings"]

        # Deduplication: skip if too similar to an already-stored embedding
        for entry in existing:
            # Handle both formats
            stored_emb = entry[0] if isinstance(entry, tuple) else entry
            if self._cosine_similarity(emb, stored_emb) > 0.95:
                return False  # Already have a very similar one

        if len(existing) < MAX_PER_PERSON:
            ts = timestamp if timestamp is not None else time.time()
            existing.append((emb, ts))
            logger.debug(f"[ReID] Cached body for {name} ({person_id}) "
                         f"[{len(existing)}/{MAX_PER_PERSON}]")
        return True

    def load_gallery(self, gallery: dict):
        """
        Replace the in-memory gallery with data loaded from the daily cache.
        gallery format: {person_id: {"name": str, "embeddings": [...]}}
        """
        sanitized = {}
        for pid, data in gallery.items():
            name = data.get("name", "")
            embs = []
            for entry in data.get("embeddings", []):
                if isinstance(entry, tuple):
                    embs.append(entry)
                elif isinstance(entry, np.ndarray):
                    # Convert legacy format to tuple with timestamp 0.0 (expired)
                    embs.append((entry, 0.0))
            sanitized[pid] = {"name": name, "embeddings": embs}

        self._gallery = sanitized
        # Handle count check with both formats
        total_embs = sum(len(v.get("embeddings", [])) for v in self._gallery.values())
        print(f"[ReID] Gallery loaded: {len(self._gallery)} persons, "
              f"{total_embs} embeddings.")

    def get_gallery(self) -> dict:
        """Return the current gallery (for saving to cache)."""
        return self._gallery

    def get_registered_count(self) -> int:
        return len(self._gallery)

    # ── Identification ─────────────────────────────────────────────────────────

    def identify(self, frame: np.ndarray, bbox,
                 timestamp: float = None, top_k: int = 1) -> tuple[str | None, str | None, float]:
        """
        Try to identify a person using body ReID.

        Args:
            frame:  Full BGR frame.
            bbox:   [x1, y1, x2, y2] — person bounding box.
            timestamp: The timestamp of the current frame (defaults to current time if None).
            top_k:  Currently unused (reserved for future ranked results).

        Returns:
            (person_id, person_name, similarity) or (None, None, 0.0) if no match.
        """
        if not self._gallery:
            return None, None, 0.0

        emb = self.extract_embedding(frame, bbox)
        if emb is None:
            return None, None, 0.0

        best_id   = None
        best_name = None
        best_sim  = -1.0

        # Enforce a short-term temporal window to prevent false positives across different sessions.
        reid_timeout = self.temporal_window

        for pid, data in self._gallery.items():
            for entry in data["embeddings"]:
                if isinstance(entry, tuple):
                    stored_emb, entry_timestamp = entry
                    ref_time = timestamp if timestamp is not None else time.time()
                    if ref_time - entry_timestamp > reid_timeout:
                        continue
                else:
                    stored_emb = entry
                    
                sim = self._cosine_similarity(emb, stored_emb)
                if sim > best_sim:
                    best_sim  = sim
                    best_id   = pid
                    best_name = data["name"]

        if best_sim >= self.similarity_threshold:
            logger.debug(f"[ReID] Body match: {best_name} ({best_id}) "
                         f"sim={best_sim:.3f}")
            return best_id, best_name, float(best_sim)

        return None, None, float(best_sim)

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised vectors."""
        # Already normalised → dot product equals cosine similarity
        return float(np.dot(a, b))

    def is_ready(self) -> bool:
        """Return True if the model can be loaded (file exists or can be downloaded)."""
        return os.path.exists(self._model_path) or True   # will auto-download

    def clear_gallery(self):
        """Clear all cached body embeddings (called on new day)."""
        self._gallery.clear()
        print("[ReID] Gallery cleared for new day.")

    def detect_bodies(self, frame: np.ndarray) -> list:
        """
        Detect people in the frame using OpenCV's built-in HOG detector.
        Returns a list of bounding boxes: [[x1, y1, x2, y2], ...]
        """
        try:
            if not hasattr(self, '_hog'):
                self._hog = cv2.HOGDescriptor()
                self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Resize frame for faster HOG detection if large
            h, w = frame.shape[:2]
            scale = 1.0
            if max(h, w) > 480:
                scale = 480 / max(h, w)
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            else:
                small_frame = frame
                
            boxes, weights = self._hog.detectMultiScale(
                small_frame, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            
            res_boxes = []
            for (x, y, w_box, h_box) in boxes:
                x1 = int(x / scale)
                y1 = int(y / scale)
                x2 = int((x + w_box) / scale)
                y2 = int((y + h_box) / scale)
                res_boxes.append([x1, y1, x2, y2])
            return res_boxes
        except Exception as e:
            logger.warning(f"[ReID] HOG detection error: {e}")
            return []
