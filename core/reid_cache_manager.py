"""
reid_cache_manager.py
=====================
Daily local cache for Person Re-Identification body embeddings.

Design decisions:
    - Each day's embeddings live in:  data/reid_cache/YYYY-MM-DD.pkl
    - On startup the manager:
        1. Loads today's cache (if it exists).
        2. Deletes ALL files older than today automatically.
    - No database, no backend API, no configuration required.
    - The cache is a plain Python dict: {person_id: {"name": str, "embeddings": [...]}}
"""

import os
import pickle
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)


class ReIDCacheManager:
    """
    Manages a daily rolling cache of body embeddings for Person ReID.

    Usage:
        cache = ReIDCacheManager("data/reid_cache")
        gallery = cache.load_today()          # Load this day's cache
        reid.load_gallery(gallery)            # Push into PersonReID

        # Later, when a new embedding is captured:
        cache.save(reid.get_gallery())        # Persist to disk
    """

    def __init__(self, cache_dir: str):
        self.cache_dir  = cache_dir
        self._today_str = date.today().isoformat()   # "2026-06-17"
        self._cache_path = os.path.join(cache_dir, f"{self._today_str}.pkl")

        os.makedirs(cache_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_today(self) -> dict:
        """
        Load today's gallery from disk. Also cleans up yesterday's files.

        Returns:
            dict: {person_id: {"name": str, "embeddings": [np.array, ...]}}
                  Empty dict if no cache exists for today.
        """
        self._cleanup_old_files()

        if not os.path.exists(self._cache_path):
            print(f"[ReID Cache] No cache for today ({self._today_str}). "
                  f"Starting fresh.")
            return {}

        try:
            with open(self._cache_path, "rb") as f:
                gallery = pickle.load(f)

            total_embs = sum(len(v.get("embeddings", [])) for v in gallery.values())
            print(f"[ReID Cache] Loaded today's cache: "
                  f"{len(gallery)} persons, {total_embs} embeddings.")
            return gallery

        except Exception as e:
            logger.error(f"[ReID Cache] Failed to load cache: {e}")
            print(f"[ReID Cache] Cache load failed ({e}). Starting fresh.")
            return {}

    def save(self, gallery: dict) -> dict:
        """
        Persist the current gallery to today's cache file, merging with any existing disk data.
        Called periodically (e.g., every 60 seconds) or on shutdown.

        Args:
            gallery: {person_id: {"name": str, "embeddings": [...]}}
        Returns:
            dict: The merged gallery.
        """
        try:
            import numpy as np
            # 1. Load existing from disk if it exists
            disk_gallery = {}
            if os.path.exists(self._cache_path):
                try:
                    with open(self._cache_path, "rb") as f:
                        disk_gallery = pickle.load(f)
                except Exception:
                    pass
            
            # Sanitize disk gallery: convert legacy format to tuples
            for pid, data in disk_gallery.items():
                embs = data.get("embeddings", [])
                sanitized_embs = []
                for entry in embs:
                    if isinstance(entry, tuple):
                        sanitized_embs.append(entry)
                    elif isinstance(entry, np.ndarray):
                        sanitized_embs.append((entry, 0.0))
                data["embeddings"] = sanitized_embs
            
            # Helper lambda to extract raw embedding array
            get_vector = lambda x: x[0] if isinstance(x, tuple) else x
            
            # 2. Merge local gallery into disk gallery
            for pid, data in gallery.items():
                if pid not in disk_gallery:
                    disk_gallery[pid] = {"name": data["name"], "embeddings": []}
                
                # Merge embeddings
                existing_embs = disk_gallery[pid]["embeddings"]
                for local_emb in data["embeddings"]:
                    # Check if already present (cosine similarity > 0.95)
                    dup = False
                    local_vec = get_vector(local_emb)
                    for stored_emb in existing_embs:
                        stored_vec = get_vector(stored_emb)
                        if float(np.dot(local_vec, stored_vec)) > 0.95:
                            dup = True
                            break
                    if not dup and len(existing_embs) < 8: # Keep max 8 embeddings per person
                        if isinstance(local_emb, tuple):
                            existing_embs.append(local_emb)
                        else:
                            existing_embs.append((local_emb, 0.0))

            # 3. Write back to disk
            with open(self._cache_path, "wb") as f:
                pickle.dump(disk_gallery, f, protocol=pickle.HIGHEST_PROTOCOL)

            total_embs = sum(len(v.get("embeddings", [])) for v in disk_gallery.values())
            logger.debug(f"[ReID Cache] Merged and Saved: {len(disk_gallery)} persons, "
                         f"{total_embs} embeddings → {self._cache_path}")
            return disk_gallery
        except Exception as e:
            logger.error(f"[ReID Cache] Save/Merge failed: {e}")
            return gallery

    def get_stats(self) -> dict:
        """Return cache statistics for display in UI."""
        if not os.path.exists(self._cache_path):
            return {"date": self._today_str, "persons": 0, "embeddings": 0,
                    "size_kb": 0}
        try:
            with open(self._cache_path, "rb") as f:
                gallery = pickle.load(f)
            total_embs = sum(len(v.get("embeddings", [])) for v in gallery.values())
            size_kb = os.path.getsize(self._cache_path) // 1024
            return {
                "date":       self._today_str,
                "persons":    len(gallery),
                "embeddings": total_embs,
                "size_kb":    size_kb,
            }
        except Exception:
            return {"date": self._today_str, "persons": 0, "embeddings": 0,
                    "size_kb": 0}

    # ── Housekeeping ───────────────────────────────────────────────────────────

    def _cleanup_old_files(self):
        """Delete all .pkl files in the cache dir that are NOT from today."""
        deleted = 0
        try:
            for fname in os.listdir(self.cache_dir):
                if not fname.endswith(".pkl"):
                    continue
                fpath = os.path.join(self.cache_dir, fname)
                file_date_str = fname.replace(".pkl", "")

                # Keep today's file, delete everything else
                if file_date_str != self._today_str:
                    try:
                        os.remove(fpath)
                        deleted += 1
                        logger.info(f"[ReID Cache] Deleted old cache: {fname}")
                    except Exception as e:
                        logger.warning(f"[ReID Cache] Could not delete {fname}: {e}")

            if deleted:
                print(f"[ReID Cache] Cleaned up {deleted} old cache file(s).")

        except Exception as e:
            logger.error(f"[ReID Cache] Cleanup error: {e}")
