import sqlite3
import os
import logging
from config.config import BASE_DIR

class OfflineStorage:
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(BASE_DIR, "data", "offline_cache.db")
        
        self.db_path = db_path
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self):
        # Use a generous timeout and enable WAL mode for safe multiprocess access
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        snapshot_path TEXT,
                        event_type TEXT
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS offline_raw_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        snapshot_path TEXT,
                        event_type TEXT
                    );
                """)
        except Exception as e:
            logging.error(f"OfflineStorage init error: {e}")
            print(f"OfflineStorage init error: {e}")
        finally:
            conn.close()

    def add_attendance(self, person_id, timestamp_str, snapshot_path, event_type):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO offline_attendance (person_id, timestamp, snapshot_path, event_type) VALUES (?, ?, ?, ?)",
                    (person_id, timestamp_str, snapshot_path, event_type)
                )
        finally:
            conn.close()

    def add_raw_log(self, person_id, name, timestamp_str, snapshot_path, event_type):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO offline_raw_logs (person_id, name, timestamp, snapshot_path, event_type) VALUES (?, ?, ?, ?, ?)",
                    (person_id, name, timestamp_str, snapshot_path, event_type)
                )
        finally:
            conn.close()

    def get_pending_attendance(self):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, person_id, timestamp, snapshot_path, event_type FROM offline_attendance ORDER BY id ASC")
            return cursor.fetchall()
        finally:
            conn.close()

    def delete_attendance(self, row_id):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM offline_attendance WHERE id = ?", (row_id,))
        finally:
            conn.close()

    def get_pending_raw_logs(self):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, person_id, name, timestamp, snapshot_path, event_type FROM offline_raw_logs ORDER BY id ASC")
            return cursor.fetchall()
        finally:
            conn.close()

    def delete_raw_log(self, row_id):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM offline_raw_logs WHERE id = ?", (row_id,))
        finally:
            conn.close()
