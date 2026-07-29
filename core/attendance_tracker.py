import time
import threading
from core.voice_handler import VoiceSystem  

class AttendanceTracker:
    def __init__(self, db_manager, face_handler, event_type=None):
        self.db_manager = db_manager
        self.face_handler = face_handler
        self.event_type = event_type
        
        # Initialize Voice
        self.voice = VoiceSystem()
        
        # Timers
        self.last_log_time = {}        
        self.last_attendance_time = {} 
        self.last_unknown_alert_time = 0 
    
    def process_recognized_face(self, person_id, person_name, snapshot_path=None, event_type=None):
        """
        1. Log Face: 90s gap
        2. Update Attendance: 5s gap (with Voice Feedback)
        """
        if not person_id or person_id == 'UNKNOWN' or str(person_id).startswith('UNKNOWN'):
            return True, None

        current_ts = time.time()
        
        # Determine event type (Method arg overrides Instance var)
        # Default to 'in' if neither is specified, to ensure we trigger "Login" logic instead of "Update" logic
        final_event = event_type if event_type else (self.event_type if self.event_type else 'in')

        # --- PART 1: RAW LOGGING ---
        if person_id not in self.last_log_time or \
           (current_ts - self.last_log_time[person_id] > 10.0):
            
            self.db_manager.log_raw_detection(person_id, person_name, snapshot_path=snapshot_path, event_type=final_event)
            self.last_log_time[person_id] = current_ts

        # --- PART 2: ATTENDANCE LOGIC & VOICE ---
        if person_id not in self.last_attendance_time or \
           (current_ts - self.last_attendance_time[person_id] > 5.0):
            
            # Perform DB Sync
            # Use mark_attendance (unified interface for DB and API)
            try:
                msg = self.db_manager.mark_attendance(person_id, snapshot_path=snapshot_path, event_type=final_event)
            except AttributeError:
                # Fallback if mark_attendance doesn't exist on manager (it should though)
                msg = self.db_manager.sync_daily_attendance(person_id, snapshot_path=snapshot_path, event_type=final_event)
                
            self.last_attendance_time[person_id] = current_ts
            
            # --- VOICE FEEDBACK -------------------------------------------
            
            if "LOGIN" in msg:
             
                self.voice.speak(f"Welcome, {person_name}. Login Successful.")
                
            elif "LOGOUT UPDATE" in msg:
              
                self.voice.speak(f"Goodbye, {person_name}. Logout Updated.")
                
            elif "Shift Ongoing" in msg:
              
                # For now, we stay silent for ongoing shifts.
                pass
            
            return True, f"{person_name}: {msg}"

        return True, None

    def process_unknown_person(self, snapshot_path, face_encoding):
        """
        Log unknown person to DB and Alert Admin
        """
        # 1. Log to DB
        result = self.db_manager.log_unknown_person(snapshot_path, face_encoding)
        
        # 2. Alert Logic (Cooldown: 15s)
        current_ts = time.time()
        if current_ts - self.last_unknown_alert_time > 15.0:
            self.last_unknown_alert_time = current_ts
            
            # Run alert in background to avoid freezing video
            threading.Thread(target=self._trigger_unknown_alert, daemon=True).start()
            
        return result

    def _trigger_unknown_alert(self):
        """Play beep and speak warning"""
        try:
            # Cross-platform Beep
            try:
                import winsound
                # Beep: Frequency 1000Hz, Duration 500ms
                winsound.Beep(1000, 500) 
            except ImportError:
                # Linux/Mac Fallback
                print('\a')
            
            self.voice.speak("Unknown person detected.")
        except Exception as e:
            print(f"Alert Error: {e}")