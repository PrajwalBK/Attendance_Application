import logging
import os
import requests
import pickle
import base64
import json
import numpy as np
from datetime import datetime, date

# Setup Logging
from config.config import BASE_DIR, get_config

base_dir = BASE_DIR

log_file = os.path.join(base_dir, 'logs', 'db_debug.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


class APIClient:
    def __init__(self, api_base_url, api_user=None, api_password=None, token=None):
        self.base_url = api_base_url.rstrip('/')
        self.connected = True
        self.token = token
        self.config = get_config()
        self.username = api_user if api_user else self.config.get('api_username')
        self.password = api_password if api_password else self.config.get('api_password')
        
        logging.info(f"API Client initialized with URL: {self.base_url}")
        print(f"API Client initialized with URL: {self.base_url}")
        
        if not self.token:
            self.login()

    def login(self):
        """Authenticate with the API"""
        if not self.username or not self.password:
            logging.warning("API Login Skipped: No credentials provided.")
            print("WARNING: API Login Skipped - No credentials configured.")
            return False

        try:
            payload = {
                "email": self.username,
                "password": self.password
            }
            response = requests.post(f"{self.base_url}/api/auth/login/", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get('access_token')
                logging.info(f"API Login Success for user: {self.username}")
                print(f"[+] API Login Success for: {self.username}")
                return True, self.token
            elif response.status_code == 403:
                # [ROBUST] Handle concurrent login peaks (Login Storm)
                logging.warning(f"API Login 403 Forbidden. Retrying in 1s...")
                import time
                time.sleep(1)
                # Second attempt with more explicit headers
                headers = self._get_headers()
                response = requests.post(f"{self.base_url}/api/auth/login/", json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get('access_token')
                    return True, self.token
                
            msg = f"HTTP {response.status_code}: {response.text}"
            logging.error(f"API Login Failed: {msg}")
            try:
                msg_json = response.json() if response.status_code in [401, 403] else {}
                detail = msg_json.get('detail', response.text)
            except:
                detail = response.text
            print(f"[-] API Login Failed for {self.username}: {detail}")
            return False, msg
        except Exception as e:
            msg = str(e)
            logging.error(f"API Login Exception: {msg}")
            print(f"[x] API Login Exception: {msg}")
            return False, msg

    def _get_headers(self):
        """Return headers with Authorization token if available"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'VisionAttendance/1.0'
        }
        if self.token:
            headers['Authorization'] = f"Bearer {self.token}"
        return headers

    def _api_request(self, method, endpoint, retry_on_401=True, **kwargs):
        """Centralized request handler with auto-token refresh on 401"""
        url = f"{self.base_url}{endpoint}"
        if 'headers' not in kwargs:
            kwargs['headers'] = self._get_headers()
        
        try:
            response = requests.request(method, url, **kwargs)
            
            if response.status_code == 401 and retry_on_401:
                logging.info(f"401 Unauthorized for {endpoint}. Attempting token refresh...")
                print(f"[DEBUG] 401 Unauthorized. Refreshing token...")
                success, _ = self.login()
                if success:
                    # Update headers with new token and retry
                    kwargs['headers'] = self._get_headers()
                    return requests.request(method, url, **kwargs)
            
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"API Connection Error ({method} {endpoint}): {e}")
            raise e
        except Exception as e:
            logging.error(f"API Request Exception ({method} {endpoint}): {e}")
            raise e

    def get_statistics(self):
        """Fetch system statistics from API"""
        try:
            response = self._api_request('GET', f"/api/attendance/summary?date={date.today().isoformat()}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logging.info(f"get_statistics success: {data}")
                return {
                    'total_persons': data.get('present', {}).get('total', 0) + data.get('not_present', {}).get('total', 0), # Approx
                    'present_today': data.get('present', {}).get('total', 0)
                }
            logging.error(f"API Error (get_statistics): {response.status_code} - {response.text}")
            print(f"API Error (get_statistics): {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"API Connection Error: {e}")
            print(f"API Connection Error: {e}")
        return {'total_persons': 0, 'present_today': 0}

    def add_person(self, person_id, name, face_encoding, email=None, department=None, designation=None, shift_start="09:00", shift_end="18:00", mask_face_encoding=None):
        """Register a new person via API"""
        try:
            # Pickle and base64 encode face_encoding
            if face_encoding is not None:
                pickled_data = pickle.dumps(face_encoding)
                face_encoding_str = base64.b64encode(pickled_data).decode('utf-8')
            else:
                face_encoding_str = None

            payload = {
                "person_id": person_id,
                "name": name,
                "email": email,
                "department": department,
                "designation": designation,
                "shift_start": shift_start,
                "shift_end": shift_end,
                "face_encoding": face_encoding_str
            }
            
            response = self._api_request('POST', "/api/persons/", json=payload, timeout=10)
            if response.status_code == 200:
                logging.info(f"add_person success: {person_id}")
                return True, "Person added successfully"
            elif response.status_code == 400:
                 logging.warning(f"add_person failed (400): {response.json()}")
                 return False, f"Error: {response.json().get('detail', 'Unknown error')}"
            else:
                logging.error(f"add_person API Error: {response.status_code}")
                return False, f"API Error: {response.status_code}"
                
        except Exception as e:
            logging.error(f"add_person Exception: {e}")
            return False, str(e)

    def get_today_attendance(self):
        """Get today's attendance records"""
        try:
            today = date.today().isoformat()
            response = self._api_request('GET', f"/api/attendance/?date={today}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logging.info(f"get_today_attendance success: {len(data)} records")
                records = []
                for item in data:
                    # Map to format expected by CLI: (id, name, arrival, leaving, status)
                    person = item.get('person', {})
                    records.append((
                        item['person_id'],
                        person.get('name', 'Unknown'),
                        item['arrival_time'],
                        item['leaving_time'],
                        item['status']
                    ))
                return records
            logging.error(f"API Error (get_today_attendance): {response.status_code}")
            print(f"API Error (get_today_attendance): {response.status_code}")
            return []
        except Exception as e:
            logging.error(f"get_today_attendance Exception: {e}")
            print(f"API Error: {e}")
            return []

    def log_raw_detection(self, person_id, person_name, timestamp=None, snapshot_path=None, event_type='in'):
        """
        Log raw detection via API.
        """
        try:
             if timestamp is None:
                timestamp = datetime.now()
                
             date_str = timestamp.strftime('%Y-%m-%d')
             time_str = timestamp.strftime('%H:%M:%S')
             
             payload = {
                 "person_id": person_id,
                 "name": person_name,
                 "date": date_str,
                 "time": time_str,
                 "snapshot_path": snapshot_path,
                 "event_type": event_type.lower() if event_type else 'in'
             }
             
             self._api_request('POST', "/api/attendance/logs", json=payload, timeout=2) # Short timeout
             
        except Exception as e:
            # Don't crash main loop for logging
            logging.error(f"log_raw_detection Error: {e}")

    def sync_daily_attendance(self, person_id, timestamp=None, snapshot_path=None, event_type='in'):
        """Alias for mark_attendance to match DatabaseManager interface"""
        return self.mark_attendance(person_id, timestamp, snapshot_path, event_type)

    def mark_attendance(self, person_id, timestamp=None, snapshot_path=None, event_type='in'):
        """Mark attendance for a person"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            payload = {
                "person_id": person_id,
                "event_type": event_type.lower() if event_type else 'in',
                "timestamp": timestamp_str
            }
            
            print(f"[DEBUG] APIClient: Sending POST to custom /api/attendance/mark_attendance/ for {person_id} | event={event_type}")
            resp = self._api_request('POST', "/api/attendance/mark_attendance/", json=payload, timeout=10)
            
            if resp.status_code == 200:
                logging.info(f"mark_attendance: Successfully marked for {person_id} | event={event_type}")
                print(f"[DEBUG] APIClient: mark_attendance Success for {person_id} | event={event_type}")
                return True, "Attendance Marked"
            else:
                logging.error(f"mark_attendance API Error: {resp.status_code} - {resp.text}")
                print(f"[DEBUG] APIClient: Error {resp.status_code} - {resp.text}")
                return False, f"API Error: {resp.text}"

        except Exception as e:
            logging.error(f"mark_attendance Exception: {e}")
            return False, str(e)

    def auto_mark_away(self, away_threshold_minutes=5):
        """
        Auto-Away Detection: sends request to API to mark Present employees as Away 
        if they haven't been detected in the last `away_threshold_minutes` minutes.
        Returns a list of (person_id, name) who were auto-marked as Away.
        """
        try:
            payload = {"threshold_minutes": away_threshold_minutes}
            response = self._api_request('POST', "/api/attendance/auto_away", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                marked = data.get("marked_away", [])
                
                # The GUI expects a list of tuples (person_id, name)
                result = []
                for item in marked:
                    result.append((item.get("person_id"), item.get("name", "Unknown")))
                    print(f"[AUTO-AWAY] {item.get('name')} ({item.get('person_id')}) → Away via API.")
                return result
            else:
                logging.warning(f"auto_mark_away API Error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logging.error(f"auto_mark_away Exception: {e}")
            return []

    def get_all_face_encodings(self):
        """Get all face encodings for recognition"""
        try:
            response = self._api_request('GET', "/api/persons/encodings", timeout=10)
            if response.status_code == 200:
                persons = response.json()
                print(f"[SYNC] Received {len(persons)} raw personnel records from server.")
                
                encodings = {}
                for p in persons:
                    pid = p.get('person_id')
                    name = p.get('name', 'Unknown')
                    encoded_data = p.get('face_encoding')
                    
                    if encoded_data:
                        try:
                            # Format A: JSON Array (starts with '[')
                            if encoded_data.strip().startswith('['):
                                try:
                                    import json
                                    face_data = np.array(json.loads(encoded_data))
                                    encodings[pid] = {
                                        'name': name,
                                        'encoding': face_data
                                    }
                                    continue
                                except Exception as json_err:
                                    print(f"Error parsing JSON face for {name} ({pid}): {json_err}")
                            
                            # Format B: Base64 -> Pickle
                            missing_padding = len(encoded_data) % 4
                            if missing_padding:
                                encoded_data += '=' * (4 - missing_padding)
                            
                            try:
                                decoded_bytes = base64.b64decode(encoded_data)
                                face_data = pickle.loads(decoded_bytes)
                                encodings[pid] = {
                                    'name': name,
                                    'encoding': face_data
                                }
                            except Exception as decode_err:
                                first_chars = encoded_data[:15] + "..." if len(encoded_data) > 15 else encoded_data
                                print(f"Error decoding face for {name} ({pid}): {decode_err}")
                        except Exception as e:
                            print(f"Error decoding face for {name} ({pid}): {e}")
                return encodings
            else:
                 print(f"[DEBUG] Fetch Failed: {response.text}")
            return {}
        except Exception as e:
            print(f"API Error fetching encodings: {e}")
            return {}

    def log_unknown_person(self, snapshot_path, face_encoding):
        """Log unknown person detection"""
        pass

    def export_to_csv(self, filename):
        """Export data via API"""
        import csv
        records = self.get_today_attendance()
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Name', 'Login Time', 'Last Seen (Logout)', 'Status'])
                writer.writerows(records)
            return True, f"Exported to {filename}"
        except Exception as e:
            return False, str(e)

    def get_all_persons_details(self):
        """Fetch all person details for management"""
        try:
            response = self._api_request('GET', "/api/persons/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = []
                for p in data:
                     result.append((
                          p['person_id'],
                          p['name'],
                          p.get('email'),
                          p.get('department'),
                          p.get('designation'),
                          p.get('shift_start'),
                          p.get('shift_end'),
                          p.get('face_encoding') 
                     ))
                return result
            else:
                logging.error(f"API Error (get_persons): {response.status_code}")
                return []
        except Exception as e:
             logging.error(f"get_persons Exception: {e}")
             return []

    def get_person(self, person_id):
        """Retrieve a person's details by ID"""
        persons = self.get_all_persons_details()
        for p in persons:
            if p[0] == person_id:
                return p
        return None

    def delete_person(self, person_id):
        """Delete a person via API"""
        try:
            response = self._api_request('DELETE', f"/api/persons/{person_id}", timeout=10)
            if response.status_code == 200:
                return True, "Deleted Successfully"
            else:
                return False, f"API Error: {response.status_code}"
        except Exception as e:
            return False, str(e)

    def get_recent_logs(self, limit=100):
        """Fetch recent raw detection logs via API"""
        try:
            response = self._api_request('GET', f"/api/attendance/logs?limit={limit}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logging.info(f"get_recent_logs success: {len(data)} records")
                return [(item.get('person_id'), item.get('name'), item.get('date'), item.get('time'), item.get('event_type')) for item in data]
            logging.error(f"API Error (get_recent_logs): {response.status_code}")
            return []
        except Exception as e:
            logging.error(f"get_recent_logs Exception: {e}")
            return []

    def is_video_processed(self, filename):
        """Check if video is processed (Client-side stub or API call)"""
        return False

    def mark_video_as_processed(self, filename):
        """Mark video as processed (Client-side stub)"""
        pass
