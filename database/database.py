import mysql.connector
from mysql.connector import errorcode
import pickle
import base64
from datetime import datetime, date, timedelta

class DatabaseManager:
    def __init__(self, custom_config=None):
        print("DEBUG: DatabaseManager.__init__ called")
        if custom_config:
            self.config = custom_config
        else:
            # Import here to avoid circular dependency if needed
            from config.db_config_manager import DBConfigManager
            self.config = DBConfigManager.load_config()
            
        self.connected = False
        try:
            self.init_database()
            self.connected = True
        except Exception as e:
            print(f"Database Initial Connection Failed: {e}")
            self.connected = False
    
    def get_connection(self):
        """Helper to get a fresh connection"""
        return mysql.connector.connect(**self.config)

    def init_database(self):
        """Initialize MySQL tables"""
        try:
            # 1. Create Database if not exists
            self.create_database_if_not_exists()

            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 1. Persons Table (With Shift Columns)
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS persons (
                        person_id VARCHAR(50) PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        email VARCHAR(100),
                        department VARCHAR(100),
                        designation VARCHAR(100),
                        shift_start VARCHAR(10) DEFAULT '09:00',
                        shift_end VARCHAR(10) DEFAULT '18:00',
                        registered_date VARCHAR(30) NOT NULL,
                        face_encoding LONGTEXT
                    )
                ''')
                print("Table 'persons' checked/created.")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("Table 'persons' already exists (Checked).")
                else:
                    print(f"Error creating 'persons' table: {err}")
            
            # Migration for designation
            try:
                cursor.execute("ALTER TABLE persons ADD COLUMN designation VARCHAR(100) AFTER department")
            except mysql.connector.Error as err:
                if err.errno == 1060: pass

            # Migration for mask_face_encoding
            try:
                cursor.execute("ALTER TABLE persons ADD COLUMN mask_face_encoding LONGTEXT AFTER face_encoding")
                print("Added 'mask_face_encoding' column to 'persons' (migration).")
            except mysql.connector.Error as err:
                if err.errno == 1060: pass

            # 2. Attendance Summary
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        person_id VARCHAR(50) NOT NULL,
                        date VARCHAR(20) NOT NULL,
                        arrival_time VARCHAR(20),
                        leaving_time VARCHAR(20),
                        status VARCHAR(20) DEFAULT 'Present',
                        total_break_seconds INT DEFAULT 0,
                        break_count INT DEFAULT 0,
                        FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
                    )
                ''')
                print("Table 'attendance' checked/created.")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("Table 'attendance' already exists (Checked).")
                else:
                    print(f"Error creating 'attendance' table: {err}")

            # 3. Raw Logs (With Name Column)
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        person_id VARCHAR(50),
                        name VARCHAR(100),
                        date VARCHAR(20),
                        time VARCHAR(20),
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        snapshot_path VARCHAR(255),
                        FOREIGN KEY (person_id) REFERENCES persons(person_id) ON DELETE CASCADE
                    )
                ''')
                print("Table 'face_logs' checked/created.")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("Table 'face_logs' already exists (Checked).")
                else:
                    print(f"Error creating 'face_logs' table: {err}")

            # 4. Unknown Faces Table
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS unknown_faces (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        snapshot_path VARCHAR(255),
                        face_encoding LONGTEXT
                    )
                ''')
                print("Table 'unknown_faces' checked/created.")
                print("Table 'unknown_faces' checked/created.")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("Table 'unknown_faces' already exists (Checked).")
                else:
                    print(f"Error creating 'unknown_faces' table: {err}")

            # 5. Processed Videos Table (Deduplication)
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processed_videos (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        filename VARCHAR(255) UNIQUE NOT NULL,
                        processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                print("Table 'processed_videos' checked/created.")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("Table 'processed_videos' already exists (Checked).")
                else:
                    print(f"Error creating 'processed_videos' table: {err}")
            
            # --- MIGRATIONS ---
            # Ensure snapshot_path exists in face_logs (for existing DBs)
            try:
                cursor.execute("DESCRIBE face_logs")
                columns = [column[0] for column in cursor.fetchall()]
                if 'snapshot_path' not in columns:
                    cursor.execute("ALTER TABLE face_logs ADD COLUMN snapshot_path VARCHAR(255)")
                    print("Added 'snapshot_path' column to 'face_logs'.")
            except Exception as e:
                print(f"Migration Error (snapshot_path): {e}")

            # Ensure total_break_seconds exists in attendance
            try:
                cursor.execute("DESCRIBE attendance")
                columns = [column[0] for column in cursor.fetchall()]
                if 'total_break_seconds' not in columns:
                    cursor.execute("ALTER TABLE attendance ADD COLUMN total_break_seconds INT DEFAULT 0")
                    print("Added 'total_break_seconds' column to 'attendance'.")
            except Exception as e:
                print(f"Migration Error (total_break_seconds): {e}")

            # Ensure break_count exists in attendance
            try:
                cursor.execute("DESCRIBE attendance")
                columns = [column[0] for column in cursor.fetchall()]
                if 'break_count' not in columns:
                    cursor.execute("ALTER TABLE attendance ADD COLUMN break_count INT DEFAULT 0")
                    print("Added 'break_count' column to 'attendance'.")
            except Exception as e:
                print(f"Migration Error (break_count): {e}")

            # Ensure event_type exists in face_logs
            try:
                cursor.execute("DESCRIBE face_logs")
                columns = [column[0] for column in cursor.fetchall()]
                if 'event_type' not in columns:
                    cursor.execute("ALTER TABLE face_logs ADD COLUMN event_type VARCHAR(20)")
                    print("Added 'event_type' column to 'face_logs'.")
            except Exception as e:
                print(f"Migration Error (event_type): {e}")
            

            # 6. Users Table (Authentication)
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        password VARCHAR(255) NOT NULL,
                        full_name VARCHAR(100),
                        role VARCHAR(20) DEFAULT 'admin',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                print("Table 'users' checked/created.")
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("Table 'users' already exists (Checked).")
                else:
                    print(f"Error creating 'users' table: {err}")

            conn.commit()
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
            raise err

    def create_database_if_not_exists(self):
        """Creates the database if it doesn't exist"""
        db_name = self.config.get('database')
        if not db_name: return

        # Connect without database
        temp_config = self.config.copy()
        if 'database' in temp_config:
            del temp_config['database']
        
        try:
            conn = mysql.connector.connect(**temp_config)
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            conn.commit()
            cursor.close()
            conn.close()
            print(f"Database '{db_name}' checked/created successfully.")
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_DB_CREATE_EXISTS:
                print(f"Database '{db_name}' already exists (Checked).")
            else:
                print(f"Error creating database: {err}")
                raise err

    # --- CORE LOGGING & ATTENDANCE ---

    def log_raw_detection(self, person_id, person_name, timestamp=None, snapshot_path=None, event_type=None):
        """Logs detection with Name, Date and Time"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if timestamp is None:
                now = datetime.now()
            else:
                now = timestamp
                
            current_date = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')
            
            cursor.execute('''
                INSERT INTO face_logs (person_id, name, date, time, timestamp, snapshot_path, event_type) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (person_id, person_name, current_date, current_time, now, snapshot_path, event_type))
            
            conn.commit()
        except Exception as e:
            print(f"Log Error: {e}")
        finally:
            conn.close()

    def mark_attendance(self, person_id, timestamp=None, snapshot_path=None, event_type=None):
        """
        Marks attendance for a person.
        Wrapper for sync_daily_attendance to match calling convention.
        """
        return self.sync_daily_attendance(person_id, timestamp, snapshot_path, event_type)

    def sync_daily_attendance(self, person_id, timestamp=None, snapshot_path=None, event_type=None):
        """
        Updates the daily attendance summary.
        Supports multiple sessions (rows) per day.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 1. Get Name
            cursor.execute("SELECT name FROM persons WHERE person_id = %s", (person_id,))
            result = cursor.fetchone()
            if not result: return False, "Person not found"
            name = result[0]
            
            # 2. Log Raw Event (Every detection)
            self.log_raw_detection(person_id, name, timestamp, snapshot_path, event_type)
            
            # 3. Update Daily Summary
            if timestamp is None:
                now = datetime.now()
            else:
                now = timestamp
                
            today = now.strftime('%Y-%m-%d')
            current_time = now.strftime('%H:%M:%S')
            
            # Check for the LATEST entry for today
            cursor.execute('''
                SELECT id, arrival_time, leaving_time, status, total_break_seconds, break_count 
                FROM attendance 
                WHERE person_id = %s AND date = %s 
                ORDER BY id DESC LIMIT 1
            ''', (person_id, today))
            record = cursor.fetchone()
            
            if record:
                rec_id, first_arrival, last_leaving, current_status, total_break, break_count = record
                
                if event_type == 'out':
                    # LOGOUT: Update existing open session
                    
                    # Minimum Session Duration check removed to allow immediate marking
                    pass

                    print(f"[DEBUG] DB: Found record {rec_id} (Status: {current_status}). Event: OUT. Updating...")
                    cursor.execute('''
                        UPDATE attendance SET leaving_time = %s, status = 'Away'
                        WHERE id = %s
                    ''', (current_time, rec_id))
                    msg_type = "LOGOUT/BREAK"
                    
                elif event_type == 'in':
                    # LOGIN
                    if current_status == 'Away':
                        # New Session - Calculate Break Duration
                        break_duration = 0
                        if last_leaving:
                            try:
                                fmt = '%H:%M:%S'
                                # Handle potentially different checking of formats if needed, but usually consistent
                                t_out = datetime.strptime(str(last_leaving), fmt)
                                t_in = datetime.strptime(current_time, fmt)
                                dt = (t_in - t_out).total_seconds()
                                if dt > 0:
                                    break_duration = int(dt)
                            except Exception as e:
                                print(f"Error calc break: {e}")
                        
                        # Increment break count from previous session? or just set to 1 for this session?
                        # Better to keep a running total or just let the aggregation handle it.
                        # We store the break leading UP TO this session IN this session row.
                        
                        print(f"[DEBUG] DB: Found record {rec_id} (Status: {current_status}). Event: IN. New Session. Break: {break_duration}s")
                        cursor.execute('''
                            INSERT INTO attendance (person_id, date, arrival_time, leaving_time, status, total_break_seconds, break_count)
                            VALUES (%s, %s, %s, %s, 'Present', %s, 1)
                        ''', (person_id, today, current_time, current_time, break_duration))
                        msg_type = "LOGIN (NEW SESSION)"
                    else:
                        # Already Present -> Update Last Seen
                        print(f"[DEBUG] DB: Found record {rec_id} (Status: {current_status}). Event: IN. Already Present. Updating Last Seen.")
                        cursor.execute('''
                            UPDATE attendance SET leaving_time = %s 
                            WHERE id = %s
                        ''', (current_time, rec_id))
                        msg_type = "ALREADY PRESENT (Updated Last Seen)"
                else:
                    # Generic Event -> Update Last Seen
                    cursor.execute('''
                        UPDATE attendance SET leaving_time = %s 
                        WHERE id = %s
                    ''', (current_time, rec_id))
                    msg_type = "ALREADY PRESENT (Updated Last Seen)"

            else:
                # FIRST RECORD OF THE DAY
                cursor.execute("SELECT shift_start FROM persons WHERE person_id = %s", (person_id,))
                shift_result = cursor.fetchone()
                shift_start = shift_result[0] if shift_result else "09:00"
                
                # Compare arrival time with shift start (30 min grace period)
                try:
                    arrival_dt = datetime.strptime(current_time, '%H:%M:%S')
                    # Fix: Check length or count to distinguish format
                    shift_dt = datetime.strptime(shift_start, '%H:%M:%S') if shift_start.count(':') == 2 else datetime.strptime(shift_start, '%H:%M')
                    
                    grace_period = timedelta(minutes=30)
                    
                    if event_type == 'out':
                        status = 'Away'
                    else:
                        status = 'Late' if arrival_dt > (shift_dt + grace_period) else 'Present'
                except:
                    status = 'Present'
                
                cursor.execute('''
                    INSERT INTO attendance (person_id, date, arrival_time, leaving_time, status, total_break_seconds)
                    VALUES (%s, %s, %s, %s, %s, 0)
                ''', (person_id, today, current_time, current_time, status))
                msg_type = "LOGIN" if event_type != 'out' else "FIRST DETECTION (OUT)"
                
            conn.commit()
            return True, f"Success: {msg_type}"
            
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def auto_mark_away(self, away_threshold_minutes=5):
        """
        Auto-Away Detection: marks Present employees as Away if they haven't
        been detected by any camera in the last `away_threshold_minutes` minutes.
        Called periodically from the GUI loop.
        Returns a list of (person_id, name) who were auto-marked as Away.
        """
        marked = []
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            today = datetime.now().strftime('%Y-%m-%d')
            now = datetime.now()

            # Get all Present/Late records for today
            cursor.execute('''
                SELECT a.id, a.person_id, p.name, a.arrival_time
                FROM attendance a
                JOIN persons p ON a.person_id = p.person_id
                WHERE a.date = %s AND a.status IN ('Present', 'Late')
            ''', (today,))
            active_records = cursor.fetchall()

            for rec_id, person_id, name, arrival_time in active_records:
                # Find the last time this person was seen in face_logs today
                cursor.execute('''
                    SELECT MAX(time) FROM face_logs
                    WHERE person_id = %s AND date = %s
                ''', (person_id, today))
                result = cursor.fetchone()
                last_seen_time = result[0] if result and result[0] else arrival_time

                # Parse last_seen into datetime for comparison
                try:
                    fmt = '%H:%M:%S' if str(last_seen_time).count(':') == 2 else '%H:%M'
                    last_seen_dt = datetime.combine(now.date(), 
                                   datetime.strptime(str(last_seen_time), fmt).time())
                    minutes_since = (now - last_seen_dt).total_seconds() / 60.0

                    if minutes_since >= away_threshold_minutes:
                        # Auto-mark as Away, using last-seen time as leaving_time
                        leaving_time = datetime.strptime(str(last_seen_time), fmt).strftime('%H:%M:%S')
                        cursor.execute('''
                            UPDATE attendance SET leaving_time = %s, status = 'Away'
                            WHERE id = %s
                        ''', (leaving_time, rec_id))
                        marked.append((person_id, name))
                        print(f"[AUTO-AWAY] {name} ({person_id}) → Away. Last seen: {last_seen_time} ({minutes_since:.1f} min ago)")
                except Exception as e:
                    print(f"[AUTO-AWAY] Parse error for {person_id}: {e}")

            if marked:
                conn.commit()
            conn.close()
        except Exception as e:
            print(f"[AUTO-AWAY] DB Error: {e}")
        return marked

    # --- VIDEO DEDUPLICATION ---

    def is_video_processed(self, filename):
        """Checks if a video file has already been processed"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id FROM processed_videos WHERE filename = %s", (filename,))
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            print(f"Error checking processed status: {e}")
            return False
        finally:
            conn.close()

    def mark_video_as_processed(self, filename):
        """Marks a video file as processed to prevent re-analysis"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO processed_videos (filename) VALUES (%s)", (filename,))
            conn.commit()
            return True
        except mysql.connector.Error as err:
            print(f"Error marking video as processed: {err}")
            return False
        finally:
            conn.close()

    def log_unknown_person(self, snapshot_path, face_encoding):
        """Logs unknown person with snapshot and encoding"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            pickled_data = pickle.dumps(face_encoding)
            safe_data_string = base64.b64encode(pickled_data).decode('utf-8')
            
            cursor.execute('''
                INSERT INTO unknown_faces (snapshot_path, face_encoding) 
                VALUES (%s, %s)
            ''', (snapshot_path, safe_data_string))
            
        except Exception as e:
            print(f"Log Unknown Error: {e}")
            return False
        finally:
            conn.close()

    def add_person(self, person_id, name, face_encoding, mask_face_encoding=None, email=None, department=None, designation=None, shift_start="09:00", shift_end="18:00"):
        """Add a new person to the database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            pickled_data = pickle.dumps(face_encoding)
            safe_data_string = base64.b64encode(pickled_data).decode('utf-8')
            
            safe_mask_data_string = None
            if mask_face_encoding is not None:
                pickled_mask_data = pickle.dumps(mask_face_encoding)
                safe_mask_data_string = base64.b64encode(pickled_mask_data).decode('utf-8')
            
            cursor.execute('''
                INSERT INTO persons (person_id, name, email, department, designation, shift_start, shift_end, registered_date, face_encoding, mask_face_encoding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (person_id, name, email, department, designation, shift_start, shift_end, datetime.now().isoformat(), safe_data_string, safe_mask_data_string))
            
            conn.commit()
            return True, "Person added successfully"
        except mysql.connector.IntegrityError:
            return False, "Person ID already exists"
        except Exception as e:
            return False, str(e)
        finally:
            if conn.is_connected(): conn.close()

    def update_person(self, person_id, name, email, dept, s_start, s_end, designation=None):
        """Update a person's details"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE persons 
                SET name=%s, email=%s, department=%s, designation=%s, shift_start=%s, shift_end=%s
                WHERE person_id=%s
            ''', (name, email, dept, designation, s_start, s_end, person_id))
            conn.commit()
            return True, "Update Successful"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    def get_all_face_encodings(self):
        """
        Retrieve all face encodings from the database.
        Returns: Dictionary {person_id: {'name': name, 'encoding': numpy_array, 'mask_encoding': numpy_array_or_None}}
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT person_id, name, face_encoding, mask_face_encoding FROM persons")
            rows = cursor.fetchall()
            
            encodings = {}
            for pid, name, encoded_data, encoded_mask_data in rows:
                face_data = None
                mask_data = None
                
                if encoded_data:
                    try:
                        # Decode based on format
                        # Format A: JSON Array (starts with '[')
                        if isinstance(encoded_data, str) and encoded_data.strip().startswith('['):
                            try:
                                import json
                                import numpy as np
                                face_data = np.array(json.loads(encoded_data))
                            except Exception as json_err:
                                print(f"Error parsing JSON face for {name} ({pid}): {json_err}")
                        else:
                            # Format B: Base64 -> Pickle (legacy/standard)
                            missing_padding = len(encoded_data) % 4
                            if missing_padding:
                                encoded_data += '=' * (4 - missing_padding)
                            try:
                                decoded_bytes = base64.b64decode(encoded_data)
                                face_data = pickle.loads(decoded_bytes)
                            except Exception as decode_err:
                                print(f"Error decoding face for {name} ({pid}): {decode_err}")
                    except Exception as e:
                        print(f"Error decoding face for {name} ({pid}): {e}")
                
                if encoded_mask_data:
                    try:
                        # Format A: JSON Array (starts with '[')
                        if isinstance(encoded_mask_data, str) and encoded_mask_data.strip().startswith('['):
                            try:
                                import json
                                import numpy as np
                                mask_data = np.array(json.loads(encoded_mask_data))
                            except Exception as json_err:
                                print(f"Error parsing JSON mask face for {name} ({pid}): {json_err}")
                        else:
                            # Format B: Base64 -> Pickle (legacy/standard)
                            missing_padding = len(encoded_mask_data) % 4
                            if missing_padding:
                                encoded_mask_data += '=' * (4 - missing_padding)
                            try:
                                decoded_bytes = base64.b64decode(encoded_mask_data)
                                mask_data = pickle.loads(decoded_bytes)
                            except Exception as decode_err:
                                print(f"Error decoding mask face for {name} ({pid}): {decode_err}")
                    except Exception as e:
                        print(f"Error decoding mask face for {name} ({pid}): {e}")
                
                if face_data is not None:
                    encodings[pid] = {
                        'name': name,
                        'encoding': face_data,
                        'mask_encoding': mask_data
                    }
            return encodings
        except Exception as e:
            print(f"DB Error fetching encodings: {e}")
            return {}
        finally:
            conn.close()

    def get_all_face_encodings_raw(self):
        """
        Retrieve raw face encodings (base64 strings) for API synchronization.
        Used by: routers/persons.py -> /persons/encodings
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT person_id, name, face_encoding FROM persons")
            # Returns list of tuples: (id, name, b64_string)
            return cursor.fetchall()
        except Exception as e:
            print(f"DB Error fetching raw encodings: {e}")
            return []
        finally:
            conn.close()

    def delete_person(self, person_id):
        """Delete a person and their logs"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM persons WHERE person_id=%s', (person_id,))
            conn.commit()
            return True, "Deleted Successfully"
        except Exception as e:
            return False, str(e)
        finally:
            conn.close()

    # --- STATS & REPORTS (This was missing!) ---

    def get_person_stats(self, person_id):
        """
        Calculate detailed stats: Late In, Early Out, Avg Hours
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Get Person Details (Shift info)
        cursor.execute('SELECT name, shift_start, shift_end FROM persons WHERE person_id = %s', (person_id,))
        person = cursor.fetchone()
        
        if not person:
            conn.close()
            return None, "Person not found"
            
        name, s_start_str, s_end_str = person
        
        # 2. Get All Attendance Records for this person
        cursor.execute('SELECT arrival_time, leaving_time FROM attendance WHERE person_id = %s', (person_id,))
        records = cursor.fetchall()
        conn.close()
        
        # 3. Calculate Stats
        total_days = len(records)
        late_count = 0
        early_out_count = 0
        total_seconds_worked = 0
        days_with_full_data = 0
        
        def parse_time(t_str, fmt='%H:%M:%S'):
            try: return datetime.strptime(t_str, fmt)
            except: return None
            
        try:
            shift_start_dt = datetime.strptime(s_start_str, '%H:%M')
            shift_end_dt = datetime.strptime(s_end_str, '%H:%M')
        except:
            return None, "Error parsing shift times in DB"

        for arrival, leaving in records:
            # Check Late
            if arrival:
                arr_dt = parse_time(arrival)
                if arr_dt and arr_dt.time() > shift_start_dt.time():
                    late_count += 1
            
            # Check Early Out & Avg Hours
            if leaving:
                leave_dt = parse_time(leaving)
                if leave_dt:
                    if leave_dt.time() < shift_end_dt.time():
                        early_out_count += 1
                    
                    if arrival:
                        arr_dt = parse_time(arrival)
                        if arr_dt:
                            duration = (leave_dt - arr_dt).total_seconds()
                            if duration > 0:
                                total_seconds_worked += duration
                                days_with_full_data += 1

        avg_hours = 0
        if days_with_full_data > 0:
            avg_hours = (total_seconds_worked / days_with_full_data) / 3600 

        return {
            'name': name,
            'id': person_id,
            'shift': f"{s_start_str} - {s_end_str}",
            'total_days': total_days,
            'late': late_count,
            'early': early_out_count,
            'avg_hours': round(avg_hours, 1)
        }, "Success"

    # --- DATA FETCHING ---

    def get_statistics(self):
        today = date.today().isoformat()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM persons')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM attendance WHERE date = %s', (today,))
        present = cursor.fetchone()[0]
        conn.close()
        return {'total_persons': total, 'present_today': present}

    def get_today_attendance(self):
        today = date.today().isoformat()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT a.person_id, p.name, a.arrival_time, a.leaving_time, a.status, a.total_break_seconds
            FROM attendance a JOIN persons p ON a.person_id = p.person_id
            WHERE a.date = %s ORDER BY a.arrival_time DESC
        ''', (today,))
        records = cursor.fetchall()
        conn.close()
        return records
    
    def get_recent_logs(self):
        """Fetch the detailed raw logs (Last 100 records)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT person_id, name, date, time, event_type 
            FROM face_logs 
            ORDER BY id DESC LIMIT 100
        ''')
        records = cursor.fetchall()
        conn.close()
        return records
    
    def get_all_persons_details(self):
        """Fetch all details for the Edit View"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT person_id, name, email, department, designation, shift_start, shift_end 
            FROM persons
        ''')
        records = cursor.fetchall()
        conn.close()
        return records

    def export_to_csv(self, filename):
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

    def get_attendance_report(self, start_date, end_date, person_id=None):
        """
        Fetch attendance records for a specific date range.
        Aggregates multiple sessions per day into a single row:
        - Arrival = First IN
        - Leaving = Last OUT
        - Status = Present if any record is Present
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Aggregation Query
        query = '''
            SELECT 
                a.date, 
                p.name, 
                a.person_id, 
                MIN(a.arrival_time) as first_in, 
                MAX(a.leaving_time) as last_out, 
                MAX(a.status) as status,
                p.shift_start,
                p.shift_end
            FROM attendance a 
            JOIN persons p ON a.person_id = p.person_id
            WHERE a.date BETWEEN %s AND %s
        '''
        params = [start_date, end_date]
        
        if person_id and person_id != "All":
            query += ' AND a.person_id = %s'
            params.append(person_id)
            
        query += ' GROUP BY a.date, a.person_id, p.name, p.shift_start, p.shift_end'
        query += ' ORDER BY a.date DESC, first_in DESC'
        
        cursor.execute(query, tuple(params))
        
        # Return all columns including shift info (8 columns)
        # This fixes "General Details" displaying defaults/garbage where it expects shift info
        records = cursor.fetchall()
            
        conn.close()
        return records

    def export_to_pdf(self, data, filename, title="Attendance Report"):
        """
        Generate a PDF report using ReportLab.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            doc = SimpleDocTemplate(filename, pagesize=letter)
            elements = []
            
            styles = getSampleStyleSheet()
            elements.append(Paragraph(title, styles['Title']))
            elements.append(Spacer(1, 12))
            
            # Table Header
            table_data = [['Date', 'Name', 'ID', 'Arrival', 'Leaving', 'Status']]
            
            # Table Data
            for row in data:
                # Ensure all items are strings
                table_data.append([str(item) if item is not None else "" for item in row])
                
            # Create Table
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(t)
            doc.build(elements)
            return True, f"PDF Exported: {filename}"
        except ImportError:
            return False, "ReportLab not installed. Cannot generate PDF."
        except Exception as e:
            return False, str(e)

    def get_attendance_summary(self, date_str):
        """
        Get daily attendance summary stats (Present, Late, Absent, etc.)
        """
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            # 1. Total Persons
            cursor.execute("SELECT COUNT(*) as total FROM persons")
            total_persons = cursor.fetchone()['total']
            
            # 2. Daily Attendance Records with Shift Info
            cursor.execute('''
                SELECT a.arrival_time, p.shift_start
                FROM attendance a
                JOIN persons p ON a.person_id = p.person_id
                WHERE a.date = %s
            ''', (date_str,))
            records = cursor.fetchall()
            
            present_count = len(records)
            on_time = 0
            late = 0
            early_clock_in = 0
            
            for r in records:
                if r['arrival_time'] and r['shift_start']:
                    # Simple string compare works for HH:MM in 24h format
                    if r['arrival_time'] <= r['shift_start']:
                        on_time += 1
                        if r['arrival_time'] < r['shift_start']:
                            early_clock_in += 1
                    else:
                        late += 1
            
            not_present_count = total_persons - present_count
            
            return {
                "present": {
                    "total": present_count,
                    "on_time": on_time,
                    "late": late,
                    "early_clock_in": early_clock_in
                },
                "not_present": {
                    "total": not_present_count,
                    "absent": not_present_count,
                    "no_clock_in": 0,
                    "no_clock_out": 0 
                },
                "away": {
                    "total": 0,
                    "day_off": 0,
                    "time_off": 0
                }
            }
        except Exception as e:
            print(f"Summary Error: {e}")
            return {}
        finally:
            conn.close()