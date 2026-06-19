import os
import shutil
from datetime import datetime, date, timedelta
import csv
import json

class Utils:
    """Utility functions for the attendance system"""
    
    @staticmethod
    def create_directories():
        """Create necessary directories if they don't exist"""
        directories = ['exports', 'backups', 'logs']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    @staticmethod
    def backup_database(db_path, backup_folder='backups'):
        """Create a backup of the database"""
        if not os.path.exists(db_path):
            return False, "Database file not found"
        
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(backup_folder, f'attendance_backup_{timestamp}.db')
        
        try:
            shutil.copy2(db_path, backup_file)
            return True, f"Backup created: {backup_file}"
        except Exception as e:
            return False, f"Backup failed: {str(e)}"
    
    @staticmethod
    def format_date(date_str):
        """Format date string to readable format"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%B %d, %Y')
        except:
            return date_str
    
    @staticmethod
    def format_time(time_str):
        """Format time string to readable format"""
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            return time_obj.strftime('%I:%M %p')
        except:
            return time_str
    
    @staticmethod
    def calculate_duration(arrival_time, leaving_time):
        """Calculate duration between arrival and leaving"""
        if not arrival_time or not leaving_time:
            return "N/A"
        
        try:
            arrival = datetime.strptime(arrival_time, '%H:%M:%S')
            leaving = datetime.strptime(leaving_time, '%H:%M:%S')
            duration = leaving - arrival
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            return f"{hours}h {minutes}m"
        except:
            return "Error"
    
    @staticmethod
    def get_date_range(days_back=7):
        """Get date range for the last N days"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        return start_date.isoformat(), end_date.isoformat()
    
    @staticmethod
    def validate_person_id(person_id):
        """Validate person ID format"""
        if not person_id:
            return False, "Person ID cannot be empty"
        
        if len(person_id) < 3:
            return False, "Person ID must be at least 3 characters"
        
        if len(person_id) > 20:
            return False, "Person ID must be less than 20 characters"
        
        # Only alphanumeric and underscore
        if not person_id.replace('_', '').isalnum():
            return False, "Person ID can only contain letters, numbers, and underscores"
        
        return True, "Valid"
    
    @staticmethod
    def validate_email(email):
        """Basic email validation"""
        if not email:
            return True  # Email is optional
        
        if '@' not in email or '.' not in email:
            return False
        
        return True
    
    @staticmethod
    def export_to_json(data, filename):
        """Export data to JSON format"""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True, f"Data exported to {filename}"
        except Exception as e:
            return False, f"Export failed: {str(e)}"
    
    @staticmethod
    def generate_report(db_manager, start_date, end_date):
        """Generate attendance report for date range"""
        records = db_manager.get_all_attendance(start_date, end_date)
        
        report = {
            'report_date': datetime.now().isoformat(),
            'start_date': start_date,
            'end_date': end_date,
            'total_records': len(records),
            'records': []
        }
        
        for record in records:
            report['records'].append({
                'person_id': record[0],
                'name': record[1],
                'date': record[2],
                'arrival_time': record[3],
                'leaving_time': record[4],
                'status': record[5],
                'duration': Utils.calculate_duration(record[3], record[4])
            })
        
        return report
    
    @staticmethod
    def clean_old_backups(backup_folder='backups', keep_days=30):
        """Remove backups older than specified days"""
        if not os.path.exists(backup_folder):
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        removed_count = 0
        
        for filename in os.listdir(backup_folder):
            filepath = os.path.join(backup_folder, filename)
            
            if os.path.isfile(filepath):
                file_modified = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_modified < cutoff_date:
                    os.remove(filepath)
                    removed_count += 1
        
        return removed_count
    
    @staticmethod
    def get_system_info():
        """Get system information"""
        import platform
        import cv2
        
        return {
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__,
            'platform': platform.system(),
            'architecture': platform.machine()
        }
    
    @staticmethod
    def log_event(message, log_file='logs/system.log'):
        """Log an event to file"""
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        
        try:
            with open(log_file, 'a') as f:
                f.write(log_message)
            return True
        except:
            return False
    
    @staticmethod
    def calculate_attendance_percentage(db_manager, person_id, days=30):
        """Calculate attendance percentage for a person"""
        end_date = date.today().isoformat()
        start_date = (date.today() - timedelta(days=days)).isoformat()
        
        records = db_manager.get_person_attendance(person_id, start_date, end_date)
        
        total_days = days
        present_days = len(records)
        percentage = (present_days / total_days) * 100
        
        return {
            'total_days': total_days,
            'present_days': present_days,
            'absent_days': total_days - present_days,
            'percentage': round(percentage, 2)
        }
    
    @staticmethod
    def generate_attendance_summary(db_manager, start_date, end_date):
        """Generate attendance summary statistics"""
        records = db_manager.get_all_attendance(start_date, end_date)
        
        # Group by person
        person_attendance = {}
        for record in records:
            person_id = record[0]
            if person_id not in person_attendance:
                person_attendance[person_id] = {
                    'name': record[1],
                    'days_present': 0,
                    'total_hours': 0
                }
            
            person_attendance[person_id]['days_present'] += 1
            
            # Calculate hours if both times available
            if record[3] and record[4]:
                duration = Utils.calculate_duration(record[3], record[4])
                if 'h' in duration:
                    hours = int(duration.split('h')[0])
                    person_attendance[person_id]['total_hours'] += hours
        
        return person_attendance
    
    @staticmethod
    def export_detailed_report(db_manager, output_file, start_date=None, end_date=None):
        """Export detailed attendance report with duration calculations"""
        records = db_manager.get_all_attendance(start_date, end_date)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Person ID', 'Name', 'Date', 'Arrival Time', 
                'Leaving Time', 'Duration', 'Status'
            ])
            
            for record in records:
                duration = Utils.calculate_duration(record[3], record[4])
                writer.writerow([
                    record[0], record[1], record[2], 
                    record[3] or 'N/A', 
                    record[4] or 'N/A', 
                    duration, 
                    record[5]
                ])
        
        return True, f"Detailed report exported to {output_file}"


class ColorPrint:
    """Colored console output (for terminals that support ANSI)"""
    
    COLORS = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    
    @staticmethod
    def print(text, color='white'):
        """Print colored text"""
        if color in ColorPrint.COLORS:
            print(f"{ColorPrint.COLORS[color]}{text}{ColorPrint.COLORS['reset']}")
        else:
            print(text)
    
    @staticmethod
    def success(text):
        """Print success message in green"""
        ColorPrint.print(f"✓ {text}", 'green')
    
    @staticmethod
    def error(text):
        """Print error message in red"""
        ColorPrint.print(f"✗ {text}", 'red')
    
    @staticmethod
    def warning(text):
        """Print warning message in yellow"""
        ColorPrint.print(f"⚠ {text}", 'yellow')
    
    @staticmethod
    def info(text):
        """Print info message in blue"""
        ColorPrint.print(f"ℹ {text}", 'blue')

import threading
import cv2
import time

class ThreadedCamera:
    """
    Reads frames in a separate thread to prevent RTSP lag.
    Always returns the most recent frame.
    """
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.status = False
            self.frame = None
            return
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Minimize buffer
        self.status, self.frame = self.cap.read()
        self.stop_signal = False
        
        # Start the thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        """Thread worker function"""
        while True:
            if self.stop_signal:
                return
            
            if self.cap.isOpened():
                status, frame = self.cap.read()
                if status:
                    self.frame = frame
                    self.status = True
                else:
                    self.status = False
                    time.sleep(0.1) # Wait a bit if read fails
            else:
                time.sleep(0.1)
                
    def read(self):
        """Return the latest frame"""
        return self.status, self.frame
    
    def isOpened(self):
        return self.cap.isOpened()
        
    def release(self):
        self.stop_signal = True
        self.thread.join(timeout=1.0)
        self.cap.release()