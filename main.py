import cv2
import time
import warnings
import traceback
from datetime import datetime
from core.face_recognition import FaceRecognitionHandler
from core.attendance_tracker import AttendanceTracker
from core.video_processor import VideoProcessor
from config.config import get_config

# Filter warnings to keep console clean
warnings.filterwarnings("ignore")

class AttendanceSystem:
    def __init__(self):
        print("Initializing Face Attendance System...")
        
        # Initialize components
        config = get_config()
        if config.get('use_api', True):
            print(f"[API] Using Remote API: {config.get('api_base_url')}")
            from core.api_client import APIClient
            self.db_manager = APIClient(config.get('api_base_url'))
        else:
            print("[DB] Using Local Database directly")
            from database.database import DatabaseManager
            self.db_manager = DatabaseManager()
            
        role = config.get('camera_role', 'in').upper()
        print(f"[CAMERA] Camera Role Configured: {role}")
            
        self.face_handler = FaceRecognitionHandler(self.db_manager)
        self.attendance_tracker = AttendanceTracker(self.db_manager, self.face_handler, event_type=role.lower())
        self.video_processor = VideoProcessor(self.face_handler)
        
        print("[+] System initialized successfully!")
    
    def display_menu(self):
        """Display main menu"""
        print("\n" + "="*60)
        print("FACE ATTENDANCE SYSTEM (CLI MODE)")
        print("="*60)
        print("1. START Automatic Attendance (Camera)")
        print("2. Register New Person (With Shifts)")
        print("3. View Today's Records")
        print("4. Export Attendance to CSV")
        print("5. Exit")
        print("="*60)
    
    def start_automatic_attendance(self):
        """Run the main attendance system"""
        source = get_config()['webcam_index']
        print(f"Connecting to camera: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam ({source})")
            return
        
        print("\n" + "="*60)
        print("       AUTOMATIC ATTENDANCE RUNNING")
        print("="*60)
        print("• First Detection = LOGIN")
        print("• Shift End Rule  = Updates LOGOUT if time > Shift End")
        print("• Raw Logs        = Saved every 90 seconds")
        print("\nPress 'q' to Stop and return to menu.")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read from webcam")
                break
            
            # Process frame
            tracked_detections, labels, faces, messages = \
                self.video_processor.process_frame(
                    frame, 
                    mark_attendance_callback=self.attendance_tracker.process_recognized_face,
                    unknown_person_callback=self.attendance_tracker.process_unknown_person
                )
            
            # Generate annotated frame
            annotated_frame = self.video_processor.annotate_frame(frame, tracked_detections, labels, faces)
            
            # Print messages to console (optional)
            for msg in messages:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Draw Info on Screen
            stats = self.db_manager.get_statistics()
            info = {
                'System': 'AUTO',
                'Registered': stats['total_persons'],
                'Present': stats['present_today'],
                'FPS': f"{fps}"
            }
            
            annotated_frame = self.video_processor.draw_info_panel(annotated_frame, info)
            
            # Display
            cv2.imshow("Face Attendance System", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping attendance system...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def register_person_interactive(self):
        """Interactive person registration with Shift Support"""
        print("\n" + "="*60)
        print("       PERSON REGISTRATION")
        print("="*60)
        
        # 1. Get Inputs
        person_id = input("Enter Person ID (unique): ").strip()
        if not person_id: return
        
        name = input("Enter Full Name: ").strip()
        if not name: return
        
        email = input("Enter Email (optional): ").strip() or None
        department = input("Enter Department (optional): ").strip() or None
        
        # 2. Get Shift Info (New)
        print("\n--- Shift Settings ---")
        shift_start = input("Shift Start (HH:MM) [Default 09:00]: ").strip() or "09:00"
        shift_end = input("Shift End (HH:MM)   [Default 18:00]: ").strip() or "18:00"
        
        # 3. Capture Face
        print("\nOpening Camera for Face Capture...")
        print("Press 'c' to CAPTURE, 'q' to CANCEL")
        
        cap = cv2.VideoCapture(0)
        face_encoding = None
        success_capture = False
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Show preview
            cv2.imshow("Registration - Press 'c' to Capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Analyze frame
                print("Analyzing face...")
                face_encoding, msg = self.face_handler.extract_face_encoding(frame)
                
                if face_encoding is not None:
                    # Check for duplicates
                    self.face_handler.load_face_encodings()
                    exist_id, exist_name, sim = self.face_handler.recognize_face(face_encoding)
                    
                    if exist_id is not None and sim > 0.5:
                        print(f"\n⚠ DUPLICATE DETECTED! Matches {exist_name} ({exist_id})")
                        print("Try again or press 'q' to quit.")
                    else:
                        success_capture = True
                        break
                else:
                    print(f"⚠ Face not detected: {msg}")
            
            elif key == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        # 4. Save to DB
        if success_capture and face_encoding is not None:
            success, message = self.db_manager.add_person(
                person_id, name, face_encoding, email, department, shift_start, shift_end
            )
            
            if success:
                self.face_handler.add_face_encoding(person_id, name, face_encoding)
                print(f"\n✓ SUCCESS: {name} registered with shift {shift_start}-{shift_end}")
            else:
                print(f"\n✗ Database Error: {message}")
        else:
            print("\nRegistration Cancelled.")
    
    def view_today_attendance(self):
        """View today's attendance records"""
        records = self.db_manager.get_today_attendance()
        
        print("\n" + "="*90)
        print(f"       TODAY'S ATTENDANCE - {datetime.now().strftime('%Y-%m-%d')}")
        print("="*90)
        
        if not records:
            print("No attendance records for today")
        else:
            # Fixed format string to handle None types
            print(f"{'ID':<15} {'Name':<25} {'Login':<15} {'Last Seen':<15} {'Status':<10}")
            print("-"*90)
            
            for record in records:
                # record = (id, name, arrival, leaving, status)
                pid = record[0]
                name = record[1]
                
                # FIX: Convert None to "N/A" safely
                arrival = record[2] if record[2] else "N/A"
                leaving = record[3] if record[3] else "N/A"
                status = record[4]
                
                print(f"{pid:<15} {name:<25} {arrival:<15} {leaving:<15} {status:<10}")
        
        print("="*90 + "\n")
        input("Press Enter to continue...")
    
    def export_attendance(self):
        """Export attendance to CSV"""
        filename = f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        success, message = self.db_manager.export_to_csv(filename)
        
        if success:
            print(f"\n✓ {message}")
        else:
            print(f"\n✗ Export failed: {message}")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop"""
        while True:
            self.display_menu()
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.start_automatic_attendance()
            elif choice == '2':
                self.register_person_interactive()
            elif choice == '3':
                self.view_today_attendance()
            elif choice == '4':
                self.export_attendance()
            elif choice == '5':
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    import multiprocessing
    import traceback
    # 1. Set Start Method to Spawn (Windows Standard)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 2. Freeze Support for EXE
    multiprocessing.freeze_support()

    try:
        system = AttendanceSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()