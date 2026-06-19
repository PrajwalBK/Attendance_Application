import cv2
from core.api_client import APIClient
from core.face_recognition import FaceRecognitionHandler

class RegistrationModule:
    def __init__(self, db_manager, face_handler):
        self.db_manager = db_manager
        self.face_handler = face_handler
    
    def register_person_from_webcam(self, person_id, name, email=None, department=None):
        """Register a person using webcam capture"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return False, "Could not open webcam"
        
        print(f"\n{'='*50}")
        print(f"REGISTERING: {name} (ID: {person_id})")
        print(f"{'='*50}")
        print("Instructions:")
        print("1. Position your face in the center of the frame")
        print("2. Ensure good lighting")
        print("3. Look directly at the camera")
        print("4. Press 'c' to CAPTURE")
        print("5. Press 'q' to CANCEL")
        print(f"{'='*50}\n")
        
        face_captured = False
        face_encoding = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in real-time
            faces = self.face_handler.detect_faces(frame)
            
            # Draw rectangles around detected faces
            display_frame = frame.copy()
            
            if len(faces) == 0:
                status_text = "No face detected"
                color = (0, 0, 255)  # Red
            elif len(faces) > 1:
                status_text = "Multiple faces detected - Only one person allowed"
                color = (0, 165, 255)  # Orange
            else:
                status_text = "Face detected - Press 'c' to capture"
                color = (0, 255, 0)  # Green
                
                # Draw bounding box
                face = faces[0]
                bbox = face.bbox.astype(int)
                cv2.rectangle(display_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            color, 2)
                
                # Draw landmarks
                if hasattr(face, 'kps') and face.kps is not None:
                    for point in face.kps:
                        cv2.circle(display_frame, 
                                 (int(point[0]), int(point[1])), 
                                 3, (0, 255, 255), -1)
            
            # Display status
            cv2.putText(display_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display person info
            info_text = f"Name: {name} | ID: {person_id}"
            cv2.putText(display_frame, info_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Registration - Face Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if len(faces) == 1:
                    # Extract BOTH standard (512-dim) and mask (128-dim) encodings — V5 pipeline
                    std_encoding, mask_encoding, enc_msg = self.face_handler.extract_face_encodings(frame)
                    if std_encoding is None:
                        print(f"✗ Face encoding failed: {enc_msg}")
                        continue
                    face_captured = True
                    print("✓ Face captured successfully!")
                    break
                else:
                    print("✗ Cannot capture: Ensure only one face is visible")
            
            elif key == ord('q'):
                print("Registration cancelled by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_captured:
            return False, "Face capture cancelled or failed"
        
        # Save to database (pass both standard and mask encodings)
        success, message = self.db_manager.add_person(
            person_id, name, std_encoding, email, department,
            mask_face_encoding=mask_encoding
        )
        
        if success:
            # Also add to face recognition handler
            self.face_handler.add_face_encoding(person_id, name, std_encoding, mask_encoding)
            print(f"✓ {name} registered successfully!")
            return True, message
        else:
            print(f"✗ Registration failed: {message}")
            return False, message
    
    def register_person_from_image(self, person_id, name, image_path, 
                                   email=None, department=None):
        """Register a person from an image file"""
        # Read image
        frame = cv2.imread(image_path)
        
        if frame is None:
            return False, "Could not read image file"
        
        # Extract BOTH standard (512-dim) and mask (128-dim) encodings — V5 pipeline
        std_encoding, mask_encoding, message = self.face_handler.extract_face_encodings(frame)
        
        if std_encoding is None:
            return False, message
        
        # Save to database (pass both standard and mask encodings)
        success, message = self.db_manager.add_person(
            person_id, name, std_encoding, email, department,
            mask_face_encoding=mask_encoding
        )
        
        if success:
            # Also add to face recognition handler
            self.face_handler.add_face_encoding(person_id, name, std_encoding, mask_encoding)
            return True, message
        
        return False, message
    
    def update_person_info(self, person_id, name=None, email=None, department=None):
        """Update person information (not face encoding)"""
        # This would require additional database methods
        # Implementation depends on specific requirements
        pass
    
    def delete_person(self, person_id):
        """Delete a person from the system"""
        # Remove from database
        success, message = self.db_manager.delete_person(person_id)
        
        if success:
            # Remove from face recognition handler
            self.face_handler.remove_face_encoding(person_id)
            return True, message
        
        return False, message
    
    def list_registered_persons(self):
        """List all registered persons"""
        persons = self.db_manager.get_all_persons_details()
        
        if not persons:
            print("\nNo persons registered yet.")
            return
        
        print(f"\n{'='*80}")
        print(f"{'ID':<15} {'Name':<25} {'Email':<20} {'Department':<15}")
        print(f"{'='*80}")
        
        for person in persons:
            person_id = person[0]
            name = person[1]
            email = person[2] if person[2] else "N/A"
            department = person[3] if person[3] else "N/A"
            
            print(f"{person_id:<15} {name:<25} {email:<20} {department:<15}")
        
        print(f"{'='*80}")
        print(f"Total Registered: {len(persons)}")
        print(f"{'='*80}\n")
    
    def verify_registration(self, person_id):
        """Verify if a person is registered"""
        person = self.db_manager.get_person(person_id)
        return person is not None