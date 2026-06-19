import cv2
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QLineEdit, QFormLayout, QMessageBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import qtawesome as qta
from ui.theme_manager import ThemeManager

class RegistrationPage(QWidget):
    def __init__(self, backend_controller):
        super().__init__()
        self.backend = backend_controller
        self.setup_ui()
        
        # Timer for preview
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(50)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(40)
        
        # Left Side: Form
        form_container = QVBoxLayout()
        
        title = QLabel("User Registration")
        title.setObjectName("Header")
        form_container.addWidget(title)
        
        desc = QLabel("Enter user details and look at the camera for enrollment.")
        desc.setObjectName("SubHeader")
        desc.setWordWrap(True)
        form_container.addWidget(desc)
        
        form_container.addSpacing(30)
        
        form = QFormLayout()
        form.setSpacing(15)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full Name")
        self.name_input.setStyleSheet("padding: 12px; border-radius: 8px; background-color: #1e293b;")
        form.addRow("Name:", self.name_input)
        
        self.dept_input = QLineEdit()
        self.dept_input.setPlaceholderText("Department (e.g. Sales, dev)")
        self.dept_input.setStyleSheet("padding: 12px; border-radius: 8px; background-color: #1e293b;")
        form.addRow("Dept:", self.dept_input)
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Employee ID / Alias")
        self.id_input.setStyleSheet("padding: 12px; border-radius: 8px; background-color: #1e293b;")
        form.addRow("ID:", self.id_input)
        
        form_container.addLayout(form)
        form_container.addSpacing(30)
        
        self.reg_btn = QPushButton("CAPTURE & REGISTER")
        self.reg_btn.setObjectName("PrimaryAction")
        self.reg_btn.setFixedHeight(50)
        self.reg_btn.setIcon(qta.icon("fa5s.camera", color="#0f172a"))
        self.reg_btn.clicked.connect(self.handle_registration)
        form_container.addWidget(self.reg_btn)
        
        form_container.addStretch()
        main_layout.addLayout(form_container, 2)
        
        # Right Side: Camera Preview
        preview_container = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(640, 480)
        self.preview_label.setStyleSheet("background-color: black; border: 2px solid #334155; border-radius: 12px;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Camera 1 (IN) required for registration")
        preview_container.addWidget(self.preview_label)
        
        info = QLabel("Tip: Ensure face is well-lit and centered in the frame.")
        info.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 11px;")
        info.setAlignment(Qt.AlignCenter)
        preview_container.addWidget(info)
        
        main_layout.addLayout(preview_container, 3)

    def update_preview(self):
        if self.backend.caps[0]:
            ret, frame = self.backend.caps[0].read()
            if ret and frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.preview_label.setPixmap(QPixmap.fromImage(qt_img).scaled(640, 480, Qt.KeepAspectRatio))
                return
        
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Please Start Camera 1 (IN) in Camera Feed tab")

    def handle_registration(self):
        name = self.name_input.text().strip()
        dept = self.dept_input.text().strip()
        uid = self.id_input.text().strip()
        
        if not name or not uid:
            QMessageBox.warning(self, "Validation Error", "Name and ID are required.")
            return
            
        self.reg_btn.setEnabled(False)
        self.reg_btn.setText("PROCESSING...")
        
        success, msg = self.backend.register_user(name, dept, uid)
        
        if success:
            QMessageBox.information(self, "Success", msg)
            self.name_input.clear()
            self.dept_input.clear()
            self.id_input.clear()
        else:
            QMessageBox.critical(self, "Error", msg)
            
        self.reg_btn.setEnabled(True)
        self.reg_btn.setText("CAPTURE & REGISTER")
