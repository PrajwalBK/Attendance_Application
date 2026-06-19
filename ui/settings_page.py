from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QLineEdit, QFormLayout, QMessageBox)
from PySide6.QtCore import Qt
import qtawesome as qta
from ui.theme_manager import ThemeManager
from config.cam_config_manager import CamConfigManager

class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(30)
        
        # Header
        header = QVBoxLayout()
        title = QLabel("System Settings")
        title.setObjectName("Header")
        header.addWidget(title)
        
        desc = QLabel("Configure camera sources and link your remote account.")
        desc.setObjectName("SubHeader")
        header.addWidget(desc)
        main_layout.addLayout(header)
        
        # Sections Container
        sections = QVBoxLayout()
        sections.setSpacing(20)
        
        # Camera Settings Section
        cam_section = self.create_section("Camera Infrastructure", "fa5s.network-wired")
        cam_form = QFormLayout()
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://admin:pass@ip:port/stream")
        self.rtsp_input.setStyleSheet("padding: 10px; background-color: #1e293b; border-radius: 6px;")
        cam_form.addRow("RTSP Template:", self.rtsp_input)
        
        help_text = QLabel("Use {channel} placeholder for channel-based cameras.")
        help_text.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 10px;")
        cam_form.addRow("", help_text)
        
        cam_section.layout().addLayout(cam_form)
        sections.addWidget(cam_section)
        
        # Remote API Section
        api_section = self.create_section("Remote API Configuration", "fa5s.cloud")
        api_form = QFormLayout()
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)
        self.token_input.setPlaceholderText("eyJhbGciOiJIUzI1NiIsInR...")
        self.token_input.setStyleSheet("padding: 10px; background-color: #1e293b; border-radius: 6px;")
        api_form.addRow("Bearer Token:", self.token_input)
        
        api_section.layout().addLayout(api_form)
        sections.addWidget(api_section)
        
        main_layout.addLayout(sections)
        
        # Save Button
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_btn = QPushButton("SAVE ALL SETTINGS")
        self.save_btn.setObjectName("PrimaryAction")
        self.save_btn.setFixedHeight(50)
        self.save_btn.setFixedWidth(250)
        self.save_btn.setIcon(qta.icon("fa5s.save", color="#0f172a"))
        self.save_btn.clicked.connect(self.save_settings)
        save_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(save_layout)
        main_layout.addStretch()

    def create_section(self, title, icon_name):
        frame = QFrame()
        frame.setStyleSheet(f"background-color: {ThemeManager.COLORS['card']}; border: 1px solid #334155; border-radius: 12px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 20, 20, 20)
        
        header = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(qta.icon(icon_name, color=ThemeManager.COLORS['primary']).pixmap(20, 20))
        header.addWidget(icon)
        
        lbl = QLabel(title)
        lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        header.addWidget(lbl)
        header.addStretch()
        layout.addLayout(header)
        layout.addSpacing(10)
        
        return frame

    def load_settings(self):
        conf = CamConfigManager.load_config()
        self.rtsp_input.setText(conf.get('rtsp_template', ''))
        
        # Load token from some global storage - for now just local storage mockup
        # In a real app we might use QSettings or original config.py
        import os
        token_file = 'config/api_token.txt'
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    self.token_input.setText(f.read().strip())
            except: pass

    def save_settings(self):
        # Update Camera Config
        conf = CamConfigManager.load_config()
        conf['rtsp_template'] = self.rtsp_input.text().strip()
        success, msg = CamConfigManager.save_config(conf)
        
        # Update Token
        token = self.token_input.text().strip()
        try:
            os.makedirs('config', exist_ok=True)
            with open('config/api_token.txt', 'w') as f:
                f.write(token)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save API token: {str(e)}")
            return

        if success:
            QMessageBox.information(self, "Success", "Settings updated successfully!")
        else:
            QMessageBox.critical(self, "Error", f"Failed to save camera config: {msg}")
