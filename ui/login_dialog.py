import requests
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QFrame)
from PySide6.QtCore import Qt
import qtawesome as qta
from ui.theme_manager import ThemeManager

class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vision Attendance Cloud Login")
        self.setFixedSize(380, 480)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.token = None
        self.email = None
        self.password = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)

        # Logo/Icon
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon("fa5s.user-lock", color=ThemeManager.COLORS['primary']).pixmap(80, 80))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        title = QLabel("Cloud Access")
        title.setObjectName("Header")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Enter your api.visionattendance.com credentials")
        subtitle.setObjectName("SubHeader")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Inputs
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email Address")
        self.email_input.setFixedHeight(45)
        self.email_input.setStyleSheet(f"""
            padding: 0 12px; 
            border-radius: 8px; 
            background-color: {ThemeManager.COLORS['card']};
            border: 1px solid #334155;
            font-size: 14px;
        """)
        layout.addWidget(self.email_input)

        self.pass_input = QLineEdit()
        self.pass_input.setPlaceholderText("Password")
        self.pass_input.setEchoMode(QLineEdit.Password)
        self.pass_input.setFixedHeight(45)
        self.pass_input.setStyleSheet(f"""
            padding: 0 12px; 
            border-radius: 8px; 
            background-color: {ThemeManager.COLORS['card']};
            border: 1px solid #334155;
            font-size: 14px;
        """)
        layout.addWidget(self.pass_input)

        layout.addSpacing(5)

        self.login_btn = QPushButton("LOGIN")
        self.login_btn.setObjectName("PrimaryAction")
        self.login_btn.setFixedHeight(45)
        self.login_btn.setIcon(qta.icon("fa5s.arrow-right", color="#0f172a"))
        self.login_btn.setDefault(True)
        self.login_btn.clicked.connect(self.handle_login)
        layout.addWidget(self.login_btn)

        self.close_btn = QPushButton("CLOSE")
        self.close_btn.setObjectName("SecondaryAction")
        self.close_btn.setFixedHeight(45)
        self.close_btn.setStyleSheet(f"""
            color: {ThemeManager.COLORS['text_dim']}; 
            border: 1px solid #334155; 
            border-radius: 8px;
            background-color: transparent;
        """)
        self.close_btn.clicked.connect(self.reject)
        layout.addWidget(self.close_btn)

        layout.addStretch()

        help_label = QLabel("Forgot password? Contact your administrator.")
        help_label.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 11px; margin-top: 5px;")
        help_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(help_label)

    def handle_login(self):
        email = self.email_input.text().strip()
        password = self.pass_input.text().strip()

        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter both email and password.")
            return

        self.login_btn.setEnabled(False)
        self.login_btn.setText("AUTHENTICATING...")
        
        # Using the base URL from the user requirements
        base_url = "https://api.visionattendance.com/api"
        
        try:
            response = requests.post(
                f"{base_url}/auth/login",
                json={"email": email, "password": password},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token") or data.get("token")
                if self.token:
                    self.email = email
                    self.password = password
                    # [NEW] Save session for unattended restarts
                    from config.auth_manager import AuthManager
                    AuthManager.save_credentials(email, password)
                    self.accept()
                else:
                    QMessageBox.critical(self, "Error", "Access token not found in response.")
            elif response.status_code == 401:
                QMessageBox.warning(self, "Unauthorized", "Invalid email or password.")
            else:
                QMessageBox.critical(self, "Error", f"Server error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            QMessageBox.critical(self, "Connection Error", f"Failed to connect to cloud API: {str(e)}")
        
        self.login_btn.setEnabled(True)
        self.login_btn.setText("SIGN IN")
