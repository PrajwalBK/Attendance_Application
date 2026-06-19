from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QPushButton, QMessageBox, QFrame, QFormLayout, QComboBox, QGridLayout, QScrollArea)
from PySide6.QtCore import Qt, QSize, Slot, QMetaObject, Q_ARG, Signal
import qtawesome as qta
from ui.theme_manager import ThemeManager
from config.cam_config_manager import CamConfigManager
from core.gui_workers import MAX_CAMS

class CCTVSetupWizard(QWidget):
    setup_finished = Signal(bool)
    def __init__(self, backend_controller, parent=None):
        super().__init__(parent)
        self.backend = backend_controller
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Scroll area wrapping to prevent alignment squishing on small window sizes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        
        scroll_content = QWidget()
        scroll_content.setObjectName("ScrollContent")
        scroll_content.setStyleSheet("QWidget#ScrollContent { background: transparent; }")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setAlignment(Qt.AlignCenter)
        
        # Premium Card
        card = QFrame()
        card.setObjectName("Card")
        card.setMaximumWidth(700)
        card.setMinimumWidth(350)
        card.setStyleSheet(f"""
            QFrame#Card {{
                background-color: #0b111e; 
                border: 1px solid #1e293b; 
                border-radius: 24px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(35, 30, 35, 30) # Compacted from 50, 50, 50, 50
        card_layout.setSpacing(18)                     # Compacted from 30
        
        # Header in Card
        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)
        
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon("fa5s.network-wired", color=ThemeManager.COLORS['primary']).pixmap(44, 44))
        icon_lbl.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_lbl)
        
        title = QLabel("CCTV Infrastructure")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        
        desc = QLabel("Configure your network video recorder (NVR) or camera system to enable AI monitoring.")
        desc.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 13px; line-height: 1.4;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(desc)
        
        card_layout.addLayout(header_layout)
        
        # Form
        form_container = QWidget()
        form_layout = QGridLayout(form_container)
        form_layout.setSpacing(12) # Compacted from 20
        
        # Load existing config for pre-filling
        conf = CamConfigManager.load_config()
        rtsp = conf.get('rtsp_template', '')
        
        # Simple parser for rtsp://user:pass@ip:port/path
        parsed = {"ip": "192.168.1.100", "port": "554", "user": "admin", "pass": ""}
        if rtsp.startswith("rtsp://"):
            try:
                import urllib.parse
                # Robust parser: find the LAST '@' to split creds from host
                no_proto = rtsp[7:]  # strip "rtsp://"
                last_at = no_proto.rfind('@')
                if last_at != -1:
                    cred = no_proto[:last_at]       # admin:P@ss
                    host = no_proto[last_at + 1:]   # 192.168.1.240:554/path
                    colon = cred.index(':')
                    parsed["user"] = cred[:colon]
                    parsed["pass"] = urllib.parse.unquote(cred[colon + 1:])
                    host_port = host.split('/')[0]   # 192.168.1.240:554
                    if ':' in host_port:
                        parsed["ip"], parsed["port"] = host_port.split(':', 1)
                    else:
                        parsed["ip"] = host_port
                
                # [FIX] If the template has {ip}, pull the REAL IP from the first camera for the UI
                if parsed["ip"] == "{ip}":
                    cams = conf.get('cams', [])
                    if cams:
                        parsed["ip"] = cams[0].get('ip', '192.168.1.100')
            except: pass

        self.ip_input = self.create_styled_input("Host IP Address", "fa5s.network-wired", parsed["ip"])
        self.port_input = self.create_styled_input("RTSP Port", "fa5s.plug", parsed["port"])
        
        form_layout.addWidget(self.create_label("HOST / IP"), 0, 0)
        form_layout.addWidget(self.ip_input, 0, 1)
        form_layout.addWidget(self.create_label("PORT"), 0, 2)
        form_layout.addWidget(self.port_input, 0, 3)
        
        self.user_input = self.create_styled_input("Username", "fa5s.user", parsed["user"])
        self.pass_input = self.create_styled_input("Password", "fa5s.key", parsed["pass"], password=True)
        
        form_layout.addWidget(self.create_label("USERNAME"), 1, 0)
        form_layout.addWidget(self.user_input, 1, 1, 1, 3)
        
        form_layout.addWidget(self.create_label("PASSWORD"), 2, 0)
        form_layout.addWidget(self.pass_input, 2, 1, 1, 3)
        
        # Brand
        self.brand_selector = QComboBox()
        self.brand_selector.addItems(["AUTO-DETECT (Recommended)", "Hikvision", "CP Plus", "Eyematic", "Custom Template"])
        self.brand_selector.setFixedHeight(40) # Compacted from 45
        self.brand_selector.setStyleSheet(f"""
            QComboBox {{
                background-color: #111a2f; 
                color: white; 
                border-radius: 8px; 
                border: 1px solid #1a2540; 
                padding: 0 15px;
            }}
            QComboBox::drop-down {{ border: none; }}
        """)
        
        form_layout.addWidget(self.create_label("HARDWARE"), 3, 0)
        form_layout.addWidget(self.brand_selector, 3, 1, 1, 3)
        
        card_layout.addWidget(form_container)
        
        # Actions
        btn_layout = QHBoxLayout()
        
        self.test_btn = QPushButton("TEST CONNECTION")
        self.test_btn.setObjectName("SecondaryAction")
        self.test_btn.setFixedHeight(40) # Compacted from 45
        self.test_btn.setCursor(Qt.PointingHandCursor)
        self.test_btn.setIcon(qta.icon("fa5s.vial", color="#3b82f6"))
        self.test_btn.clicked.connect(self.test_camera_link)
        btn_layout.addWidget(self.test_btn)
        
        self.save_btn = QPushButton("SAVE & CONNECT")
        self.save_btn.setObjectName("PrimaryAction")
        self.save_btn.setFixedHeight(40) # Compacted from 45
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setIcon(qta.icon("fa5s.save", color="#0f172a"))
        self.save_btn.clicked.connect(self.generate_and_save)
        btn_layout.addWidget(self.save_btn)
        
        card_layout.addLayout(btn_layout)
        
        # [NEW] Styles for password toggle
        self.setStyleSheet(self.styleSheet() + """
            QPushButton#PasswordToggle {
                background: transparent;
                border: none;
                padding: 0;
            }
            QPushButton#PasswordToggle:hover {
                background: rgba(255,255,255,0.05);
                border-radius: 4px;
            }
        """)
        
        footer = QLabel("Changes will take effect after restarting cameras.")
        footer.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 11px; font-style: italic;")
        footer.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(footer)
        
        content_layout.addWidget(card)
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

    def create_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        return lbl

    def create_styled_input(self, placeholder, icon_name, default_val="", password=False):
        container = QFrame()
        container.setFixedHeight(40) # Compacted from 45
        container.setStyleSheet("background-color: #111a2f; border-radius: 8px; border: 1px solid #1a2540;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(15, 0, 15, 0)
        
        icon = QLabel()
        icon.setPixmap(qta.icon(icon_name, color="#475569").pixmap(14, 14))
        layout.addWidget(icon)
        
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setText(default_val)
        edit.setStyleSheet("border: none; background: transparent; color: white; font-size: 13px;")
        
        if password: 
            edit.setEchoMode(QLineEdit.Password)
            # [NEW] Add Toggle Eye Icon
            toggle_btn = QPushButton()
            toggle_btn.setObjectName("PasswordToggle")
            toggle_btn.setCursor(Qt.PointingHandCursor)
            toggle_btn.setIcon(qta.icon("fa5s.eye", color="#475569"))
            toggle_btn.setFixedSize(24, 24)
            toggle_btn.clicked.connect(lambda: self.toggle_password_visibility(edit, toggle_btn))
            layout.addWidget(edit)
            layout.addWidget(toggle_btn)
        else:
            layout.addWidget(edit)
        
        container.input_field = edit
        return container

    def toggle_password_visibility(self, line_edit, button):
        if line_edit.echoMode() == QLineEdit.Password:
            line_edit.setEchoMode(QLineEdit.Normal)
            button.setIcon(qta.icon("fa5s.eye-slash", color=ThemeManager.COLORS['primary']))
        else:
            line_edit.setEchoMode(QLineEdit.Password)
            button.setIcon(qta.icon("fa5s.eye", color="#475569"))

    def test_camera_link(self):
        import urllib.parse
        import threading
        ip = self.ip_input.input_field.text().strip()
        user = self.user_input.input_field.text().strip()
        password = self.pass_input.input_field.text().strip()
        port = self.port_input.input_field.text().strip()
        brand = self.brand_selector.currentText().strip()
        
        if not ip or not user or not password:
            QMessageBox.warning(self, "Incomplete Data", "Enter IP, Username, and Password to test.")
            return

        # [FIX] Properly encode password for the test URL to handle symbols like '@'
        safe_pass = urllib.parse.quote(password, safe='')
        test_url = f"rtsp://{user}:{safe_pass}@{ip}:{port}/{{path}}"
        self.test_btn.setText("TESTING...")
        self.test_btn.setEnabled(False)
        
        def run_test():
            path, status = self.backend.discover_brand_path(test_url, channel_hint="1", brand=brand)
            
            QMetaObject.invokeMethod(self, "show_test_result", Qt.QueuedConnection,
                                     Q_ARG(str, path if path else ""), Q_ARG(str, status))

        threading.Thread(target=run_test, daemon=True).start()

    @Slot(str, str)
    def show_test_result(self, path, status):
        if status == "OK":
            QMessageBox.information(self, "Success!", f"Connection established! Brand-path identified as:\n{path}")
        elif status == "UNAUTHORIZED":
            QMessageBox.warning(self, "Host Found", "Host IP and Port found, but Authentication failed (401).\n\nPlease verify your Username and Password.")
        else:
            QMessageBox.critical(self, "No Camera Found", "Could not find any RTSP camera at this IP/Port.\n\nTips:\n1. Check if the IP is correct.\n2. Ensure the NVR/Camera is powered on.\n3. Verify if RTSP is enabled on the camera settings.")
            
        self.test_btn.setText("TEST CONNECTION")
        self.test_btn.setEnabled(True)

    def generate_and_save(self):
        import urllib.parse
        ip = self.ip_input.input_field.text().strip()
        user = self.user_input.input_field.text().strip()
        password = self.pass_input.input_field.text().strip()
        port = self.port_input.input_field.text().strip()
        brand = self.brand_selector.currentText()

        if not ip or not user or not password:
            QMessageBox.warning(self, "Configuration Incomplete", "Please provide all required network credentials.")
            return

        self.save_btn.setEnabled(False)
        self.save_btn.setText("INITIALIZING DISCOVERY...")
        
        def run_discovery():
            import time
            import urllib.parse
            safe_pass = urllib.parse.quote(password, safe='')

            path_str = "{path}"
            brand_sel = brand.strip()
            is_specific_brand = brand_sel in ["Hikvision", "CP Plus", "Dahua / CP Plus", "Eyematic"]
            
            if brand_sel == "Hikvision":
                path_str = "Streaming/Channels/{channel}01"
            elif brand_sel in ["Dahua / CP Plus", "CP Plus"]:
                path_str = "cam/realmonitor?channel={channel}&subtype=0"
            elif brand_sel == "Eyematic":
                path_str = "ch{channel}/main"
                
            final_template = f"rtsp://{user}:{safe_pass}@{{ip}}:{port}/{path_str}"

            # Probe only the target IP address and selected brand's paths
            QMetaObject.invokeMethod(self, "update_status", Qt.QueuedConnection, Q_ARG(str, "🔍 Identifying Hardware Brand..."))
            test_url = f"rtsp://{user}:{safe_pass}@{ip}:{port}/{{path}}"
            
            brand_filter = brand_sel if is_specific_brand else None
            new_template, status = self.backend.discover_brand_path(test_url, channel_hint="1", brand=brand_filter)
            if new_template and "rtsp://" in new_template:
                final_template = new_template.replace(f"@{ip}:", "@{ip}:")
            
            discovered_ips = [ip]

            conf = CamConfigManager.load_config()
            conf['rtsp_template'] = final_template
            conf['discovered_ips'] = discovered_ips
            conf['brand'] = brand_sel
            
            QMetaObject.invokeMethod(self, "update_status", Qt.QueuedConnection, Q_ARG(str, f"✨ Connection Verified. Retrieving NVR Channels..."))
            
            active_channels = []
            test_template = final_template.replace("{ip}", ip)
            _, active_channels = self.backend.enumerate_nvr_channels(test_template)
            
            for i in range(MAX_CAMS):
                if i < len(conf['cams']):
                    conf['cams'][i]['ip'] = ip
                    conf['cams'][i]['template'] = final_template.replace(f"@{ip}:", "@{ip}:")
                    if active_channels:
                        if i < len(active_channels):
                            conf['cams'][i]['source'] = f"Camera {active_channels[i]}"
                        else:
                            conf['cams'][i]['source'] = ""
                    else:
                        conf['cams'][i]['source'] = f"Camera {i+1}"
                        
            for i in range(MAX_CAMS):
                if i < len(conf['cams']):
                    print(f"[WIZARD] Mapped CAM_{i+1} -> IP: {conf['cams'][i].get('ip')} | Source: {conf['cams'][i].get('source')}")
            
            CamConfigManager.save_config(conf)
            if self.backend:
                self.backend.restart_cameras()
                
            QMetaObject.invokeMethod(self, "finish_save", Qt.QueuedConnection, Q_ARG(bool, True))

        import threading
        threading.Thread(target=run_discovery, daemon=True).start()

    @Slot(str)
    def update_status(self, text):
        self.save_btn.setText(text.upper())

    @Slot(bool)
    def finish_save(self, success):
        self.save_btn.setEnabled(True)
        self.save_btn.setText("SAVE & CONNECT")
        if success:
            QMessageBox.information(self, "Cloud Connected", "Discovery & Linkage Complete!\n\nAll cameras have been identified and started.")
            self.setup_finished.emit(True)
