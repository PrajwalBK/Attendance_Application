from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QLineEdit, QPushButton, QMessageBox, QFrame, QGridLayout, 
                              QRadioButton, QButtonGroup, QStackedWidget, QScrollArea)
from PySide6.QtCore import Qt, Slot, QMetaObject, Q_ARG
import qtawesome as qta
from ui.theme_manager import ThemeManager
from config.auth_manager import AuthManager
from config.db_config_manager import DBConfigManager

class CloudSetupPage(QWidget):
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
        
        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(8)
        
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon("fa5s.network-wired", color=ThemeManager.COLORS['primary']).pixmap(44, 44))
        icon_lbl.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(icon_lbl)
        
        title = QLabel("Connection Setup")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: white;")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        
        desc = QLabel("Configure how the system stores and synchronizes data. Select Production Cloud, Local API or Direct Local Database.")
        desc.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 13px; line-height: 1.4;")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(desc)
        
        card_layout.addLayout(header_layout)
        
        # Mode Selector
        mode_frame = QFrame()
        mode_frame.setStyleSheet("background-color: #111a2f; border-radius: 12px; padding: 5px;")
        mode_layout = QHBoxLayout(mode_frame)
        mode_layout.setContentsMargins(15, 5, 15, 5)
        
        self.mode_group = QButtonGroup(self)
        
        self.cloud_radio = QRadioButton("Production Cloud")
        self.cloud_radio.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
        
        self.local_api_radio = QRadioButton("Local API")
        self.local_api_radio.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")
        
        self.db_radio = QRadioButton("Direct Local DB")
        self.db_radio.setStyleSheet("color: #94a3b8; font-weight: bold; font-size: 11px;")
        
        self.mode_group.addButton(self.cloud_radio, 0)
        self.mode_group.addButton(self.db_radio, 1)
        self.mode_group.addButton(self.local_api_radio, 2)
        
        mode_layout.addWidget(self.cloud_radio)
        mode_layout.addStretch()
        mode_layout.addWidget(self.local_api_radio)
        mode_layout.addStretch()
        mode_layout.addWidget(self.db_radio)
        
        card_layout.addWidget(mode_frame)
        
        # --- API FORM WIDGET ---
        self.api_form_widget = QWidget()
        self.api_form_widget.setMinimumHeight(180)
        api_grid = QGridLayout(self.api_form_widget)
        api_grid.setContentsMargins(0, 0, 0, 0)
        api_grid.setSpacing(12)
        
        # Load existing auth
        email, password, api_url = AuthManager.load_credentials()
        
        self.url_input = self.create_styled_input("API Base URL (e.g. https://api.visionattendance.com)", "fa5s.link", api_url)
        self.email_input = self.create_styled_input("Administrator Email", "fa5s.envelope", email if email else "")
        self.pass_input = self.create_styled_input("Administrator Password", "fa5s.key", password if password else "", password=True)
        
        api_grid.addWidget(self.create_label("SERVER URL"), 0, 0)
        api_grid.addWidget(self.url_input, 0, 1)
        api_grid.addWidget(self.create_label("ACCOUNT EMAIL"), 1, 0)
        api_grid.addWidget(self.email_input, 1, 1)
        api_grid.addWidget(self.create_label("PASSWORD"), 2, 0)
        api_grid.addWidget(self.pass_input, 2, 1)
        
        # --- DATABASE FORM WIDGET ---
        self.db_form_widget = QWidget()
        self.db_form_widget.setMinimumHeight(260)
        db_grid = QGridLayout(self.db_form_widget)
        db_grid.setContentsMargins(0, 0, 0, 0)
        db_grid.setSpacing(12)
        
        db_cfg = DBConfigManager.load_config()
        self.db_host_input = self.create_styled_input("Database Host (e.g. 127.0.0.1)", "fa5s.server", db_cfg.get('host', '127.0.0.1'))
        self.db_port_input = self.create_styled_input("Database Port (e.g. 3306)", "fa5s.plug", str(db_cfg.get('port', 3306)))
        self.db_user_input = self.create_styled_input("Database User", "fa5s.user", db_cfg.get('user', 'root'))
        self.db_pass_input = self.create_styled_input("Database Password", "fa5s.key", db_cfg.get('password', 'root'), password=True)
        self.db_name_input = self.create_styled_input("Database Name", "fa5s.database", db_cfg.get('database', 'demo'))
        
        db_grid.addWidget(self.create_label("DB HOST"), 0, 0)
        db_grid.addWidget(self.db_host_input, 0, 1)
        db_grid.addWidget(self.create_label("DB PORT"), 1, 0)
        db_grid.addWidget(self.db_port_input, 1, 1)
        db_grid.addWidget(self.create_label("DB USER"), 2, 0)
        db_grid.addWidget(self.db_user_input, 2, 1)
        db_grid.addWidget(self.create_label("DB PASSWORD"), 3, 0)
        db_grid.addWidget(self.db_pass_input, 3, 1)
        db_grid.addWidget(self.create_label("DB NAME"), 4, 0)
        db_grid.addWidget(self.db_name_input, 4, 1)
        
        # --- STACKED CONTAINER ---
        self.form_stack = QStackedWidget()
        self.form_stack.addWidget(self.api_form_widget)
        self.form_stack.addWidget(self.db_form_widget)
        card_layout.addWidget(self.form_stack)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        
        self.test_btn = QPushButton("TEST CONNECTION")
        self.test_btn.setFixedHeight(40)
        self.test_btn.setCursor(Qt.PointingHandCursor)
        self.test_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #94a3b8;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: white;
            }
        """)
        self.test_btn.clicked.connect(self.test_connection)
        
        self.link_btn = QPushButton("ESTABLISH CLOUD LINK")
        self.link_btn.setObjectName("PrimaryAction")
        self.link_btn.setFixedHeight(40)
        self.link_btn.setCursor(Qt.PointingHandCursor)
        self.link_btn.setIcon(qta.icon("fa5s.sync", color="#0f172a"))
        self.link_btn.clicked.connect(self.link_cloud)
        
        btn_layout.addWidget(self.test_btn, 1)
        btn_layout.addWidget(self.link_btn, 2)
        
        card_layout.addLayout(btn_layout)
        
        self.footer = QLabel("Changes will immediately trigger a data synchronization.")
        self.footer.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 11px; font-style: italic;")
        self.footer.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.footer)
        
        content_layout.addWidget(card)
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        # Connect signals
        self.cloud_radio.toggled.connect(self.on_mode_changed)
        self.local_api_radio.toggled.connect(self.on_mode_changed)
        self.db_radio.toggled.connect(self.on_mode_changed)
        
        # Initialize selected radio based on stored settings
        is_db_mode = AuthManager.load_setting("db_mode", False)
        is_local_mode = AuthManager.load_setting("is_local_mode", False)
        
        if is_db_mode:
            self.db_radio.setChecked(True)
        elif is_local_mode:
            self.local_api_radio.setChecked(True)
        else:
            self.cloud_radio.setChecked(True)
            
        self.on_mode_changed()

    def create_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 10px; font-weight: bold; letter-spacing: 1px;")
        lbl.setFixedWidth(120)
        return lbl

    def create_styled_input(self, placeholder, icon_name, default_val="", password=False):
        container = QFrame()
        container.setFixedHeight(40)
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
        if password: edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(edit)
        
        container.input_field = edit
        return container

    def link_cloud(self):
        if self.db_radio.isChecked():
            host = self.db_host_input.input_field.text().strip()
            port_str = self.db_port_input.input_field.text().strip()
            user = self.db_user_input.input_field.text().strip()
            password = self.db_pass_input.input_field.text().strip()
            db_name = self.db_name_input.input_field.text().strip()
            
            if not host or not port_str or not user or not db_name:
                QMessageBox.warning(self, "Incomplete Form", "Host, Port, User, and Database Name are required.")
                return
                
            try:
                port = int(port_str)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Database Port must be an integer.")
                return
                
            db_cfg = {
                "host": host,
                "port": port,
                "user": user,
                "password": password,
                "database": db_name
            }
            
            self.link_btn.setEnabled(False)
            self.link_btn.setText("SAVING...")
            
            DBConfigManager.save_config(db_cfg)
            AuthManager.save_setting("db_mode", True)
            AuthManager.save_setting("is_local_mode", False)
            
            from config.config import refresh_api_config
            refresh_api_config()
            
            self.backend.set_local_db_mode()
            for q in self.backend.task_queues:
                if q: q.put({'type': 'update_db_mode'})
                
            QMessageBox.information(self, "Database Configuration", "Direct Local Database Mode enabled successfully! Data will be read/written directly to your local MySQL database.")
            self.link_btn.setEnabled(True)
            self.link_btn.setText("SAVE DB CONFIGURATION")
            
        else:
            url = self.url_input.input_field.text().strip()
            email = self.email_input.input_field.text().strip()
            password = self.pass_input.input_field.text().strip()
            
            if not url or not email or not password:
                QMessageBox.warning(self, "Incomplete Form", "All fields are required for synchronization.")
                return

            url = url.replace(' ', '')
            url = url.replace('http//', 'http://').replace('https//', 'https://')
            if not url.startswith('http://') and not url.startswith('https://'):
                url = 'http://' + url
            url = url.rstrip('/')
            
            self.url_input.input_field.setText(url)
            
            is_local = self.local_api_radio.isChecked()
            AuthManager.save_setting("is_local_mode", is_local)
            AuthManager.save_setting("db_mode", False)
            AuthManager.save_credentials(email, password, url)

            from config.config import refresh_api_config
            refresh_api_config()

            self.link_btn.setEnabled(False)
            self.link_btn.setText("AUTHENTICATING...")
            
            import threading
            def run_link():
                success, msg = self.backend.reconnect_cloud(email, password, url)
                QMetaObject.invokeMethod(self, "on_link_result", Qt.QueuedConnection, 
                                         Q_ARG(bool, success), Q_ARG(str, msg))

            threading.Thread(target=run_link, daemon=True).start()

    def test_connection(self):
        if self.db_radio.isChecked():
            host = self.db_host_input.input_field.text().strip()
            port_str = self.db_port_input.input_field.text().strip()
            user = self.db_user_input.input_field.text().strip()
            password = self.db_pass_input.input_field.text().strip()
            db_name = self.db_name_input.input_field.text().strip()
            
            if not host or not port_str or not user or not db_name:
                QMessageBox.warning(self, "Incomplete Form", "Host, Port, User, and Database Name are required to test connection.")
                return
                
            try:
                port = int(port_str)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Database Port must be an integer.")
                return
                
            self.test_btn.setEnabled(False)
            self.test_btn.setText("TESTING...")
            
            import threading
            def run_db_test():
                try:
                    import mysql.connector
                    conn = mysql.connector.connect(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        connect_timeout=3
                    )
                    cursor = conn.cursor()
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    conn = mysql.connector.connect(
                        host=host,
                        port=port,
                        user=user,
                        password=password,
                        database=db_name,
                        connect_timeout=3
                    )
                    conn.close()
                    
                    success = True
                    msg = "Connected! Local database is reachable."
                except Exception as e:
                    success = False
                    msg = f"Failed to connect to MySQL database:\n{e}"
                
                QMetaObject.invokeMethod(self, "on_test_result", Qt.QueuedConnection, 
                                         Q_ARG(bool, success), Q_ARG(str, msg))

            threading.Thread(target=run_db_test, daemon=True).start()
            
        else:
            url = self.url_input.input_field.text().strip()
            if not url: return

            if not url.startswith('http'): url = 'http://' + url
            url = url.rstrip('/')

            self.test_btn.setEnabled(False)
            self.test_btn.setText("TESTING...")
            
            import requests
            import threading
            def run_test():
                try:
                    resp = requests.get(f"{url}/api/local/status", timeout=3)
                    success = resp.status_code == 200
                    msg = "Connected! Server is reachable." if success else f"Server found but returned {resp.status_code}"
                except Exception as e:
                    try:
                        resp = requests.get(url, timeout=3)
                        success = resp.status_code < 500
                        msg = "Server reachable."
                    except:
                        success = False
                        msg = "Could not reach server. Check IP and Port."
                
                QMetaObject.invokeMethod(self, "on_test_result", Qt.QueuedConnection, 
                                         Q_ARG(bool, success), Q_ARG(str, msg))

            threading.Thread(target=run_test, daemon=True).start()

    @Slot(bool, str)
    def on_test_result(self, success, msg):
        self.test_btn.setEnabled(True)
        self.test_btn.setText("TEST CONNECTION")
        if success:
            QMessageBox.information(self, "Connection Test", msg)
        else:
            QMessageBox.warning(self, "Connection Test", msg)

    @Slot(bool, str)
    def on_link_result(self, success, msg):
        if success:
            QMessageBox.information(self, "Cloud Connected", "Infrastructure linked successfully! Data sync in progress.")
        else:
            QMessageBox.critical(self, "Link Failed", f"Could not establish cloud link.\n\nError: {msg}")
            
        self.link_btn.setEnabled(True)
        if self.local_api_radio.isChecked():
            self.link_btn.setText("ESTABLISH LOCAL API LINK")
        else:
            self.link_btn.setText("ESTABLISH CLOUD LINK")

    def on_mode_changed(self):
        is_cloud = self.cloud_radio.isChecked()
        is_local = self.local_api_radio.isChecked()
        is_db = self.db_radio.isChecked()
        
        AuthManager.save_setting("db_mode", is_db)
        AuthManager.save_setting("is_local_mode", is_local)
        
        self.cloud_radio.setStyleSheet(f"color: {'white' if is_cloud else '#94a3b8'}; font-weight: bold; font-size: 11px;")
        self.local_api_radio.setStyleSheet(f"color: {'white' if is_local else '#94a3b8'}; font-weight: bold; font-size: 11px;")
        self.db_radio.setStyleSheet(f"color: {'white' if is_db else '#94a3b8'}; font-weight: bold; font-size: 11px;")
        
        if is_db:
            self.form_stack.setCurrentWidget(self.db_form_widget)
            self.link_btn.setText("SAVE DB CONFIGURATION")
            self.link_btn.setIcon(qta.icon("fa5s.save", color="#0f172a"))
            self.footer.setText("Saves parameters to config/db_config.json and runs offline direct database mode.")
        else:
            self.form_stack.setCurrentWidget(self.api_form_widget)
            if is_cloud:
                self.link_btn.setText("ESTABLISH CLOUD LINK")
                self.footer.setText("Changes will immediately trigger a data synchronization.")
                last_cloud = AuthManager.load_setting("last_cloud_url", "https://api.visionattendance.com")
                self.url_input.input_field.setText(last_cloud)
            else:
                self.link_btn.setText("ESTABLISH LOCAL API LINK")
                self.footer.setText("Changes will immediately trigger a data synchronization with local server.")
                last_local = AuthManager.load_setting("last_local_url", "http://127.0.0.1:8080")
                self.url_input.input_field.setText(last_local)
                
            self.link_btn.setIcon(qta.icon("fa5s.sync", color="#0f172a"))
            
        from config.config import refresh_api_config
        refresh_api_config()
