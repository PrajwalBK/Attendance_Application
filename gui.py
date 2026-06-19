import os
# Configure thread limits BEFORE any scientific library (numpy, opencv, onnxruntime) is imported
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["ORT_ARENA_EXTEND_STRATEGY"] = "kSameAsRequested"

# AGGRESSIVE RTSP TIMEOUT: 5 seconds (in microseconds)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000|stimeout;5000000"

import sys
import threading
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,
                             QFrame, QDialog, QGridLayout, QSpacerItem, QSizePolicy, QComboBox)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QPixmap

import qtawesome as qta
from ui.theme_manager import ThemeManager
from core.gui_workers import BackendController, MAX_CAMS
from ui.camera_page import CameraPage
from ui.registration_page import RegistrationPage
from ui.settings_page import SettingsPage
from ui.login_dialog import LoginDialog
from ui.cctv_setup import CCTVSetupWizard
from ui.cloud_setup import CloudSetupPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Vision Attendance System")
        self.setMinimumSize(800, 600)

        self.backend = BackendController()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.setup_sidebar(root)
        self.setup_content_area(root)
        self.init_pages()
        self.switch_page(0)

        # Handle auto-start behavior
        from config.cam_config_manager import CamConfigManager
        from core.startup_manager import StartupManager
        self.cam_config = CamConfigManager.load_config()
        
        # We assume auto_start_on_boot is toggled via settings (defaulting to True since the user requested it now)
        # Overriding default for this user's strict requirement, or assuming we enable it explicitly:
        if self.cam_config.get("auto_start_on_boot", True):  # Defaulting True here since user requested it
            self.cam_config["auto_start_on_boot"] = True
            CamConfigManager.save_config(self.cam_config)
            StartupManager.enable_auto_startup()
            # Start faster (500ms instead of 1500ms)
            QTimer.singleShot(500, self._auto_start_sequence)
        else:
            StartupManager.disable_auto_startup()

    def _auto_start_sequence(self):
        # Sync faces in background first so it doesn't block UI during slow boot
        threading.Thread(target=self.backend.sync_remote_faces, daemon=True).start()
        
        # 1. Start Cameras (Instant)
        if not self.backend.are_cameras_active:
            self.toggle_cameras()
            
        # 2. Wait 3 seconds for rtsp handshakes, then start AI Workers
        # Reduced from 5s to 3s for "fast start" requirement while keeping stability
        QTimer.singleShot(3000, self.backend.start_detection)

    def _update_queue_status(self):
        try:
            from config.config import get_config, BASE_DIR
            config = get_config()
            
            # Use absolute path to avoid "0" status in packaged EXE
            rel_dir = config.get('temp_recordings_dir', 'data/temp_recordings')
            recording_dir = os.path.join(BASE_DIR, rel_dir) if not os.path.isabs(rel_dir) else rel_dir
            pending_dir = os.path.join(BASE_DIR, "data", "pending_snapshots")

            count = 0
            if os.path.exists(recording_dir):
                files = [f for f in os.listdir(recording_dir) if f.endswith('.avi') or f.endswith('.mp4') or f.endswith('.tmp')]
                # Count files that aren't currently being written (modified > 5s ago)
                for f in files:
                    try:
                        if time.time() - os.path.getmtime(os.path.join(recording_dir, f)) > 5:
                            count += 1
                    except: pass
            
            if os.path.exists(pending_dir):
                files = [f for f in os.listdir(pending_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count += len(files)
            
            # Standardize on Sidebar Status Grid if available
            if hasattr(self, '_status_vals') and "pending" in self._status_vals:
                dot, val_lbl = self._status_vals["pending"]
                val_lbl.setText(str(count))
                if count > 0:
                    val_lbl.setStyleSheet(f"color: #3b82f6; font-size: 15px; font-weight: 800;")
                    dot.setStyleSheet(f"color: #3b82f6; font-size: 14px;")
                else:
                    val_lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 15px; font-weight: 800;")
                    dot.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 14px;")
            
            # Fallback to stats widgets if they exist
            if hasattr(self, '_stats_widgets') and "pending" in self._stats_widgets:
                val_lbl = self._stats_widgets.get("pending", {}).get("value")
                dot = self._stats_widgets.get("pending", {}).get("dot")
                if val_lbl and dot:
                    val_lbl.setText(str(count))
                    if count > 0:
                        val_lbl.setStyleSheet(f"color: #3b82f6; font-size: 15px; font-weight: 800;")
                        dot.setStyleSheet(f"color: #3b82f6; font-size: 14px;")
                    else:
                        val_lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 15px; font-weight: 800;")
                        dot.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 14px;")
        except:
            pass
# ─────────────────────────────────────────────────────────────────────────
    def setup_sidebar(self, root_layout):
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setMinimumWidth(200)
        sidebar.setMaximumWidth(240)

        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        # ── Logo ──────────────────────────────────────────────────────────
        logo_frame = QWidget()
        logo_frame.setFixedHeight(120)
        logo_frame.setStyleSheet(f"border-bottom: 1px solid #1a2540;")
        lf = QHBoxLayout(logo_frame)
        lf.setContentsMargins(24, 0, 24, 0)
        lf.setSpacing(15)
        lf.setAlignment(Qt.AlignCenter)

        logo_img = QLabel()
        from config.config import BASE_DIR
        logo_path = os.path.join(BASE_DIR, "ui", "WhatsApp Image 2026-02-17 at 19.51.32.jpeg")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(190, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_img.setPixmap(pix)
        logo_img.setAlignment(Qt.AlignCenter)
        logo_img.setStyleSheet("background: transparent; border: none;")
        lf.addWidget(logo_img)
        sl.addWidget(logo_frame)

        # ── Navigation ────────────────────────────────────────────────────
        nav_items = [
            ("Dashboard",      "fa5s.th-large"),
            ("Detection Logs", "fa5s.clipboard-list"),
            ("CCTV Setup",     "fa5s.cog"),
            ("Connection Setup", "fa5s.network-wired"),
        ]

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 20, 12, 0)
        nav_layout.setSpacing(8)

        self.nav_buttons = []
        for i, (text, icon_name) in enumerate(nav_items):
            btn = QPushButton(f"   {text}")
            btn.setObjectName("NavButton")
            btn.setProperty("icon_name", icon_name)
            btn.setIcon(qta.icon(icon_name, color=ThemeManager.COLORS['text_dim']))
            btn.setIconSize(QSize(16, 16))
            btn.setCheckable(True)
            btn.setFixedHeight(42)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        nav_layout.addStretch()
        sl.addWidget(nav_container, 1)

        # ── Status Bar ────────────────────────────────────────────────────
        status_frame = QFrame()
        status_frame.setStyleSheet("border-top: 1px solid #1a2540;")
        status_frame.setMaximumHeight(240) # Large for high visibility but allow shrinking on small screens
        sf = QGridLayout(status_frame)
        sf.setContentsMargins(22, 20, 22, 20)
        sf.setVerticalSpacing(14)

        self._status_vals = {}
        statuses = [
            ("system",   "System",     "Idle",      ThemeManager.COLORS['text_dim']),
            ("cameras",  "Cameras",    "Inactive",  ThemeManager.COLORS['text_dim']),
            ("workers",  "AI Workers", "Inactive",  ThemeManager.COLORS['text_dim']),
            ("cloud",    "Cloud",      "---",       ThemeManager.COLORS['text_dim']),
            ("pending",  "Pending",    "0",         ThemeManager.COLORS['text_dim']),
        ]
        for row, (key, label, val, color) in enumerate(statuses):
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 15px;")
            val_lbl = QLabel(val)
            # Fix: Start Cloud status as 'Disconnected' if no token yet
            if key == "cloud":
                val = "---"
                color = ThemeManager.COLORS['text_dim']
            val_lbl.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: 800;")
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            sf.addWidget(dot,     row, 0)
            sf.addWidget(lbl,     row, 1)
            sf.addWidget(val_lbl, row, 2)
            self._status_vals[key] = (dot, val_lbl)

        sl.addWidget(status_frame)
        root_layout.addWidget(sidebar)

        # Connect backend status signal to sidebar updater
        self.backend.worker_signals.status_updated.connect(self._update_sidebar_status)

    def _update_sidebar_status(self, text):
        t = text.upper()
        green  = f"color: {ThemeManager.COLORS['success']}; font-size: 14px;"
        dim    = f"color: {ThemeManager.COLORS['text_dim']}; font-size: 14px;"
        orange = f"color: {ThemeManager.COLORS['warning']}; font-size: 14px;"
        
        green_text = f"color: {ThemeManager.COLORS['success']}; font-size: 15px; font-weight: 800;"
        dim_text   = f"color: {ThemeManager.COLORS['text_dim']}; font-size: 15px; font-weight: 800;"
        orange_text = f"color: {ThemeManager.COLORS['warning']}; font-size: 15px; font-weight: 800;"

        def _set(key, dot_style, val_text, val_color):
            dot, lbl = self._status_vals[key]
            dot.setStyleSheet(dot_style)
            lbl.setText(val_text)
            lbl.setStyleSheet(val_color)

        if "CONNECTING CAMERAS" in t or "CAMERAS STARTING" in t:
            _set("cameras", orange, "Starting…", orange_text)
        elif "CAMERAS ONLINE" in t or "CAMERAS READY" in t or "CAMERAS ACTIVE" in t:
            _set("cameras", green, "Active", green_text)
        elif "CAMERAS OFFLINE" in t:
            _set("cameras", dim, "Inactive", dim_text)

        if "STARTING AI" in t:
            _set("system", orange, "Starting…", orange_text)
            _set("workers", orange, "Starting…", orange_text)
        elif "DETECTION ACTIVE" in t:
            _set("system",  green, "Running", green_text)
            _set("workers", green, "Running", green_text)
        elif "DETECTION STOPPED" in t:
            _set("system",  dim, "Idle",     dim_text)
            _set("workers", dim, "Inactive", dim_text)

        if "SYNC COMPLETE" in t or "CLOUD CONNECTED" in t:
            _set("cloud", green, "Synced", green_text)
        elif "SYNCING" in t:
            _set("cloud", orange, "Syncing…", orange_text)

    # ─────────────────────────────────────────────────────────────────────────
    def setup_content_area(self, root_layout):
        content_main = QWidget()
        cm = QVBoxLayout(content_main)
        cm.setContentsMargins(0, 0, 0, 0)
        cm.setSpacing(0)

        # ── Header Bar ────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(64)
        header.setStyleSheet(f"background-color: {ThemeManager.COLORS['background']}; border-bottom: 1px solid #1a2540;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        hl.setSpacing(10)

        self.header_title = QLabel("Dashboard")
        self.header_title.setObjectName("HeaderTitle")
        hl.addWidget(self.header_title)
        hl.addStretch()

        # --- Camera Scaling Selector ---
        cam_count_label = QLabel("Active Channels: ")
        cam_count_label.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 11px; font-weight: bold;")
        hl.addWidget(cam_count_label)
        
        self.cam_count_sel = QComboBox()
        self.cam_count_sel.addItems(["2 Cameras", "4 Cameras", "6 Cameras", "8 Cameras"])
        self.cam_count_sel.setMinimumWidth(100)
        self.cam_count_sel.setFixedHeight(30)
        self.cam_count_sel.setStyleSheet(f"""
            QComboBox {{
                background: rgba(255,255,255,0.05);
                border: 1px solid #1a2540;
                border-radius: 6px;
                color: white;
                padding: 0 8px;
                font-size: 11px;
            }}
            QComboBox::drop-down {{ border: none; }}
        """)
        
        # Load current count and set index
        from config.cam_config_manager import CamConfigManager
        config = CamConfigManager.load_config()
        current_count = min(config.get("active_cam_count", 6), 8)
        mapping = {2:0, 4:1, 6:2, 8:3}
        self.cam_count_sel.setCurrentIndex(mapping.get(current_count, 2))
        
        self.cam_count_sel.currentIndexChanged.connect(self._on_cam_count_changed)
        hl.addWidget(self.cam_count_sel)
        
        hl.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Fixed, QSizePolicy.Minimum))

        # Live / Rec Badges — start dimmed (nothing is active yet)
        BADGE_BASE = "border-radius: 12px; font-size: 10px; font-weight: 800; padding: 4px 14px; letter-spacing: 0.6px; border: none;"
        DIMMED = f"color: {ThemeManager.COLORS['text_dim']}; background: rgba(255,255,255,0.04); {BADGE_BASE}"
        
        self.live_badge = QLabel(" ● LIVE ")
        self.live_badge.setStyleSheet(DIMMED)
        hl.addWidget(self.live_badge)

        self.rec_badge = QLabel(" ● REC ")
        self.rec_badge.setStyleSheet(DIMMED)
        hl.addWidget(self.rec_badge)

        # Update Timer for Queue Count
        self.queue_timer = QTimer(self)
        self.queue_timer.timeout.connect(self._update_queue_status)
        self.queue_timer.start(2000) # Every 2 seconds

        # Header Button Base Styles (Bypass qt-material)
        self.START_STYLE = "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #059669); border: none; border-radius: 8px; padding: 6px 16px; font-weight: bold; font-size: 12px; color: white;"
        self.STOP_STYLE = "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ef4444, stop:1 #b91c1c); border: none; border-radius: 8px; padding: 6px 16px; font-weight: bold; font-size: 12px; color: white;"

        # --- Camera Controls ---
        self.cam_btn = QPushButton(" Start Cameras")
        self.cam_btn.setStyleSheet(self.START_STYLE)
        self.cam_btn.setIcon(qta.icon("fa5s.video", color="white"))
        self.cam_btn.setIconSize(QSize(16, 16))
        self.cam_btn.setFixedHeight(38)
        self.cam_btn.setMinimumWidth(120)
        self.cam_btn.setCursor(Qt.PointingHandCursor)
        self.cam_btn.clicked.connect(self.toggle_cameras)
        hl.addWidget(self.cam_btn)

        # --- System Controls ---
        self.sys_btn = QPushButton(" Start AI")
        self.sys_btn.setStyleSheet(self.START_STYLE)
        self.sys_btn.setIcon(qta.icon("fa5s.play", color="white"))
        self.sys_btn.setIconSize(QSize(16, 16))
        self.sys_btn.setFixedHeight(38)
        self.sys_btn.setMinimumWidth(120)
        self.sys_btn.setCursor(Qt.PointingHandCursor)
        self.sys_btn.clicked.connect(self.toggle_system)
        hl.addWidget(self.sys_btn)

        cm.addWidget(header)

        self.content_stack = QStackedWidget()
        cm.addWidget(self.content_stack)
        root_layout.addWidget(content_main)

    # ─────────────────────────────────────────────────────────────────────────
    def init_pages(self):
        self.camera_page = CameraPage(self.backend)
        self.content_stack.addWidget(self.camera_page)     # 0

        from ui.records_page import RecordsPage
        self.records_page = RecordsPage(self.backend)
        self.content_stack.addWidget(self.records_page)    # 2

        # ── Connect System Signals ──────────────────────────────────────
        self.backend.worker_signals.detection_occurred.connect(self.records_page.add_detection)
        self.backend.worker_signals.stats_updated.connect(self.records_page.update_summary_from_stats)

        self.cctv_page = CCTVSetupWizard(self.backend)
        self.cctv_page.setup_finished.connect(self.camera_page.reload_config)
        self.content_stack.addWidget(self.cctv_page)       # 2

        self.cloud_page = CloudSetupPage(self.backend)
        self.content_stack.addWidget(self.cloud_page)      # 3

        # Wire backend status → button states
        self.backend.worker_signals.status_updated.connect(self._sync_header_buttons)

    # ─────────────────────────────────────────────────────────────────────────
    def _sync_header_buttons(self, text):
        t = text.upper()
        BADGE_BASE = "border-radius: 12px; font-size: 10px; font-weight: 800; padding: 4px 14px; letter-spacing: 0.6px; border: none;"
        ACTIVE_CAM_STYLE  = f"color: {ThemeManager.COLORS['success']}; background: rgba(34,197,94,0.18); {BADGE_BASE}"
        ACTIVE_REC_STYLE  = f"color: #ef4444; background: rgba(239,68,68,0.18); {BADGE_BASE}"
        DIMMED            = f"color: {ThemeManager.COLORS['text_dim']}; background: rgba(255,255,255,0.04); {BADGE_BASE}"

        if "CONNECTING" in t or "STARTING" in t or "INITIALIZING" in t:
            if "CAMERA" in t:
                self.cam_btn.setText(" Starting...")
                self.cam_btn.setIcon(qta.icon("fa5s.sync", color="white", animation=qta.Spin(self.cam_btn)))
                self.cam_btn.setEnabled(False)
            if "AI" in t:
                self.sys_btn.setText(" Starting AI...")
                self.sys_btn.setIcon(qta.icon("fa5s.sync", color="white", animation=qta.Spin(self.sys_btn)))
                self.sys_btn.setEnabled(False)

        if "CAMERAS ONLINE" in t:
            self.live_badge.setStyleSheet(ACTIVE_CAM_STYLE)
            self.cam_btn.setText(" Stop Cameras")
            self.cam_btn.setStyleSheet(self.STOP_STYLE)
            self.cam_btn.setIcon(qta.icon("fa5s.video-slash", color="white"))
            self.cam_btn.setEnabled(True)

        elif "CAMERAS OFFLINE" in t:
            self.live_badge.setStyleSheet(DIMMED)
            self.cam_btn.setText(" Start Cameras")
            self.cam_btn.setStyleSheet(self.START_STYLE)
            self.cam_btn.setIcon(qta.icon("fa5s.video", color="white"))
            self.cam_btn.setEnabled(True)

        if "DETECTION ACTIVE" in t:
            self.rec_badge.setStyleSheet(ACTIVE_REC_STYLE)
            self.sys_btn.setText(" Stop AI")
            self.sys_btn.setStyleSheet(self.STOP_STYLE)
            self.sys_btn.setIcon(qta.icon("fa5s.power-off", color="white"))
            self.sys_btn.setEnabled(True)

        elif "DETECTION STOPPED" in t:
            self.rec_badge.setStyleSheet(DIMMED)
            self.sys_btn.setText(" Start AI")
            self.sys_btn.setStyleSheet(self.START_STYLE)
            self.sys_btn.setIcon(qta.icon("fa5s.play", color="white"))
            self.sys_btn.setEnabled(True)

    def _on_cam_count_changed(self, index):
        counts = [2, 4, 6, 8]
        count = counts[index]
        
        # 1. Update Backend
        self.backend.set_active_count(count)
        
        # 2. Update UI Grid
        self.camera_page.set_active_count(count)
        
        # [ROBUSTNESS] If cameras are already active, we must re-initialize the new slots
        if self.backend.are_cameras_active:
             self.toggle_cameras() # Stop
             QTimer.singleShot(500, self.toggle_cameras) # Start again with new slots

    def toggle_cameras(self):
        if self.backend.are_cameras_active:
            self.backend.stop_cameras()
        else:
            try:
                widgets = self.camera_page.cam_widgets
                active_count = self.backend.active_cam_count
                srcs = [self._parse_source(widgets[i].source_selector.currentText()) if i < len(widgets) else None for i in range(active_count)]
                # Set roles from UI before starting
                for i in range(active_count):
                    w = widgets[i]
                    role = ["entrance", "exit", "monitor"][w.role_selector.currentIndex()]
                    self.backend.set_cam_role(i, role)
                self.backend.start_cameras(*srcs)
                
                # Save config WITHOUT destroying existing IP data
                from config.cam_config_manager import CamConfigManager
                config = CamConfigManager.load_config()
                existing_cams = config.get("cams", [])
                
                new_cams = []
                for i in range(MAX_CAMS):
                    w = widgets[i]
                    cam_ip = existing_cams[i].get("ip") if i < len(existing_cams) else None
                    new_cams.append({
                        "name": f"CAMERA {i+1}",
                        "source": w.source_selector.currentText(),
                        "role": ["entrance", "exit", "monitor"][w.role_selector.currentIndex()],
                        "ip": cam_ip
                    })
                config["cams"] = new_cams
                CamConfigManager.save_config(config)
            except Exception as e:
                print(f"Error starting cameras: {e}")

    def toggle_system(self):
        if self.backend.is_detection_running:
            self.backend.stop_detection()
        else:
            self.backend.start_detection()

    # Duplicate _update_queue_status method definition removed (unified above)

    def _parse_source(self, text):
        if "Ch" in text:
            return text.split(" ")[1]
        elif "Webcam" in text:
            return int(text.split(" ")[1])
        return text

    def switch_page(self, index):
        self.content_stack.setCurrentIndex(index)
        titles = ["Dashboard — CCTV Monitor", "Detection Logs", "CCTV Configuration", "Connection Configuration"]
        if index < len(titles):
            self.header_title.setText(titles[index])
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
            icon_name = btn.property("icon_name")
            btn.setIcon(qta.icon(icon_name, color="white" if i == index else ThemeManager.COLORS['text_dim']))

    def closeEvent(self, event):
        # Stop timers to prevent further callbacks during shutdown
        try:
            self.queue_timer.stop()
        except: pass
        try:
            self.camera_page.timer.stop()
        except: pass

        self.backend.stop_system()
        self.backend.wait(1000)  # Wait max 1 second for graceful shutdown
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    import signal
    multiprocessing.freeze_support()

    # Custom signal handler for Ctrl+C
    def sigint_handler(*args):
        print("\n[GUI] Received SIGINT (Ctrl+C). Closing application...")
        QApplication.closeAllWindows()
        QApplication.quit()

    signal.signal(signal.SIGINT, sigint_handler)

    app = QApplication(sys.argv)
    ThemeManager.apply_theme(app)

    # Try Auto-Login first for unattended restarts
    from config.auth_manager import AuthManager
    from config.config import get_config
    conf = get_config()
    
    use_api = conf.get('use_api', True)
    auth_success = False
    email, password, token = None, None, None

    if not use_api:
        # Bypassing login dialog for local database mode
        auth_success = True
    else:
        saved_user, saved_pass, saved_url = AuthManager.load_credentials()
        if saved_user and saved_pass:
            print(f"[AUTO-LOGIN] Attempting for {saved_user}...")
            email, password = saved_user, saved_pass
            auth_success = True
        else:
            login = LoginDialog()
            if login.exec() == QDialog.Accepted:
                email, password, token = login.email, login.password, login.token
                auth_success = True

    if auth_success:
        window = MainWindow()
        if use_api:
            # Use the saved URL from the disk if this was an auto-login
            target_url = saved_url if saved_user else None
            window.backend.set_auth_credentials(email, password, token, url=target_url)
        else:
            window.backend.set_local_db_mode()
        window.backend.start()
        window.show()
        sys.exit(app.exec())
    else:
        sys.exit(0)
