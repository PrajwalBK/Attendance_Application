from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QListWidgetItem, QLabel, QPushButton, QFrame, QGridLayout, QProgressBar)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
import qtawesome as qta
from ui.theme_manager import ThemeManager
from datetime import datetime
import os

class RecordsPage(QWidget):
    def __init__(self, backend_controller):
        super().__init__()
        self.backend = backend_controller
        self.setup_ui()
        
        # Load history on startup
        from PySide6.QtCore import QTimer
        QTimer.singleShot(500, self.load_historical_data)

    def load_historical_data(self):
        """Fetches last 100 logs from DB or API on startup."""
        try:
            from config.config import get_config
            cfg = get_config()
            
            db = None
            if not cfg.get('use_api', True):
                from database.database import DatabaseManager
                db = DatabaseManager()
            elif hasattr(self.backend, 'api_client') and self.backend.api_client:
                db = self.backend.api_client
            elif hasattr(self.backend, 'db') and self.backend.db:
                db = self.backend.db
            else:
                # Guard: Don't attempt API calls if we haven't authenticated yet
                token = getattr(self.backend, 'auth_token', None)
                if not token:
                    return
                
                email = getattr(self.backend, 'auth_email', None)
                password = getattr(self.backend, 'auth_pass', None)
                
                from core.api_client import APIClient
                db = APIClient(
                    cfg.get('api_base_url'),
                    api_user=email,
                    api_password=password,
                    token=token
                )
            
            if db:
                logs = db.get_recent_logs()
                if logs:
                    for log in reversed(logs):
                        pid, name, date, time_str, event_type = log
                        data = {
                            'type': 'match',
                            'id': pid,
                            'name': name,
                            'timestamp': time_str,
                            'event_type': (event_type or 'IN').upper()
                        }
                        self.add_detection(data)
        except Exception as e:
            print(f"Error loading historical logs: {e}")
        self.load_data()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(25)
        
        # ── Metric counters (attendance focus) ─────────────────────────
        self._counts = {"present": 0, "absent": 0, "logins": 0, "logouts": 0}
        self._present_set = set()
        self._total_employees = 0
        self._metric_labels = {}
        
        # 2. Filter Bar
        filter_bar = QHBoxLayout()
        title = QLabel("Real-Time Detection Logs")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        filter_bar.addWidget(title)
        
        self.count_lbl = QLabel("0")
        self.count_lbl.setStyleSheet(f"background: rgba(21, 155, 146, 0.1); color: {ThemeManager.COLORS['primary']}; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;")
        filter_bar.addWidget(self.count_lbl)
        
        filter_bar.addStretch()
        
        # Filter Buttons
        self.filter_buttons = {}
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        for text in ["Login", "Logout", "Unknown"]:
            btn = QPushButton(text)
            btn.setObjectName("FilterButton")
            btn.setCheckable(True)
            if text == "Login": btn.setChecked(True)
            btn.clicked.connect(lambda checked, t=text: self.filter_logs(t))
            btn_layout.addWidget(btn)
            self.filter_buttons[text] = btn
        filter_bar.addLayout(btn_layout)
        
        main_layout.addLayout(filter_bar)
        
        # 3. Detection Log Table (List-based for custom rows)
        # Header for the "table"
        table_header = QFrame()
        table_header.setObjectName("TableHeader")
        h_layout = QHBoxLayout(table_header)
        h_layout.setContentsMargins(15, 8, 15, 8)
        
        headers = [("TIME", 1), ("NAME", 3), ("CAMERA", 2), ("CONFIDENCE", 2), ("STATUS", 1)]
        for h_text, stretch in headers:
            h_lbl = QLabel(h_text)
            h_layout.addWidget(h_lbl, stretch)
        main_layout.addWidget(table_header)
        
        self.log_list = QListWidget()
        self.log_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        main_layout.addWidget(self.log_list)

    def load_data(self):
        # Empty on startup — populated by real backend detections only
        self.log_list.clear()

    def add_detection(self, data):
        """Called by MainWindow when backend emits detection_occurred."""
        evt = str(data.get('event_type', 'IN')).upper()
        name = str(data.get('name') or 'Unknown')
        worker_id = data.get('worker', 0)
        confidence = data.get('sim') or data.get('confidence') or 0.0
        timestamp = data.get('timestamp', '--:--:--')
        cam_ch = f"Cam {worker_id + 1}"
        department = data.get('dept') or data.get('department') or "---"
        # Map to display status
        pid = data.get('id')
        
        if name.upper() == 'UNKNOWN':
            status = 'Alert'
            # [USER REQUEST] Hide alerts from detection logs by default
            return 
            
        elif evt in ('IN', 'LOGIN'):
            status = 'Login'
            self._counts['logins'] += 1
            if pid: self._present_set.add(pid)
        else:
            status = 'Logout'
            self._counts['logouts'] += 1
            # If they logout, they are still "Present" in terms of having been seen today?
            # Or does Present mean "Currently In"? 
            # Usually Attendance Present = Seen Today.
            if pid: self._present_set.add(pid)

        # Update metrics
        self._counts['present'] = len(self._present_set)
        self._counts['absent'] = max(0, self._total_employees - self._counts['present'])

        # Safety check: Update labels only if they exist
        if 'present' in self._metric_labels: self._metric_labels['present'].setText(str(self._counts['present']))
        if 'absent' in self._metric_labels: self._metric_labels['absent'].setText(str(self._counts['absent']))
        if 'logins' in self._metric_labels: self._metric_labels['logins'].setText(str(self._counts['logins']))
        if 'logouts' in self._metric_labels: self._metric_labels['logouts'].setText(str(self._counts['logouts']))
        
        # Bottom log counter
        total_logs = self._counts['logins'] + self._counts['logouts']
        self.count_lbl.setText(str(total_logs))

        row_data = {
            'time': timestamp,
            'name': name,
            'dept': department,
            'cam': cam_ch,
            'conf': float(confidence),
            'status': status
        }
        
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 65))
        item.setData(Qt.UserRole, status) # Store status for filtering
        
        self.log_list.insertItem(0, item)
        self.log_list.setItemWidget(item, self.create_log_row(row_data))
        
        # Apply current filter to new item
        active_filter = "All"
        for text, btn in self.filter_buttons.items():
            if btn.isChecked():
                active_filter = text
                break
        self.filter_logs(active_filter)

        if self.log_list.count() > 200:
            self.log_list.takeItem(self.log_list.count() - 1)

    def filter_logs(self, filter_type):
        """Filters the list based on status."""
        # Update button states
        for text, btn in self.filter_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(text == filter_type)
            btn.blockSignals(False)

        # Show/Hide items
        for i in range(self.log_list.count()):
            item = self.log_list.item(i)
            status = item.data(Qt.UserRole)
            if filter_type == "All":
                item.setHidden(False)
            else:
                item.setHidden(status != filter_type)

    def update_summary_from_stats(self, stats):
        """Updates summary cards from backend stats signal."""
        if 'total' in stats:
            self._total_employees = stats['total']
            self._counts['absent'] = max(0, self._total_employees - self._counts['present'])
            if 'absent' in self._metric_labels:
                self._metric_labels['absent'].setText(str(self._counts['absent']))

    def create_log_row(self, data):
        widget = QFrame()
        widget.setStyleSheet("border-bottom: 1px solid #1a253a;")
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(15, 10, 15, 10)
        
        # 1. Time
        time_lbl = QLabel(data['time'])
        time_lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 12px;")
        layout.addWidget(time_lbl, 1)
        
        # 2. Name & Avatar
        name_container = QWidget()
        name_layout = QHBoxLayout(name_container)
        name_layout.setContentsMargins(0, 0, 0, 0)
        
        avatar = QLabel()
        avatar.setFixedSize(32, 32)
        initials = data['name'][0] if data['name'] != "Unknown" else "?"
        bg_color = ThemeManager.COLORS['primary'] if data['status'] == "Login" else "#475569"
        if data['status'] == "Alert": bg_color = ThemeManager.COLORS['error']
        avatar.setText(initials)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"background-color: {bg_color}; color: white; border-radius: 16px; font-weight: bold; font-size: 11px;")
        name_layout.addWidget(avatar)
        
        text_v_layout = QVBoxLayout()
        text_v_layout.setSpacing(0)
        name_lbl = QLabel(data['name'])
        name_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        dept_lbl = QLabel(data['dept'])
        dept_lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 11px;")
        text_v_layout.addWidget(name_lbl)
        text_v_layout.addWidget(dept_lbl)
        name_layout.addLayout(text_v_layout)
        layout.addWidget(name_container, 3)
        
        # 3. Camera
        cam_lbl = QLabel(data['cam'])
        cam_lbl.setStyleSheet(f"color: {ThemeManager.COLORS['text_dim']}; font-size: 12px;")
        layout.addWidget(cam_lbl, 2)
        
        # 4. Confidence
        conf_container = QWidget()
        conf_layout = QVBoxLayout(conf_container)
        conf_layout.setContentsMargins(0, 5, 20, 5)
        
        conf_percent = int(data['conf'] * 100)
        conf_text = QLabel(f"{conf_percent}%")
        conf_text.setStyleSheet("color: white; font-size: 10px; font-weight: bold;")
        conf_bar = QProgressBar()
        conf_bar.setValue(conf_percent)
        conf_bar.setTextVisible(False)
        conf_bar.setFixedHeight(4)
        
        conf_layout.addWidget(conf_text)
        conf_layout.addWidget(conf_bar)
        layout.addWidget(conf_container, 2)
        
        # 5. Status Pill
        status_pill = QLabel(data['status'].upper())
        status_pill.setAlignment(Qt.AlignCenter)
        status_pill.setFixedSize(85, 26)
        
        is_login = data['status'] == "Login"
        p_color = ThemeManager.COLORS['success'] if is_login else ThemeManager.COLORS['warning']
        bg_opacity = 0.1
        
        status_pill.setStyleSheet(f"""
            color: {p_color}; 
            background: rgba({34 if is_login else 245}, 
                           {197 if is_login else 158}, 
                           {94 if is_login else 11}, {bg_opacity}); 
            border: 1px solid {p_color}; 
            border-radius: 4px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        
        status_wrapper = QWidget()
        status_layout = QHBoxLayout(status_wrapper)
        status_layout.setContentsMargins(0,0,0,0)
        status_layout.setAlignment(Qt.AlignLeft)
        status_layout.addWidget(status_pill)
        
        layout.addWidget(status_wrapper, 1)
        
        return widget
