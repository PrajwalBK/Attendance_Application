import cv2
from datetime import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QComboBox, QGridLayout, QSizePolicy)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction
import qtawesome as qta
from ui.theme_manager import ThemeManager


class ROILabel(QLabel):
    def __init__(self, cam_index, backend_controller, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.backend = backend_controller
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        # ROI drawing disabled
        pass
            
    def mouseMoveEvent(self, event):
        # ROI drawing disabled
        pass
            
    def mouseReleaseEvent(self, event):
        # ROI drawing disabled
        pass
            
    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #0f172a;
                color: white;
                border: 1px solid #1e293b;
                border-radius: 6px;
            }
            QMenu::item {
                padding: 6px 20px;
                color: white;
            }
            QMenu::item:selected {
                background-color: #1e293b;
            }
            QMenu::separator {
                height: 1px;
                background-color: #1e293b;
                margin: 4px 0px;
            }
        """)
        
        # Get current rules dict
        current_rules = {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}
        if hasattr(self.backend, 'cam_direction_rules') and self.cam_index < len(self.backend.cam_direction_rules):
            current_rules = self.backend.cam_direction_rules[self.cam_index]
            
        directions = [
            ("Upward Motion (▲)", "up"),
            ("Downward Motion (▼)", "down"),
            ("Leftward Motion (◀)", "left"),
            ("Rightward Motion (▶)", "right")
        ]
        
        for dir_name, key in directions:
            sub = menu.addMenu(dir_name)
            sub.setStyleSheet(menu.styleSheet())
            cur_val = current_rules.get(key, "ignore")
            
            for option in ["in", "out", "ignore"]:
                act = QAction(option.upper(), self)
                act.setCheckable(True)
                if option == cur_val:
                    act.setChecked(True)
                act.triggered.connect(lambda checked, k=key, o=option: self.set_direction_rule(k, o))
                sub.addAction(act)
                
        menu.exec(event.globalPos())
        
    def set_direction_rule(self, key, option):
        if hasattr(self.backend, 'cam_direction_rules') and self.cam_index < len(self.backend.cam_direction_rules):
            self.backend.cam_direction_rules[self.cam_index][key] = option
            print(f"[BACKEND] Set CAM_{self.cam_index+1} Direction Rule '{key}' -> {option.upper()}")
            
            try:
                from config.cam_config_manager import CamConfigManager
                conf = CamConfigManager.load_config()
                if self.cam_index < len(conf.get('cams', [])):
                    conf['cams'][self.cam_index]['direction_rules'] = self.backend.cam_direction_rules[self.cam_index]
                    CamConfigManager.save_config(conf)
            except Exception as e:
                print(f"[BACKEND] Error saving direction rules: {e}")

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing and self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(QColor(76, 175, 80), 2, Qt.DashLine)
            painter.setPen(pen)
            x = min(self.start_point.x(), self.end_point.x())
            y = min(self.start_point.y(), self.end_point.y())
            w = abs(self.start_point.x() - self.end_point.x())
            h = abs(self.start_point.y() - self.end_point.y())
            painter.drawRect(x, y, w, h)
            

class CameraPage(QWidget):
    def __init__(self, backend_controller):
        super().__init__()
        self.backend = backend_controller
        self.setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)  # ~33 FPS zero-lag real-time preview rendering

        self.connecting_status = False
        self.backend.worker_signals.status_updated.connect(self.on_status_updated)

    def on_status_updated(self, text):
        t = text.upper()
        # [FIX] Detect SWAPPING or CONNECTING to show pulse in UI
        if ("CONNECTING" in t or "STARTING" in t or "SWAPPING" in t) and ("CAMERA" in t or "CAM_" in t):
            self.connecting_status = True
        elif "ONLINE" in t or "READY" in t or "STOPPED" in t or "OFFLINE" in t:
            self.connecting_status = False

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        self.main_layout.addLayout(self.grid)

        self.cam_widgets = []
        from config.cam_config_manager import CamConfigManager
        config = CamConfigManager.load_config()
        cams = config.get("cams", [])
        
        # Max capacity is 8
        MAX_CAMS = 8
        for i in range(MAX_CAMS):
            c = cams[i] if i < len(cams) else {"name": f"CAMERA {i+1}", "source": f"Camera {i+1}", "role": "monitor"}
            widget = self.create_camera_widget(i, c.get("name", f"CAMERA {i+1}"), c.get("source", f"Camera {i+1}"), c.get("role", "monitor"))
            widget.hide() # Hidden by default, shown by set_active_count
            self.cam_widgets.append(widget)

        # Initial layout based on config
        active_count = config.get("active_cam_count", 6)
        self.set_active_count(active_count)

    def set_active_count(self, count):
        """Re-arranges the camera grid for the specified number of cameras."""
        # 1. Clear current grid layout
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # 2. Hide all widgets first
        for w in self.cam_widgets:
            w.hide()

        # 3. Determine Grid dimensions
        # 2 cams -> 1x2
        # 4 cams -> 2x2
        # 6 cams -> 2x3
        # 8 cams -> 2x4
        
        if count == 2:
            rows, cols = 1, 2
        else:
            rows = 2
            cols = (count + 1) // 2
            
        # 4. Add active widgets back to grid
        for i in range(count):
            if i < len(self.cam_widgets):
                row, col = divmod(i, cols)
                self.grid.addWidget(self.cam_widgets[i], row, col)
                self.cam_widgets[i].show()

        # 5. Reset stretches
        for r in range(10): self.grid.setRowStretch(r, 0)
        for c in range(10): self.grid.setColumnStretch(c, 0)
        
        for r in range(rows): self.grid.setRowStretch(r, 1)
        for c in range(cols): self.grid.setColumnStretch(c, 1)
        
        print(f"[UI] Grid updated for {count} cameras ({rows}x{cols})")

    def create_camera_widget(self, cam_index, name, default_ch, default_role):
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #070d18;
                border: none;
                border-radius: 12px;
            }
        """)
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Top Bar ──────────────────────────────────────────────────────
        top_bar = QWidget()
        top_bar.setFixedHeight(36)
        top_bar.setStyleSheet("background: transparent;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(12, 0, 12, 0)
        top_layout.setSpacing(6)

        status_dot = QLabel("●")
        status_dot.setStyleSheet(f"color: #334155; font-size: 9px;")
        top_layout.addWidget(status_dot)

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 11px;")
        top_layout.addWidget(name_lbl)
        top_layout.addStretch()

        # Role dropdown
        role_selector = QComboBox()
        role_selector.setObjectName("RoleSelector")
        role_selector.addItems(["Entrance", "Exit", "Monitor"])
        idx_map = {"entrance": 0, "exit": 1, "monitor": 2}
        role_selector.setCurrentIndex(idx_map.get(default_role, 2))
        role_selector.setMinimumWidth(70)
        role_selector.setFixedHeight(24)
        role_selector.currentIndexChanged.connect(
            lambda idx, ci=cam_index: self.backend.set_cam_role(ci, ["entrance", "exit", "monitor"][idx])
        )
        top_layout.addWidget(role_selector)

        # Source dropdown (Dynamic based on detected NVR channels)
        from config.cam_config_manager import CamConfigManager
        config = CamConfigManager.load_config()
        # [SMART UI] Unified Camera dropdown that doesn't confuse the user
        discovered_ips = config.get("discovered_ips", [])
        num_channels = max(config.get("num_channels", 6), len(discovered_ips), 20) # Default to at least 20 slots for swarms
        
        source_selector = QComboBox()
        source_selector.setObjectName("SourceSelector")
        source_selector.setEditable(True)
        source_selector.setInsertPolicy(QComboBox.NoInsert)
        source_selector.addItems([f"Camera {i+1}" for i in range(num_channels)] + ["Webcam 0", "Webcam 1", "Reset to Discovery"])
        
        # [FIX] Maps older 'Ch X' format to 'Camera X' to prevent all dropdowns collapsing to 'Camera 1'
        if default_ch.startswith("Ch "):
            default_ch = default_ch.replace("Ch ", "Camera ")
            
        source_selector.setCurrentText(default_ch)
        source_selector.setMinimumWidth(80)  # Flexible for small screens
        source_selector.setFixedHeight(24)
        
        # [HOT-SWAP FEATURE] Dynamically bind the backend change module
        # Use activated for dropdowns and returnPressed for manual entries
        source_selector.activated.connect(
            lambda idx, ci=cam_index, sel=source_selector: self._on_source_activated(ci, sel)
        )
        source_selector.lineEdit().returnPressed.connect(
            lambda ci=cam_index, sel=source_selector: self.backend.swap_camera_source(ci, sel.currentText())
        )
        
        top_layout.addWidget(source_selector)

        layout.addWidget(top_bar)

        # ── Video Area ────────────────────────────────────────────────────
        video_label = ROILabel(cam_index, self.backend)
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setText("NO SIGNAL")
        video_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 13px; letter-spacing: 2px; background: transparent;")
        video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(video_label, 1)

        # ── Bottom Bar ────────────────────────────────────────────────────
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(28)
        bottom_bar.setStyleSheet("background: transparent;")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 0, 12, 0)

        time_lbl = QLabel(datetime.now().strftime('%H:%M:%S'))
        time_lbl.setObjectName("Timestamp")
        time_lbl.setStyleSheet("color: #475569; font-size: 10px;")
        bottom_layout.addWidget(time_lbl)
        bottom_layout.addStretch()

        fps_lbl = QLabel("-- FPS")
        fps_lbl.setStyleSheet("color: #334155; font-size: 10px; font-weight: bold;")
        bottom_layout.addWidget(fps_lbl)

        layout.addWidget(bottom_bar)

        # Store references on the container object
        container.video_label = video_label
        container.fps_lbl = fps_lbl
        container.time_lbl = time_lbl
        container.status_dot = status_dot
        container.source_selector = source_selector
        container.role_selector = role_selector
        container._last_ts = 0
        container._fps_count = 0

        return container

    def _on_source_activated(self, cam_index, selector):
        """Internal helper to handle dropdown selection including the reset option."""
        text = selector.currentText()
        if "Reset" in text:
            # Revert to standard camera channel format
            text = f"Camera {cam_index + 1}"
            selector.setCurrentText(text)
            
        self.backend.swap_camera_source(cam_index, text)

    def reload_config(self):
        """Re-loads camera configuration and updates UI widgets."""
        from config.cam_config_manager import CamConfigManager
        config = CamConfigManager.load_config()
        cams = config.get("cams", [])
        
        for i, widget in enumerate(self.cam_widgets):
            if i < len(cams):
                c = cams[i]
                # Update the source selector dropdown
                if hasattr(widget, 'source_selector'):
                    new_src = c.get("source", f"Camera {i+1}")
                    if new_src.startswith("Ch "):
                        new_src = new_src.replace("Ch ", "Camera ")
                        
                    index = widget.source_selector.findText(new_src)
                    if index >= 0:
                        widget.source_selector.setCurrentIndex(index)
                    else:
                        widget.source_selector.setCurrentText(new_src)
                
                # Update role
                if hasattr(widget, 'role_selector'):
                    # Use findText instead of findData since we use strings directly
                    role_idx = widget.role_selector.findText(c.get("role", "monitor").capitalize())
                    if role_idx >= 0:
                        widget.role_selector.setCurrentIndex(role_idx)

    def update_frames(self):
        try:
            now = datetime.now()
            for i, widget in enumerate(self.cam_widgets):
                frame = None
                if self.backend.are_cameras_active:
                    # Read directly from camera threads for smooth, real-time display
                    # (bypasses the triage-bottlenecked latest_frames buffer)
                    if i < len(self.backend.caps) and self.backend.caps[i] is not None:
                        try:
                            ret, frame = self.backend.caps[i].read()
                            if not ret:
                                frame = None
                        except Exception:
                            frame = None

                if frame is not None:
                    widget._fps_count += 1
                    elapsed = now.timestamp() - widget._last_ts
                    if elapsed >= 1.0:
                        widget.fps_lbl.setText(f"{widget._fps_count} FPS")
                        widget.fps_lbl.setStyleSheet(f"color: {ThemeManager.COLORS['success']}; font-size: 10px; font-weight: bold;")
                        widget._fps_count = 0
                        widget._last_ts = now.timestamp()

                    widget.status_dot.setStyleSheet(f"color: {ThemeManager.COLORS['success']}; font-size: 9px;")
                    widget.time_lbl.setText(now.strftime('%H:%M:%S'))

                    # Draw ROI overlay if configured (Bypassed)
                    display_frame = frame
                    # if hasattr(self.backend, 'cam_rois') and i < len(self.backend.cam_rois):
                    #     roi = self.backend.cam_rois[i]
                    #     if roi != [0.0, 0.0, 1.0, 1.0]:
                    #         fh, fw = display_frame.shape[:2]
                    #         x1 = int(roi[0] * fw)
                    #         y1 = int(roi[1] * fh)
                    #         x2 = int(roi[2] * fw)
                    #         y2 = int(roi[3] * fh)
                    #         # Draw emerald green bounding box for the ROI active zone
                    #         cv2.rectangle(display_frame, (x1, y1), (x2, y2), (76, 175, 80), 2)
                    #         cv2.putText(display_frame, "ACTIVE ZONE", (x1 + 6, y1 + 18),
                    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (76, 175, 80), 1, cv2.LINE_AA)

                    # [PERFORMANCE] Resize in OpenCV first using fast INTER_NEAREST.
                    # This avoids converting/scaling large images in Python/Qt,
                    # reducing GUI thread CPU usage and GIL contention by 20x.
                    target_w = widget.video_label.width()
                    target_h = widget.video_label.height()
                    if target_w > 10 and target_h > 10:
                        display_frame = cv2.resize(display_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                    # Draw static border direction overlays showing current rules
                    if hasattr(self.backend, 'cam_direction_rules') and i < len(self.backend.cam_direction_rules):
                        rules = self.backend.cam_direction_rules[i]
                        fh, fw = display_frame.shape[:2]
                        
                        def get_action_color(act):
                            # (B, G, R) format
                            return (80, 175, 76) if act == "in" else ((80, 80, 244) if act == "out" else None)
                            
                        def draw_text_with_shadow(img, text, pos, scale, color):
                            # Draw drop shadow outline in black
                            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
                            # Draw foreground text
                            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
                            
                        # Top (UP)
                        up_color = get_action_color(rules.get("up"))
                        if up_color:
                            draw_text_with_shadow(display_frame, f"UP: {rules['up'].upper()} ^", (fw // 2 - 40, 20), 0.4, up_color)
                                        
                        # Bottom (DOWN)
                        down_color = get_action_color(rules.get("down"))
                        if down_color:
                            draw_text_with_shadow(display_frame, f"DOWN: {rules['down'].upper()} v", (fw // 2 - 45, fh - 10), 0.4, down_color)
                                        
                        # Left (LEFT)
                        left_color = get_action_color(rules.get("left"))
                        if left_color:
                            draw_text_with_shadow(display_frame, f"< {rules['left'].upper()}", (10, fh // 2), 0.4, left_color)
                                        
                        # Right (RIGHT)
                        right_color = get_action_color(rules.get("right"))
                        if right_color:
                            draw_text_with_shadow(display_frame, f"{rules['right'].upper()} >", (fw - 70, fh // 2), 0.4, right_color)
                        
                    rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb.shape
                    qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                    widget.video_label.setPixmap(QPixmap.fromImage(qt_img))
                else:
                    widget.video_label.setPixmap(QPixmap())
                    if self.connecting_status:
                        widget.video_label.setText("ESTABLISHING LINK...")
                        widget.video_label.setStyleSheet("color: #f59e0b; font-weight: 800; font-size: 13px; letter-spacing: 1px;")
                        widget.status_dot.setStyleSheet("color: #f59e0b; font-size: 9px;")
                    else:
                        widget.video_label.setText("NO SIGNAL")
                        widget.video_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 13px; letter-spacing: 2px;")
                        widget.status_dot.setStyleSheet("color: #334155; font-size: 9px;")
                    
                    widget.fps_lbl.setText("-- FPS")
                    widget.fps_lbl.setStyleSheet("color: #334155; font-size: 10px; font-weight: bold;")
        except (KeyboardInterrupt, SystemExit, RuntimeError):
            pass
