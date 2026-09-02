"""
WebRTC Video Widget for PySide6 (Qt) using QWebEngineView and go2rtc.

Renders zero-latency (<50ms) hardware-accelerated WebRTC camera streams
directly inside the PySide6 desktop GUI window with fallback to native QPixmap.
"""

import os
import cv2
from datetime import datetime
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QStackedLayout, QSizePolicy
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QImage, QPixmap

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEngineSettings
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False


class WebRTCQtWidget(QWidget):
    """
    Embedded WebRTC Camera Player for PySide6.
    Uses Chromium WebEngine to decode and display RTSP streams via go2rtc with zero delay,
    with native OpenCV QPixmap drawing support as instant fallback.
    """
    
    def __init__(self, cam_index: int, backend_controller=None, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.backend = backend_controller
        self.is_playing = False
        self.stream_name = f"cam{cam_index}"
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 120)
        
        self.stacked_layout = QStackedLayout(self)
        self.stacked_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Native Fallback / Placeholder Label (Dark Aesthetic)
        self.native_label = QLabel("NO SIGNAL")
        self.native_label.setAlignment(Qt.AlignCenter)
        self.native_label.setStyleSheet("""
            QLabel {
                background-color: #070d18;
                color: #1e293b;
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 2px;
                border-radius: 8px;
            }
        """)
        self.stacked_layout.addWidget(self.native_label)
        
        # 2. Embedded WebEngine WebRTC View
        self.web_view = None
        if WEBENGINE_AVAILABLE:
            try:
                self.web_view = QWebEngineView()
                self.web_view.setStyleSheet("background: #070d18; border-radius: 8px;")
                self.web_view.page().setBackgroundColor(QColor("#070d18"))
                
                # Performance & Autoplay settings
                settings = self.web_view.settings()
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
                settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls, True)
                settings.setAttribute(QWebEngineSettings.PlaybackRequiresUserGesture, False)
                settings.setAttribute(QWebEngineSettings.ShowScrollBars, False)
                settings.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)
                settings.setAttribute(QWebEngineSettings.WebGLEnabled, True)
                
                self.web_view.setContextMenuPolicy(Qt.NoContextMenu)
                self.stacked_layout.addWidget(self.web_view)
            except Exception as e:
                print(f"[WebRTC Qt Cam {self.cam_index}] WebEngine initialization error: {e}")
                self.web_view = None

    def start_stream(self, stream_name: str = None):
        """Connects to the go2rtc WebRTC stream."""
        if stream_name:
            self.stream_name = stream_name
            
        self.is_playing = True
        
        if self.web_view and WEBENGINE_AVAILABLE:
            # go2rtc stream URL using WebSocket / WebRTC / MSE auto-negotiation
            url = f"http://127.0.0.1:1984/stream.html?src={self.stream_name}&background=true"
            self.web_view.setUrl(QUrl(url))
            self.stacked_layout.setCurrentWidget(self.web_view)
        else:
            self.stacked_layout.setCurrentWidget(self.native_label)

    def stop_stream(self):
        """Disconnects WebRTC and displays the NO SIGNAL screen."""
        self.is_playing = False
        if self.web_view and WEBENGINE_AVAILABLE:
            self.web_view.setUrl(QUrl("about:blank"))
        self.set_placeholder_text("NO SIGNAL")
        self.stacked_layout.setCurrentWidget(self.native_label)

    def update_native_frame(self, frame):
        """Fallback method to display OpenCV numpy frames directly on native label."""
        if frame is None:
            return
            
        try:
            target_w = self.width()
            target_h = self.height()
            if target_w > 10 and target_h > 10:
                display_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            else:
                display_frame = frame
                
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.native_label.setPixmap(QPixmap.fromImage(qt_img))
            self.stacked_layout.setCurrentWidget(self.native_label)
        except Exception:
            pass

    def set_placeholder_text(self, text: str, is_connecting: bool = False):
        """Update placeholder display text."""
        self.native_label.setText(text)
        self.native_label.setPixmap(QPixmap())
        if is_connecting:
            self.native_label.setStyleSheet("""
                QLabel {
                    background-color: #070d18;
                    color: #f59e0b;
                    font-weight: 800;
                    font-size: 13px;
                    letter-spacing: 1px;
                    border-radius: 8px;
                }
            """)
        else:
            self.native_label.setStyleSheet("""
                QLabel {
                    background-color: #070d18;
                    color: #1e293b;
                    font-weight: bold;
                    font-size: 13px;
                    letter-spacing: 2px;
                    border-radius: 8px;
                }
            """)
        self.stacked_layout.setCurrentWidget(self.native_label)
