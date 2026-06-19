import os
from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

class ThemeManager:
    """Handles the visual styling of the PySide6 application."""

    COLORS = {
        'primary':    '#159b92',
        'secondary':  '#94a3b8',
        'background': '#121b2d', # Sleek Slate (Lighter than Deep Navy)
        'sidebar':    '#0a1120',
        'card':       '#1a253a', # High Contrast Card
        'text':       '#f8fafc',
        'text_dim':   '#94a3b8',
        'success':    '#10b981',
        'warning':    '#f59e0b',
        'error':      '#ef4444',
        'accent':     '#1abc9c',
    }

    @staticmethod
    def apply_theme(app: QApplication):
        apply_stylesheet(app, theme='dark_blue.xml')

        C = ThemeManager.COLORS
        qss = f"""
/* ── Global ─────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {C['background']};
    color: {C['text']};
    font-family: 'Segoe UI', Inter, Arial, sans-serif;
    font-size: 13px;
}}

/* ── Sidebar ─────────────────────────────────────────────────── */
QFrame#Sidebar {{
    background-color: {C['sidebar']};
    border-right: 1px solid #1e293b;
}}

/* ── Nav Buttons ─────────────────────────────────────────────── */
QPushButton#NavButton {{
    background-color: transparent !important;
    border: none !important;
    border-radius: 10px;
    padding: 12px 20px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
    color: {C['text_dim']};
    outline: none !important;
    qproperty-flat: true;
}}
QPushButton#NavButton:hover {{
    background-color: rgba(21, 155, 146, 0.08);
    color: {C['text']};
    border: none !important;
}}
QPushButton#NavButton:checked {{
    background-color: rgba(21, 155, 146, 0.15);
    color: {C['primary']};
    font-weight: 700;
    border-left: 4px solid {C['primary']};
    border-radius: 0 10px 10px 0;
}}
/* Kill any icon-specific background boxes */
QPushButton#NavButton QIcon {{
    background: transparent !important;
    border: none !important;
}}

/* ── Action Buttons ────────────────────────────────────────── */
/* Start (Positive Action) */
QPushButton#StartBtn {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #059669) !important;
    border: none !important;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 700;
    font-size: 11px;
    color: white !important;
    min-width: 120px;
}}
QPushButton#StartBtn:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #34d399, stop:1 #10b981) !important;
}}

/* Stop (Negative Action) */
QPushButton#StopBtn {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ef4444, stop:1 #b91c1c) !important;
    border: none !important;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 700;
    font-size: 11px;
    color: white !important;
    min-width: 120px;
}}
QPushButton#StopBtn:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f87171, stop:1 #ef4444) !important;
}}

/* Disabled / Inactive State */
QPushButton#DisabledAction {{
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 700;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.2) !important;
    min-width: 120px;
}}

/* ── Filter Buttons (Segmented Look) ─────────────────────────── */
QPushButton#FilterButton {{
    background-color: rgba(255, 255, 255, 0.04);
    color: {C['text_dim']};
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 6px 16px;
    font-size: 11px;
    font-weight: 700;
    min-width: 70px;
}}
QPushButton#FilterButton:hover {{
    background-color: rgba(21, 155, 146, 0.1);
    color: {C['text']};
    border-color: {C['primary']};
}}
QPushButton#FilterButton:checked {{
    background-color: {C['primary']};
    color: white;
    border: none;
}}

/* ── Camera Action Buttons (Stop — destructive state) ────────── */
QPushButton#StopCamBtn {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ef4444, stop:1 #b91c1c);
    border: none !important;
    border-radius: 10px;
    padding: 8px 24px;
    font-weight: 700;
    font-size: 13px;
    color: white;
    min-width: 140px;
}}
QPushButton#StopCamBtn:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f87171, stop:1 #ef4444);
}}
QPushButton#StopCamBtn:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b91c1c, stop:1 #991b1b);
}}

/* ── Cards ───────────────────────────────────────────────────── */
QFrame#Card {{
    background-color: {C['card']};
    border: 1px solid #1e293b;
    border-radius: 16px;
}}
/* Final fix for timestamps - absolute no border */
QFrame#Card QLabel {{
    background: transparent;
}}
/* Labels in metric cards */
QLabel#MetricVal {{
    font-size: 32px;
    font-weight: 800;
}}
QLabel#MetricTitle {{
    font-size: 13px;
    font-weight: 600;
    color: {C['text_dim']};
}}
QLabel#Timestamp {{
    border: none !important;
    background: transparent !important;
}}

/* ── Header title ────────────────────────────────────────────── */
QLabel#HeaderTitle {{
    font-size: 17px;
    font-weight: bold;
    color: white;
    letter-spacing: 0.3px;
}}
QFrame#TableHeader {{
    background-color: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid #1e293b;
    border-top: 1px solid #1e293b;
    padding: 2px 0;
}}
QFrame#TableHeader QLabel {{
    font-size: 11px;
    font-weight: 800;
    color: #475569;
    letter-spacing: 1px;
}}

/* ── Lists ───────────────────────────────────────────────────── */
QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
}}
QListWidget::item:selected {{
    background-color: rgba(21, 155, 146, 0.1);
}}

/* ── Progress Bar ────────────────────────────────────────────── */
QProgressBar {{
    border: none;
    background-color: #0f172a;
    height: 5px;
    border-radius: 3px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {C['primary']};
    border-radius: 3px;
}}

/* ── QComboBox (global dark styling) ─────────────────────────── */
QComboBox {{
    background-color: rgba(255, 255, 255, 0.04);
    color: {C['text']};
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 11px;
    combobox-popup: 0;
}}
QComboBox:hover {{
    border-color: {C['primary']};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {C['text_dim']};
    width: 0;
    height: 0;
    margin-right: 4px;
}}


/* ── Specific Selectors (Camera Page) ───────────────────────── */
QComboBox#RoleSelector {{
    background-color: rgba(21, 155, 146, 0.15);
    color: {C['primary']};
    border: 1px solid rgba(21, 155, 146, 0.3) !important;
    font-weight: bold;
    border-radius: 6px;
}}

QComboBox#SourceSelector {{
    background-color: rgba(255, 255, 255, 0.08);
    color: {C['secondary']};
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 6px;
}}

/* ── QComboBox dropdown popup (Extreme Force) ─────────────────── */
QComboBox QAbstractItemView, 
QComboBox QListView, 
QComboBox QAbstractItemView::viewport {{
    background-color: #0c1425 !important;
    color: #f1f5f9 !important;
    border: 1px solid #159b92 !important;
    selection-background-color: #159b92 !important;
    selection-color: white !important;
    outline: none !important;
}}

QComboBox QAbstractItemView::item {{
    background-color: #0c1425 !important;
    color: #f1f5f9 !important;
    min-height: 30px !important;
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: #159b92 !important;
    color: white !important;
}}


/* ── Scrollbar minimal ───────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #334155;
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── Input fields ────────────────────────────────────────────── */
QLineEdit {{
    background-color: {C['card']};
    border: 1px solid #334155;
    border-radius: 6px;
    color: {C['text']};
    padding: 6px 12px;
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {C['primary']};
    background-color: #1e293b;
}}

/* ── Message boxes ───────────────────────────────────────────── */
QMessageBox {{
    background-color: {C['background']};
    color: {C['text']};
}}
QMessageBox QPushButton {{
    background-color: {C['primary']};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 6px 20px;
    font-weight: bold;
}}
"""
        app.setStyleSheet(app.styleSheet() + qss)
