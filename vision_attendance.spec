# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect dynamic libraries and data assets for onnxruntime, insightface, qtawesome, and qt_material
onnx_datas = collect_data_files('onnxruntime')
onnx_binaries = collect_dynamic_libs('onnxruntime')
insightface_datas = collect_data_files('insightface')

qtawesome_datas = collect_data_files('qtawesome')
qtmaterial_datas = collect_data_files('qt_material')
matplotlib_datas = collect_data_files('matplotlib')

# Collect skimage and lazy_loader with .pyi stub files (fixes ValueError: Cannot load imports from non-existent stub)
skimage_datas = collect_data_files('skimage', include_py_files=True)
lazy_loader_datas = collect_data_files('lazy_loader', include_py_files=True)
albumentations_datas = collect_data_files('albumentations', include_py_files=True)

# Collect OpenVINO if present in the environment
try:
    openvino_datas = collect_data_files('openvino')
    openvino_binaries = collect_dynamic_libs('openvino')
except Exception:
    openvino_datas = []
    openvino_binaries = []

# Project folders and data dependencies (dynamically checked)
added_datas = []
for folder in ['models', 'data/models', 'data', 'config', 'core', 'ui', 'database', 'assets', 'icons', 'bin']:
    if os.path.exists(folder):
        added_datas.append((folder, folder))

added_datas += (
    onnx_datas
    + insightface_datas
    + qtawesome_datas
    + qtmaterial_datas
    + matplotlib_datas
    + skimage_datas
    + lazy_loader_datas
    + albumentations_datas
    + openvino_datas
)
added_binaries = onnx_binaries + openvino_binaries

hidden_imports = [
    'insightface',
    'insightface.model_zoo',
    'insightface.app',
    'insightface.utils',
    'insightface.data',
    'onnxruntime',
    'onnxruntime.capi._pybind_state',
    'multiprocessing',
    'multiprocessing.popen_spawn_win32',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'PySide6.QtNetwork',
    'PySide6.QtSvg',
    'qtawesome',
    'qt_material',
    'matplotlib',
    'matplotlib.pyplot',
    'skimage',
    'skimage.transform',
    'skimage.filters',
    'skimage.draw',
    'skimage.color',
    'skimage.exposure',
    'skimage.feature',
    'skimage.io',
    'skimage.measure',
    'skimage.metrics',
    'skimage.morphology',
    'skimage.restoration',
    'skimage.segmentation',
    'skimage.util',
    'lazy_loader',
    'albumentations',
    'albucore',
    'supervision',
    'scipy',
    'scipy.spatial',
    'scipy.signal',
    'sklearn',
    'cv2',
    'numpy',
    'PIL',
    'requests',
    'urllib3',
    'sqlite3',
    'mysql',
    'mysql.connector',
    'pymysql',
    'pyttsx3',
    'reportlab',
    'core',
    'core.camera',
    'core.face_recognition',
    'core.multiprocess_handler',
    'core.gui_workers',
    'core.snapshot_manager',
    'core.track_manager',
    'core.tracker',
    'core.api_client',
    'core.frame_buffer',
    'core.voice_handler',
    'core.recorder',
    'core.attendance_tracker',
    'core.registration',
    'core.mask_detector',
    'core.video_processor',
    'core.utils',
    'core.startup_manager',
    'database',
    'database.database',
    'database.offline_storage',
    'config',
    'config.config',
    'config.auth_manager',
    'config.cam_config_manager',
    'config.db_config_manager',
    'ui',
    'ui.camera_page',
    'ui.cctv_setup',
    'ui.cloud_setup',
    'ui.records_page',
    'ui.registration_page',
    'ui.settings_page',
    'ui.theme_manager',
] + (
    collect_submodules('insightface')
    + collect_submodules('qtawesome')
    + collect_submodules('qt_material')
    + collect_submodules('skimage')
    + collect_submodules('lazy_loader')
    + collect_submodules('albumentations')
    + collect_submodules('supervision')
)

a = Analysis(
    ['gui.py'],
    pathex=['.', os.path.abspath('.')],
    binaries=added_binaries,
    datas=added_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'notebook', 'PyQt5'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionAttendance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if os.path.exists('assets/icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionAttendance',
)
