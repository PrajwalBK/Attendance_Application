# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

block_cipher = None

# --- Automated Dependency Collection ---
# Collect ONNX Runtime DLLs (Critical for Face Detection)
binaries = collect_dynamic_libs('onnxruntime')

# Collect hidden imports for AI libraries
hidden_imports = (
    collect_submodules('insightface') +
    collect_submodules('skimage') +
    collect_submodules('sklearn') +
    collect_submodules('onnxruntime') +
    collect_submodules('supervision') +
    collect_submodules('qt_material') +
    [
        'PySide6.QtPrintSupport',
        'qtawesome.iconic_font',
        'cv2',
        'numpy',
        'PIL',
        'mysql.connector',
        'requests',
        'scipy.special.cython_special',
        'scipy.linalg.cython_blas',
        'scipy.linalg.cython_lapack',
        'onnxruntime.capi.onnxruntime_pybind11_state'
    ]
)

# Collect data files (Models, Icons, etc.)
added_files = [
    ('ui', 'ui'),
    ('config', 'config'),
    ('data/models', 'data/models'),
    ('database', 'database'),
    ('README.md', '.'),
]
added_files += collect_data_files('qtawesome')
added_files += collect_data_files('insightface')
added_files += collect_data_files('qt_material')

a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory='.'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionAttendance'
)
