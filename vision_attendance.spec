# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

block_cipher = None

# Collect ONNX Runtime DLLs (excluding OpenVINO provider to prevent DLL version mismatch popups)
binaries = []
for path, dest in collect_dynamic_libs('onnxruntime'):
    if 'openvino' not in os.path.basename(path).lower():
        binaries.append((path, dest))
    else:
        print(f"[SPEC] Excluded OpenVINO provider: {path}")

# [DLL FIX] Add OpenCV FFMPEG DLL to the root so Windows LoadLibrary can find it
import cv2
cv2_dir = os.path.dirname(cv2.__file__)
for f in os.listdir(cv2_dir):
    if 'ffmpeg' in f.lower() and f.endswith('.dll'):
        binaries.append((os.path.join(cv2_dir, f), '.'))
        print(f"[SPEC] Added OpenCV FFMPEG DLL: {f}")

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
    excludes=['tkinter', 'tcl', 'tk'],
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

# [POST-BUILD DLL COPY] Copy FFMPEG DLL to the root directory to guarantee it's found by LoadLibrary
try:
    import shutil
    for folder in ['VisionAttendance', 'VisionAttendanceYolo4']:
        dist_dir = os.path.join('dist', folder)
        cv2_dir_in_dist = os.path.join(dist_dir, 'cv2')
        if os.path.exists(cv2_dir_in_dist):
            for f in os.listdir(cv2_dir_in_dist):
                if 'ffmpeg' in f.lower() and f.endswith('.dll'):
                    src_dll = os.path.join(cv2_dir_in_dist, f)
                    dst_dll = os.path.join(dist_dir, f)
                    shutil.copy2(src_dll, dst_dll)
                    print(f"\n[POST-BUILD SUCCESS] Copied FFMPEG DLL to root: {dst_dll}\n")
except Exception as post_err:
    print(f"\n[POST-BUILD ERROR] Failed to copy FFMPEG DLL to root: {post_err}\n")
