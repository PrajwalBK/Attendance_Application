import os
import zipfile
import sys

def create_pi_zip():
    print("============================================================")
    print("  Vision Attendance System - Python Zip Packager (Pi)")
    print("============================================================")
    
    zip_name = "VisionAttendance_Pi.zip"
    
    # Files and folders to include
    include_dirs = ['config', 'core', 'database', 'ui', 'scripts']
    include_files = [
        'gui.py', 'main.py', 'api.py', 'requirements.txt', 
        'README.md', 'RASPBERRY_PI_SETUP.md', 'AttendanceSystem.desktop',
        'vision_attendance.spec'
    ]
    
    # Exclude patterns
    exclude_extensions = ('.pyc', '.pyo', '.log', '.db', '.sqlite3', '.zip')
    exclude_dirs = ['__pycache__', '.git', 'venv', 'build', 'dist', 'reid_cache', 'attendance_snapshots', 'pending_snapshots']

    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. Package specified directories
            for directory in include_dirs:
                if not os.path.exists(directory):
                    continue
                print(f"Packaging directory: {directory}...")
                for root, dirs, files in os.walk(directory):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    
                    for file in files:
                        if file.endswith(exclude_extensions):
                            continue
                        file_path = os.path.join(root, file)
                        # Store path relative to workspace root
                        archive_name = os.path.relpath(file_path, start=os.getcwd())
                        zipf.write(file_path, archive_name)

            # 2. Package root level files
            for file in include_files:
                if os.path.exists(file):
                    print(f"Packaging root file: {file}...")
                    zipf.write(file, file)

            # 3. Package models folder (EXCLUDING backup .zip files to keep package size small)
            models_dir = os.path.join('data', 'models')
            if os.path.exists(models_dir):
                print("Packaging models (excluding redundant zip backups)...")
                for root, dirs, files in os.walk(models_dir):
                    dirs[:] = [d for d in dirs if d not in exclude_dirs]
                    for file in files:
                        # Exclude any zip file backups inside the models folder
                        if file.endswith(exclude_extensions):
                            continue
                        file_path = os.path.join(root, file)
                        archive_name = os.path.relpath(file_path, start=os.getcwd())
                        zipf.write(file_path, archive_name)

        size_mb = os.path.getsize(zip_name) / (1024 * 1024)
        print("\n============================================================")
        print(f"SUCCESS: Created standard cross-platform zip archive:")
        print(f"   Name: {zip_name}")
        print(f"   Size: {size_mb:.2f} MB")
        print("============================================================")
        print("This ZIP uses standard Deflate compression and can be extracted")
        print("easily on Linux (Raspberry Pi) and Windows.")
        
    except Exception as e:
        print(f"Error creating zip archive: {e}")

if __name__ == '__main__':
    create_pi_zip()
