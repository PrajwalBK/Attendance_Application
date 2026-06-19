import json
import os

import sys
def _get_base_dir():
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        test_file = os.path.join(exe_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return exe_dir
        except Exception:
            local_app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
            fallback_dir = os.path.join(local_app_data, 'VisionAttendance')
            os.makedirs(fallback_dir, exist_ok=True)
            # Bootstrap: copy bundled configurations if they don't exist in LocalAppData yet
            import shutil
            for cfg in ['auth_config.json', 'cam_config.json', 'db_config.json']:
                src = os.path.join(exe_dir, 'config', cfg)
                dst = os.path.join(fallback_dir, 'config', cfg)
                if os.path.exists(src) and not os.path.exists(dst):
                    try:
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(src, dst)
                    except: pass
            return fallback_dir
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = _get_base_dir()
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'db_config.json')

# Default MySQL configuration
DEFAULT_MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "database": "demo",
    "port": 3306,
}

class DBConfigManager:
    @staticmethod
    def load_config():
        """Load database configuration from JSON file or return defaults"""
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading {CONFIG_PATH}: {e}")
        
        # Return defaults if no file exists
        return DEFAULT_MYSQL_CONFIG

    @staticmethod
    def save_config(config_dict):
        """Save database configuration to JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config_dict, f, indent=4)
            return True, "Config saved successfully"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def delete_config():
        """Reset to defaults by deleting JSON file"""
        if os.path.exists(CONFIG_PATH):
            os.remove(CONFIG_PATH)
            return True
        return False
