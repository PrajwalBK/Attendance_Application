import json
import os
import base64
# Calculate BASE_DIR locally to avoid circular import with config.py
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
            asset_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            for cfg in ['auth_config.json', 'cam_config.json', 'db_config.json']:
                src = os.path.join(asset_dir, 'config', cfg)
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

AUTH_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'auth_config.json')

class AuthManager:
    @staticmethod
    def get_cloud_url():
        return AuthManager.load_setting("last_cloud_url", "https://api.visionattendance.com")

    @staticmethod
    def is_local_mode():
        """Returns True if the system is currently in Local Testing mode."""
        return AuthManager.load_setting("is_local_mode", False)

    @staticmethod
    def get_active_url():
        """Returns the appropriate URL based on the current mode."""
        is_local = AuthManager.is_local_mode()
        if is_local:
            return AuthManager.load_setting("last_local_url", "http://127.0.0.1:8080")
        else:
            return AuthManager.get_cloud_url()

    @staticmethod
    def save_setting(key, value):
        """Save a single setting to the config file while preserving others."""
        try:
            data = {}
            if os.path.exists(AUTH_CONFIG_PATH):
                with open(AUTH_CONFIG_PATH, 'r') as f:
                    data = json.load(f)
            
            data[key] = value
            os.makedirs(os.path.dirname(AUTH_CONFIG_PATH), exist_ok=True)
            with open(AUTH_CONFIG_PATH, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print(f"[AUTH ERROR] Failed to save setting: {e}")
            return False

    @staticmethod
    def load_setting(key, default=None):
        """Load a single setting from the config file."""
        if not os.path.exists(AUTH_CONFIG_PATH):
            return default
        try:
            with open(AUTH_CONFIG_PATH, 'r') as f:
                data = json.load(f)
                return data.get(key, default)
        except:
            return default

    @staticmethod
    def save_credentials(email, password, api_url=None):
        """Saves obfuscated credentials to a local JSON file (Merges with existing settings)."""
        from config.config import API_BASE_URL
        target_url = api_url if api_url else API_BASE_URL
        try:
            # Load existing first to preserve other settings
            data = {}
            if os.path.exists(AUTH_CONFIG_PATH):
                with open(AUTH_CONFIG_PATH, 'r') as f:
                    data = json.load(f)

            # Simple obfuscation to prevent plain-text reading
            data["u"] = base64.b64encode(email.encode()).decode()
            data["p"] = base64.b64encode(password.encode()).decode()
            data["a"] = base64.b64encode(target_url.encode()).decode()
            
            # [PROFILE] Save URL history based on mode
            is_local = any(x in target_url for x in ["192.168.", "localhost", "127.0.0.1", "10."])
            if is_local:
                data["last_local_url"] = target_url
            else:
                data["last_cloud_url"] = target_url

            # Ensure folder exists
            os.makedirs(os.path.dirname(AUTH_CONFIG_PATH), exist_ok=True)
            with open(AUTH_CONFIG_PATH, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print(f"[AUTH ERROR] Failed to save session: {e}")
            return False

    @staticmethod
    def load_credentials():
        """Loads and de-obfuscates credentials if they exist."""
        # Use dynamic URL helper to support mode switching
        active_url = AuthManager.get_active_url()
        
        if not os.path.exists(AUTH_CONFIG_PATH):
            return None, None, active_url
            
        try:
            with open(AUTH_CONFIG_PATH, 'r') as f:
                data = json.load(f)
                # Fail gracefully if keys are missing
                if "u" not in data or "p" not in data:
                    return None, None, active_url
                    
                email = base64.b64decode(data.get("u")).decode()
                password = base64.b64decode(data.get("p")).decode()
                
                # [DYNAMIC] Priority: Honor the active mode selection
                is_local = data.get("is_local_mode", False)
                mode_url = data.get("last_local_url") if is_local else data.get("last_cloud_url")
                
                # If we have a mode-specific URL, use it. 
                # Otherwise fall back to 'a' or default
                if mode_url:
                    api_url = mode_url
                else:
                    api_url = base64.b64decode(data.get("a", "")).decode()
                
                if not api_url: 
                    api_url = active_url
                
                return email, password, api_url
        except Exception as e:
            print(f"[AUTH ERROR] Failed to load session: {e}")
            return None, None, active_url

    @staticmethod
    def clear_credentials():
        """Removes only the credentials while preserving other settings."""
        if os.path.exists(AUTH_CONFIG_PATH):
            try:
                with open(AUTH_CONFIG_PATH, 'r') as f:
                    data = json.load(f)
                for key in ["u", "p", "a"]:
                    if key in data: del data[key]
                with open(AUTH_CONFIG_PATH, 'w') as f:
                    json.dump(data, f)
            except:
                os.remove(AUTH_CONFIG_PATH)
