import json
import os
import threading
import time
from config.config import BASE_DIR, WEBCAM_INDEX

# Resolves accurately via the shared absolute root
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'cam_config.json')

class CamConfigManager:
    _lock = threading.Lock()

    @staticmethod
    def load_config():
        """Load camera configuration from JSON file or return defaults"""
        config = None
        for attempt in range(5):
            if os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, 'r') as f:
                        config = json.load(f)
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"Error loading {CONFIG_PATH} (attempt {attempt+1}/5): {e}")
                    time.sleep(0.05)
            else:
                break

        if config is not None:
            try:
                needs_save = False
                cams = config.get("cams", [])
                discovered_ips = config.get("discovered_ips", [])
                
                # [MIGRATION] Support scaling to 10 cameras
                if "active_cam_count" not in config:
                    config["active_cam_count"] = 6
                    needs_save = True
                
                while len(cams) < 8:
                    i = len(cams)
                    cams.append({
                        "name": f"CAMERA {i+1}", 
                        "source": f"Camera {i+1}", 
                        "role": "monitor", 
                        "roi": [0.0, 0.0, 1.0, 1.0], 
                        "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}
                    })
                    needs_save = True
                    
                # [MIGRATION] If switching from 10 down to 8, trim the list
                if len(cams) > 8:
                    cams = cams[:8]
                    needs_save = True
                    
                # [MIGRATION] Auto-heal missing parameters
                for c in cams:
                    if "roi" not in c:
                        c["roi"] = [0.0, 0.0, 1.0, 1.0]
                        needs_save = True
                    if "direction_rules" not in c:
                        c["direction_rules"] = {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}
                        needs_save = True
                        
                config["cams"] = cams
                
                # Detect if all slots got stuck on "Camera 1" improperly
                if cams and all(c.get("source") in ["Camera 1", "Ch 1"] for c in cams[1:]):
                    print("[AUTO-HEALING] Detecting structural mapping glitch. Restoring incremental slots...")
                    for i, c in enumerate(cams):
                        # Only auto-fix 'Camera X' vs 'Ch X' format if they are all identical
                        c["source"] = f"Camera {i+1}"
                        # If NVR mode (single IP), ensure all slots share the primary IP
                        if len(discovered_ips) == 1:
                            c["ip"] = discovered_ips[0]
                    needs_save = True
                
                # Only write to disk if an actual migration/healing change was made
                if needs_save:
                    CamConfigManager.save_config(config)
                    
                return config
            except Exception as e:
                print(f"Error migrating configuration: {e}")
        
        # Default configuration
        return {
            "rtsp_template": str(WEBCAM_INDEX), # Use configured RTSP or webcam index
            "primary_source": "Channel 1",
            "secondary_source": "Channel 2",
            "num_channels": 16,
            "auto_start_on_boot": False,
            "active_cam_count": 6,
            "cams": [
                {"name": "CAMERA 1", "source": "Ch 1", "role": "entrance", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 2", "source": "Ch 2", "role": "exit", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 3", "source": "Ch 3", "role": "monitor", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 4", "source": "Ch 4", "role": "monitor", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 5", "source": "Ch 5", "role": "monitor", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 6", "source": "Ch 6", "role": "monitor", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 7", "source": "Ch 7", "role": "monitor", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}},
                {"name": "CAMERA 8", "source": "Ch 8", "role": "monitor", "roi": [0.0, 0.0, 1.0, 1.0], "direction_rules": {"up": "out", "down": "in", "left": "ignore", "right": "ignore"}}
            ]
        }

    @staticmethod
    def save_config(config_dict):
        """Save camera configuration to JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with CamConfigManager._lock:
                # Write to a temporary file first, then replace atomically to avoid truncation corruption
                temp_path = CONFIG_PATH + ".tmp"
                with open(temp_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
                os.replace(temp_path, CONFIG_PATH)
            return True, "Config saved successfully"
        except Exception as e:
            return False, str(e)
