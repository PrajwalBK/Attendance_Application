import winreg as reg
import sys
import os

class StartupManager:
    APP_NAME = "VisionAttendance"

    @staticmethod
    def enable_auto_startup():
        """Adds the application to the Windows startup registry."""
        try:
            # We use sys.executable which evaluates to python.exe in dev, 
            # or the compiled .exe when running as a PyInstaller bundle.
            # To avoid running 'python.exe' on user boot when testing in IDE,
            # we only activate this if we're running as a bundled exe.
            if getattr(sys, 'frozen', False):
                exe_path = sys.executable
                
                key = reg.HKEY_CURRENT_USER
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
                
                # Open the registry key
                open_key = reg.OpenKey(key, key_path, 0, reg.KEY_ALL_ACCESS)
                
                # Set the key to the path of this executable
                reg.SetValueEx(open_key, StartupManager.APP_NAME, 0, reg.REG_SZ, exe_path)
                
                # Close the key
                reg.CloseKey(open_key)
                print(f"[StartupManager] Enabled auto-startup for {exe_path}")
                return True
            else:
                print("[StartupManager] Development environment detected. Auto-start ignored.")
                return False
        except Exception as e:
            print(f"[StartupManager] Error enabling startup: {e}")
            return False

    @staticmethod
    def disable_auto_startup():
        """Removes the application from the Windows startup registry."""
        try:
            key = reg.HKEY_CURRENT_USER
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
            open_key = reg.OpenKey(key, key_path, 0, reg.KEY_ALL_ACCESS)
            reg.DeleteValue(open_key, StartupManager.APP_NAME)
            reg.CloseKey(open_key)
            print("[StartupManager] Disabled auto-startup.")
            return True
        except FileNotFoundError:
            # Key didn't exist
            return True
        except Exception as e:
            print(f"[StartupManager] Error disabling startup: {e}")
            return False
