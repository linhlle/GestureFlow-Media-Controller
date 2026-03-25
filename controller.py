import pyautogui
import subprocess
import threading

class SystemController:
    def __init__(self):
        self.pyautogui = pyautogui
        self.pyautogui.FAILSAFE = True
        self.pyautogui.PAUSE = 0

        self.gesture_map = {
            1: {"name": "Spotlight", "keys": ["command", "space"]},
            2: {"name": "Mission Control", "keys": ["ctrl", "up"]},
            3: {"name": "App Switcher", "keys": ["command", "tab"]}
        }

    def get_volume(self):
        try:
            cmd = ["osascript", "-e", "output volume of (get volume settings)"]
            result = subprocess.check_output(cmd).decode('utf-8').strip()
            return int(result)
        except Exception as e:
            print(f"Volume error: {e}")
            return 0
        
    def set_volume(self, value):
        try:
            value = max(0, min(100, value))
            cmd = ["osascript", "-e", f"set volume output volume {value}"]
            # subprocess.run(cmd)
            self._execute_async(cmd)
        except Exception as e:
            print(f"Volume error: {e}")
            return 0
        
    def adjust_volume(self, delta):
        current = self.get_volume()
        self.set_volume(current + delta)

    def execute_gesture(self, gesture_id):
        if gesture_id in self.gesture_map:
            keys = self.gesture_map[gesture_id]["keys"]
            self.pyautogui.hotkey(*keys)
            print(f"Executed: {self.gesture_map[gesture_id]['name']}")

    def _execute_async(self, cmd):
        threading.Thread(target=lambda: subprocess.run(cmd), daemon=True).start()