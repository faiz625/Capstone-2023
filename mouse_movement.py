import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import platform
pyautogui.FAILSAFE = False

if platform.system() == "Windows":
    import win32gui

class MoveMouse:
    def __init__(self, frame_width=640, frame_height=480, smoothing_factor=0.7, move_threshold=50, ma_window_size=10, inactivity_threshold=3):
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_factor = smoothing_factor
        self.move_threshold = move_threshold
        self.ma_window_size = ma_window_size  # Moving average window size
        self.gaze_points_window = []  # List to store recent gaze points
        self.last_x, self.last_y = None, None
        self.dpi = self.get_dpi()
        self.inactivity_threshold = inactivity_threshold  # Seconds
        self.last_active_time = time.time()
        # Initialize minimum and maximum gaze points
        self.gaze_min = [float('inf'), float('inf')]
        self.gaze_max = [-float('inf'), -float('inf')]

        # Define zones as tuples of (x_start, y_start, x_end, y_end)
        self.zones = [
            ((0, 0, self.screen_width / 3, self.screen_height / 3), 'Top-Left Zone'),
            ((self.screen_width / 3, 0, 2 * self.screen_width / 3, self.screen_height / 3), 'Top-Center Zone'),
            ((2 * self.screen_width / 3, 0, self.screen_width, self.screen_height / 3), 'Top-Right Zone'),
            # Add other zones
        ]

    def get_dpi(self):
        if platform.system() == "Windows":
            try:
                hdc = win32gui.GetDC(0)
                dpi = win32gui.GetDeviceCaps(hdc, 88)  # 88 is the LOGPIXELSX index
                win32gui.ReleaseDC(0, hdc)
                return dpi
            except Exception as e:
                print(f"Error getting DPI: {e}")
                return 96  # A common default DPI
        else:
            return 96
        
    def update_gaze_range(self, gaze_point):
        """Update the observed range of gaze points."""
        self.gaze_min[0] = min(self.gaze_min[0], gaze_point[0])
        self.gaze_min[1] = min(self.gaze_min[1], gaze_point[1])
        self.gaze_max[0] = max(self.gaze_max[0], gaze_point[0])
        self.gaze_max[1] = max(self.gaze_max[1], gaze_point[1])

    def scale_gaze_to_screen(self, gaze_point):
        """Dynamically scale gaze points to screen coordinates."""
        # Avoid division by zero
        range_x = max(self.gaze_max[0] - self.gaze_min[0], 1)
        range_y = max(self.gaze_max[1] - self.gaze_min[1], 1)
        
        # Normalize and scale the gaze point
        x_scaled = ((gaze_point[0] - self.gaze_min[0]) / range_x) * self.screen_width
        y_scaled = ((gaze_point[1] - self.gaze_min[1]) / range_y) * self.screen_height
        return int(x_scaled), int(y_scaled)
    
    def move_cursor(self, gaze_point):
        
        current_time = time.time()
        self.update_gaze_range(gaze_point)  # Update min and max gaze points
        x, y = self.scale_gaze_to_screen(gaze_point)  # Scale gaze point

        if self.last_x is not None and self.last_y is not None:
            distance = ((gaze_point[0] - self.last_x) ** 2 + (gaze_point[1] - self.last_y) ** 2) ** 0.5
            if distance < self.move_threshold:
                # If the gaze hasn't moved significantly, check the inactivity duration
                if current_time - self.last_active_time > self.inactivity_threshold:
                    # Assume eyes might be closed or user is not actively looking
                    print("Possible inactivity detected. Retaining cursor position.")
                    return
            else: 
                self.last_active_time = current_time
            # Calculate movement speed
            speed = ((x - self.last_x) ** 2 + (y - self.last_y) ** 2) ** 0.5
            # Adjust smoothing factor based on speed
            dynamic_smoothing = max(1, min(self.smoothing_factor, 10 / max(1, speed)))

            # Output the dynamic smoothing factor value
            print(f"Dynamic Smoothing Factor: {dynamic_smoothing}")

            x_smoothed = int(x * dynamic_smoothing + self.last_x * (1 - dynamic_smoothing))
            y_smoothed = int(y * dynamic_smoothing + self.last_y * (1 - dynamic_smoothing))
            if abs(x_smoothed - self.last_x) >= self.move_threshold or abs(y_smoothed - self.last_y) >= self.move_threshold:
                pyautogui.moveTo(x_smoothed, y_smoothed)
                self.last_x, self.last_y = x_smoothed, y_smoothed
        else:
            pyautogui.moveTo(x, y)
            self.last_x, self.last_y = x, y
            self.last_x, self.last_y = gaze_point[0], gaze_point[1]

        

        