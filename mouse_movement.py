import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import platform

if platform.system() == "Windows":
    import win32gui

class MoveMouse:
    def __init__(self, frame_width=640, frame_height=480, smoothing_factor=0.7, move_threshold=10):
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_factor = smoothing_factor
        self.move_threshold = move_threshold
        self.last_x, self.last_y = None, None
        self.dpi = self.get_dpi()
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
        self.update_gaze_range(gaze_point)  # Update min and max gaze points
        x, y = self.scale_gaze_to_screen(gaze_point)  # Scale gaze point
        
        x = int(gaze_point[0] * self.screen_width / self.frame_width)
        y = int(gaze_point[1] * self.screen_height / self.frame_height)

        if self.last_x is not None and self.last_y is not None:
            x_smoothed = int(x * self.smoothing_factor + self.last_x * (1 - self.smoothing_factor))
            y_smoothed = int(y * self.smoothing_factor + self.last_y * (1 - self.smoothing_factor))
            if abs(x_smoothed - self.last_x) >= self.move_threshold or abs(y_smoothed - self.last_y) >= self.move_threshold:

                # Update the cursor position
                pyautogui.moveTo(x_smoothed, y_smoothed)
                # Update last positions
                self.last_x, self.last_y = x_smoothed, y_smoothed
        else:
            pyautogui.moveTo(x, y)
            self.last_x, self.last_y = x, y
        

        