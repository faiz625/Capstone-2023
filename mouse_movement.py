import numpy as np
import pyautogui
from collections import deque 
from utils import clamp_value
pyautogui.FAILSAFE = False

class MoveMouse:
    def __init__(self, frame_width=640, frame_height=480):
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.track_x = deque([0] * 5, maxlen=5)
        self.track_y = deque([0] * 5, maxlen=5)

        # Define zones as tuples of (x_start, y_start, x_end, y_end)
        self.zones = [
            ((0, 0, self.screen_width / 3, self.screen_height / 3), 'Top-Left Zone'),
            ((self.screen_width / 3, 0, 2 * self.screen_width / 3, self.screen_height / 3), 'Top-Center Zone'),
            ((2 * self.screen_width / 3, 0, self.screen_width, self.screen_height / 3), 'Top-Right Zone'),
            # Add other zones
        ]

    def scale_gaze_to_screen(self, gaze_point):
        # Normalize and scale the gaze point
        r_x = int(self.screen_width / self.frame_width)
        r_y = int(self.screen_height / self.frame_height)
        x_scaled = gaze_point[0] * r_x
        y_scaled = gaze_point[1] * r_y
        return int(x_scaled), int(y_scaled)

    def smoothing(self, x, y):
        x_hat, y_hat = x, y
        x_hat_clamp = clamp_value(x_hat, self.screen_width)
        y_hat_clamp = clamp_value(y_hat, self.screen_height)
        self.track_x.append(x_hat_clamp)
        self.track_y.append(y_hat_clamp)
        weights = np.arange(1, 6)
        f_x = np.average(self.track_x, weights=weights)
        f_y = np.average(self.track_y, weights=weights)
        return f_x, f_y
    
    def final_coords(self, gaze_point):
        x, y = self.scale_gaze_to_screen(gaze_point)  # Scale gaze point
        f_x, f_y = self.smoothing(x, y)
        return f_x, f_y

    def move_cursor(self, f_x, f_y):
        pyautogui.moveTo(f_x, f_y)
        