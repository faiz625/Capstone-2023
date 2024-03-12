import cv2
import numpy as np
import mediapipe as mp
import pyautogui

class MoveMouse:
    def __init__(self, frame_width=640, frame_height=480, smoothing_factor=0.7, move_threshold=10):
        # Increased smoothing factor and move threshold for less sensitivity
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_factor = smoothing_factor  # Consider values closer to 1 for more smoothing
        self.move_threshold = move_threshold  # Increase for less sensitivity to minor movements
        self.last_x, self.last_y = None, None  # Initialize last cursor position

        # Define zones as tuples of (x_start, y_start, x_end, y_end)
        self.zones = [
            ((0, 0, self.screen_width / 3, self.screen_height / 3), 'Top-Left Zone'),
            ((self.screen_width / 3, 0, 2 * self.screen_width / 3, self.screen_height / 3), 'Top-Center Zone'),
            ((2 * self.screen_width / 3, 0, self.screen_width, self.screen_height / 3), 'Top-Right Zone'),
            # Add other zones as needed
        ]

    def move_cursor(self, gaze_point):
        x = int(gaze_point[0] * self.screen_width / self.frame_width)
        y = int(gaze_point[1] * self.screen_height / self.frame_height)

        if self.last_x is not None and self.last_y is not None:
            x_smoothed = int(x * self.smoothing_factor + self.last_x * (1 - self.smoothing_factor))
            y_smoothed = int(y * self.smoothing_factor + self.last_y * (1 - self.smoothing_factor))

            if abs(x_smoothed - self.last_x) >= self.move_threshold or abs(y_smoothed - self.last_y) >= self.move_threshold:
                # Check which zone the cursor is in
                for zone, name in self.zones:
                    if zone[0] <= x_smoothed <= zone[2] and zone[1] <= y_smoothed <= zone[3]:
                        print(f"Cursor in {name}")  # Trigger actions or provide feedback for specific zones
                        break

                # Update the cursor position
                pyautogui.moveTo(x_smoothed, y_smoothed)
                # Update last positions
                self.last_x, self.last_y = x_smoothed, y_smoothed
        else:
            pyautogui.moveTo(x, y)
            self.last_x, self.last_y = x, y

        