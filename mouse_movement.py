import cv2
import numpy as np
import mediapipe as mp
import pyautogui

class MoveMouse:
    def __init__(self, frame_width=640, frame_height=480, update_frequency=0.01):
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.update_frequency = update_frequency  # Time in seconds between cursor updates
        pyautogui.FAILSAFE = True

    def move_cursor(self, avg):
        if avg is not None:
            # get ratio of screen dimension to frame size
            r_x = int(self.screen_width / self.frame_width)
            r_y = int(self.screen_height / self.frame_height)
            # multiply center coords by ratio
            x = int(avg[0] * r_x)
            y = int(avg[1] * r_y)
            print(x, y)
            
            pyautogui.moveTo(x, y)
        