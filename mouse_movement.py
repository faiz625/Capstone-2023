import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

class MoveMouse:
    def __init__(self, frame_width=640, frame_height=480, sensitivity=2.0, update_frequency=0.01):
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sensitivity = sensitivity
        self.update_frequency = update_frequency  # Time in seconds between cursor updates
        pyautogui.FAILSAFE = True

    def move_cursor(self, avg):
        x = int((avg[0] / self.frame_width) * self.screen_width)
        y = int((avg[1] / self.frame_height) * self.screen_height)
        x = min(max(x, 0), self.screen_width)
        y = min(max(y, 0), self.screen_height)
        print(x, y)
        pyautogui.moveTo(x, y)

