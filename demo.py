from mouse_movement import MoveMouse
from detector import Detector
from clicker import Clicker
import keyboard
import os
import pandas as pd

move_mouse = MoveMouse(frame_width=640, frame_height=480)
file_path = "calibration_data.xlsx"
DEBUG = False

if __name__ == "__main__":
    if os.path.exists(file_path):
        data = pd.read_excel("calibration_data.xlsx")
        average_distance = (data["distance"].mean()) // 2
        average_tvec = data["distance"].mean()
        if DEBUG:
            print(f"dist: {average_distance} tvec: {average_tvec}")
        detector = Detector(loaded_dist=average_distance, tvec=average_tvec, move=True)
        clicker = Clicker()
    else:
        print("Please run the calibration program first.")
        quit(1)
    while True:
        frame = detector.grab_frame()
        detector.get_frame(frame)
        clicker.clickLoop(frame)
        if keyboard.is_pressed('q'): # exit program
            break
    detector.close_cap()
