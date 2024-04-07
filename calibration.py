import tkinter as tk
from tkinter import simpledialog
import pygame
import numpy
from detector import Detector
import itertools
import random
from mouse_movement import MoveMouse
import pandas as pd
from pymongo import MongoClient

# Declare username as a global variable
username = None

def ask_username():
    global username
    # Create a Tkinter root window
    root = tk.Tk()
    # Hide the root window (optional, if you only want to show the dialog)
    root.withdraw()
    # Ask for the username using a simple dialog
    username = simpledialog.askstring("Username", "Please enter your username:")
    print(f"Welcome, {username}!")

# Call ask_username at the start to prompt for the username
ask_username()

move_mouse = MoveMouse(frame_width=640, frame_height=480) 

class Target:
    def __init__(self, pos, radius=10):
        super().__init__()
        self.x = pos[0]
        self.y = pos[1]
        self.radius = radius

    def render(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), self.radius, 0)

def get_calibration_zones(w, h, target_radius):
    xs = (0 + target_radius, w // 2, w - target_radius)
    ys = (0 + target_radius, h // 2, h - target_radius)
    zones = list(itertools.product(xs, ys))
    return zones

pygame.init()
screen_size = pygame.display.Info()
w, h = screen_size.current_w, screen_size.current_h
screen = pygame.display.set_mode((w, h))
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

calibration_zones = get_calibration_zones(w, h, 15)
center = (w // 2, h // 2)
detector = Detector()
target = Target(
    center, radius=25
)
data = pd.DataFrame(columns=["calibrate_idx", "target_x", "target_y", "f_x", "f_y", "distance", "error_percentage"])

def calculate_error_percentage(target_x, target_y, f_x, f_y):
    error_x = abs(target_x - f_x)
    error_y = abs(target_y - f_y)
    error_percentage_x = (error_x / w) * 100
    error_percentage_y = (error_y / h) * 100
    error_percentage = (error_percentage_x + error_percentage_y) / 2
    return error_percentage

def save_data(calibrate_idx, target_x, target_y, f_x, f_y, distance, data):
    # Check if f_x or f_y is None and handle accordingly
    if f_x is None or f_y is None:
        print("Error: f_x or f_y is None. Skipping data save for this frame.")
        return  # Skip saving this frame's data

    error_percentage = calculate_error_percentage(target_x, target_y, f_x, f_y)
    data.loc[len(data)] = [calibrate_idx, target_x, target_y, f_x, f_y, distance, error_percentage]
    data.to_excel("calibration_data.xlsx", index=False)

def run_calibration():
    bg = random.choice(((0, 0, 0), (200, 200, 200)))
    calibrate_idx = 0
    running = True

    while running:
        screen.fill(bg)
        # vary bg colour so we get variation in data
        bg_origin = screen.get_at((0, 0))
        if bg_origin[0] <= (0, 0, 0)[0]:
            bg_should_increase = True
        elif bg_origin[0] >= (200, 200, 200)[0]:
            bg_should_increase = False

        if bg_should_increase:
            bg = (bg_origin[0] + 1, bg_origin[1] + 1, bg_origin[2] + 1, bg_origin[3])
        else:
            bg = (bg_origin[0] - 1, bg_origin[1] - 1, bg_origin[2] - 1, bg_origin[3])
        frame = detector.grab_frame()
        f_x, f_y, distance = detector.get_frame(frame)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                calibrate_idx += 1

        if calibrate_idx < len(calibration_zones):
            target.x, target.y = calibration_zones[calibrate_idx]
            target.render(screen)
            text = font.render("Stare at red circle and press space bar", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
            save_data(calibrate_idx, target.x, target.y, f_x, f_y, distance, data)
        elif calibrate_idx == len(calibration_zones):
            screen.fill((0, 0, 0))
            text = font.render("Calibration done. Press space bar to exit.", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=screen.get_rect().center))
        elif calibrate_idx > len(calibration_zones):
            calibrate_idx = 0
            break

        clock.tick(60)
        pygame.display.update()

run_calibration()
print(username)
# Read the Excel file into a DataFrame
df = pd.read_excel('calibration_data.xlsx')

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['EyeTrackerTest']
collection = db['calibrationData']

# Insert data into MongoDB
for index, row in df.iterrows():
    data_dict = row.to_dict()
    data_dict['username'] = username
    collection.insert_one(data_dict)

print('Data inserted into MongoDB successfully.')
pygame.quit()