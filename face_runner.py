import os
import subprocess
import tkinter as tk
import tkinter as tk
from tkinter import simpledialog

def get_username():
    ROOT = tk.Tk()
    ROOT.withdraw()
    username = simpledialog.askstring(title="Facial Recognition Setup",
                                      prompt="What's your username?")
    return username

def run_face_datasets_script(username):
    os.system(f"python face_datasets.py {username}")

def run_script(script_name):
    # Runs a Python script in a subprocess
    subprocess.run(['python', script_name], check=True)

def run_faceRecognition():
    username = get_username()
    print(f"Facial Recognition Calibration for {username}")
    # Run the face_datasets.py script with the obtained username
    run_face_datasets_script(username)
    print("Obtained 100 Image Samples")

    # After collecting the dataset, run training.py to train the model
    run_script('training.py')
    print("Facial Recognition Algorithm Trained")

run_faceRecognition()