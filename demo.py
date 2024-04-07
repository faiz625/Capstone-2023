from mouse_movement import MoveMouse
from detector import Detector
from clicker import Clicker
import keyboard
import os
import threading
import pandas as pd
from flask import request, Flask, jsonify
from flask_cors import CORS
import psutil
import signal
import time
from multiprocessing import Process
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

move_mouse = MoveMouse(frame_width=640, frame_height=480)
file_path = "calibration_data.xlsx"
DEBUG = False

# Global variables to manage the detector thread
detector_thread = None
detector_running = False

# Function to run the detector in a separate thread
def run_detector():
    global detector_running
    move_mouse = MoveMouse(frame_width=640, frame_height=480)
    file_path = "calibration_data.xlsx"
    DEBUG = False
    
    if os.path.exists(file_path):
        data = pd.read_excel(file_path)
        average_distance = int(data["distance"].mean()) // 2
        average_tvec = data["distance"].mean()
        if DEBUG:
            print(f"dist: {average_distance}, tvec: {average_tvec}")
        detector = Detector(loaded_dist=average_distance, tvec=average_tvec, move=True)
        while detector_running:
            frame = detector.grab_frame()
            detector.get_frame(frame)

        detector.close_cap()
    else:
        print("Please run the calibration program first.")
        detector_running = False

def runner():
    """Wait for the main process to exit and then restart it."""
    time.sleep(1)  # Wait a bit for the main process to shutdown
    # Replace 'yourscript.py' with the name of your Flask application script
    os.system(f'py {sys.argv[0]}')

def restart():
    """Spawn a detached process to restart the application and exit the current process."""
    print("Restarting the Flask application...")
    # Starting the runner process
    p = Process(target=runner)
    p.start()
    # Exiting the current application
    os._exit(0)

@app.route('/start')
def start():
    print("Inside Start")
    global detector_thread, detector_running
    if detector_running:
        return jsonify({'message': 'Detector is already running'}), 400
    detector_running = True
    detector_thread = threading.Thread(target=run_detector)
    detector_thread.start()
    return jsonify({'message': 'Detector started'}), 200

@app.route('/stop')
def stop():
    print("Inside stop")
    global detector_running
    if not detector_running:
        return jsonify({'message': 'Detector is not running'}), 400
    detector_running = False
    return jsonify({'message': 'Detector stopped'}), 200

@app.route('/start_calibration')
def start_calibration_endpoint():
    print("water")
    from calibration import run_calibration
    # clicked = True
    # # username = request.form['username']
    # # print(f"Starting calibration for username: {username}") 
    
    # # Use threading to run the calibration without blocking the Flask server
    # thread = threading.Thread(target=run_calibration)
    # thread.start()

    # stop_flask_server()
    # time.sleep(5)
    # start_flask_server()
    # restart()

    return jsonify({"message": "Calibration started for user "}), 200

if __name__ == "__main__":
    app.run(port=8000, debug=True)
