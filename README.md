<h1 align="center">Capstone Project 2023</h1>
<p align="center">Eye Feature Extraction with OpenCV and Python</p>

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

Clone this repository. You will need `Python v3.9` or above and `PIP` installed globally on your machine.

Then run the following command in a terminal:

`pip install -r requirements.txt`

## Usage

Run main.py script in a terminal

Camera setting: If you have a built-in webcam, it's usually Camera #0. If you have an external camera, you might need to change the VideoCapture(0) argument - use this to find where it exists:

    import cv2
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            cap.release()
    cv2.destroyAllWindows()

After you run it, you should see a window pop up with a video feed from your webcam. It'll start recognizing your face and draw a horizontal line from the edges of your eye and a vertical line from the top and bottom of your eye.

To stop the script, press 'q' on the window that popped up.
