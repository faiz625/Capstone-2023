import cv2
import numpy as np
from utility import draw_rectangle, draw_circle

def detect_eyes_and_pupils(frame):
    '''
    Detect eyes and pupils in a frame and draw rectangles and circles around them.

    Parameters:
        frame (numpy.ndarray): Input video frame.

    Returns:
        numpy.ndarray: The frame with detected eyes and pupils.
    '''
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        draw_rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            draw_rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]

            detect_pupils(frame, eye_roi, x + ex, y + ey)

    return frame

def detect_pupils(frame, eye_roi, eye_x, eye_y):
    '''
    Detect pupils within the given eye region and draw circles around them.

    Parameters:
        frame (numpy.ndarray): Input video frame.
        eye_roi (numpy.ndarray): Region of the eye for pupil detection.
        eye_x (int): X-coordinate of the eye within the frame.
        eye_y (int): Y-coordinate of the eye within the frame.

    '''
    circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (eye_x + circle[0], eye_y + circle[1])
            radius = circle[2]
            draw_circle(frame, center, radius, (0, 0, 255), 2)
