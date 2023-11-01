import cv2
import numpy as np
import dlib
from collections import namedtuple

def detect_face(face_detector, predictor, frame, gray, page):
    # array of faces coordinates
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for f in faces:
        # top left point of rectangle
        x0, y0 = f[0], f[1]
        # bottom right point of rectangle
        x1, y1 = f[0] + f[2], f[1] + f[3]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # using detected face region, find eyes
        dlib_rect = dlib.rectangle(x0, y0, x1, y1)
        detect_eyes(predictor, frame, gray, dlib_rect, page)

def detect_eyes(predictor, frame, gray, f, page):
    # get landmark points (reference face_landmarks.png)
    points = predictor(gray, f)

    # named tuples to store eye landmark points
    RightEye = namedtuple("RightEye", ['left', 'right', 'top', 'bottom'])
    LeftEye = namedtuple("LeftEye", ['left', 'right', 'top', 'bottom'])

    # right eye landmark points
    rEye = RightEye(36, 39, [37, 38], [41, 40])
    # left eye landmark points
    lEye = LeftEye(42, 45, [43, 44], [47, 46])

    # right eye points
    r_left_point, r_right_point, r_top_point, r_bottom_point = get_eye_points(points, rEye)
    # draw a horizontal line connecting the left and right points
    r_horizontal_line = cv2.line(frame, r_left_point, r_right_point, (0, 255, 0), 2)
    # draw a vertical line connecting the top and bottom points
    r_vertical_line = cv2.line(frame, r_top_point, r_bottom_point, (0, 255, 0), 2)

    # left eye points
    l_left_point, l_right_point, l_top_point, l_bottom_point = get_eye_points(points, lEye)
    # draw a horizontal line connecting the left and right points
    l_horizontal_line = cv2.line(frame, l_left_point, l_right_point, (0, 255, 0), 2)
    # draw a vertical line connecting the top and bottom points
    l_vertical_line = cv2.line(frame, l_top_point, l_bottom_point, (0, 255, 0), 2)

    # use right eye for gaze detection
    gaze_detection(points, rEye, page)

def get_eye_points(points, Eye):
    # eye's left corner
    left_point = (points.part(Eye.left).x, points.part(Eye.left).y)
    # eye's right corner
    right_point = (points.part(Eye.right).x, points.part(Eye.right).y)
    # eye's top
    top_point = calc_midpoint(points.part(Eye.top[0]), points.part(Eye.top[1]))
    # eye's bottom
    bottom_point = calc_midpoint(points.part(Eye.bottom[0]), points.part(Eye.bottom[1]))

    return left_point, right_point, top_point, bottom_point

def calc_midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def gaze_detection(points, rEye, page):
    pass

def main():
    # Initialize the webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)

    # get screen size
    size_screen = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # blank page for eye gaze
    page = (np.zeros((int(size_screen[0]), int(size_screen[1]), 3))).astype('uint8')

    # face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # eye detector using pre-trained model
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detect_face(face_detector, predictor, frame, gray, page)

        # Display the frame
        cv2.imshow("Frame", frame)
        cv2.imshow("test", page)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()