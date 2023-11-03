from __future__ import division
from collections import namedtuple
import cv2
import dlib
from eye import Eye
from pupil_calibration import PupilCalibration


class GazeTracking:
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = PupilCalibration()

        # _face_detector is used to detect faces
        self._face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # _predictor is used to get facial landmarks of a given face
        self._predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False
        
    def detect_face(self, frame):
        # array of faces coordinates
        faces = self._face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        dlib_faces = []
        for f in faces:
            # top left point of rectangle
            x0, y0 = f[0], f[1]
            # bottom right point of rectangle
            x1, y1 = f[0] + f[2], f[1] + f[3]
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # using detected face region, find eyes
            dlib_rect = dlib.rectangle(x0, y0, x1, y1)
            dlib_faces.append(dlib_rect)
        
        return dlib_faces

    def detect_eyes(self, frame, f):
        # get landmark points (reference face_landmarks.png)
        landmarks = self._predictor(frame, f)

        # named tuples to store eye landmark points
        RightEye = namedtuple("RightEye", ['left', 'right', 'top', 'bottom'])
        LeftEye = namedtuple("LeftEye", ['left', 'right', 'top', 'bottom'])

        # right eye landmark points
        rEye = RightEye(36, 39, [37, 38], [41, 40])
        # left eye landmark points
        lEye = LeftEye(42, 45, [43, 44], [47, 46])

        # right eye points
        r_left_point, r_right_point, r_top_point, r_bottom_point = self.get_eye_points(landmarks, rEye)
        # draw a horizontal line connecting the left and right points
        r_horizontal_line = cv2.line(frame, r_left_point, r_right_point, (0, 255, 0), 2)
        # draw a vertical line connecting the top and bottom points
        r_vertical_line = cv2.line(frame, r_top_point, r_bottom_point, (0, 255, 0), 2)

        # left eye points
        l_left_point, l_right_point, l_top_point, l_bottom_point = self.get_eye_points(landmarks, lEye)
        # draw a horizontal line connecting the left and right points
        l_horizontal_line = cv2.line(frame, l_left_point, l_right_point, (0, 255, 0), 2)
        # draw a vertical line connecting the top and bottom points
        l_vertical_line = cv2.line(frame, l_top_point, l_bottom_point, (0, 255, 0), 2)

        return landmarks
    
    def get_eye_points(self, points, Eye):
        # eye's left corner
        left_point = (points.part(Eye.left).x, points.part(Eye.left).y)
        # eye's right corner
        right_point = (points.part(Eye.right).x, points.part(Eye.right).y)
        # eye's top
        top_point = self.calc_midpoint(points.part(Eye.top[0]), points.part(Eye.top[1]))
        # eye's bottom
        bottom_point = self.calc_midpoint(points.part(Eye.bottom[0]), points.part(Eye.bottom[1]))

        return left_point, right_point, top_point, bottom_point

    def calc_midpoint(self, p1, p2):
        return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_face(frame)

        try:
            # eye detection
            landmarks = self.detect_eyes(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame