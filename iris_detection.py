import cv2
import numpy as np
import mediapipe as mp

class IrisDetection:
    def __init__(self):
        self._mp_drawing = mp.solutions.drawing_utils
        self._left_iris_landmarks = [474, 475, 476, 477]
        self._right_iris_landmarks = [469, 470, 471, 472]
        self._iris_landmarks = None
        
    def iris_tracking(self, frame, face_mesh):
        left_pupil, right_pupil, avg = None, None, (0, 0)  # Initialize avg with a default value
        # Convert the frame from RGB to BGR color space
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process the face mesh to detect facial landmarks
        results = face_mesh.process(rgb_frame)

        # Get the height and width of the frame
        f_h, f_w = frame.shape[:2]

        # Check if there are multiple face landmarks detected
        if results.multi_face_landmarks:
            # Extract the x,y coords for each face landmark
            face_landmarks = results.multi_face_landmarks[0].landmark
            self._iris_landmarks = np.array([
                np.multiply([p.x, p.y], [f_w, f_h]).astype(int)
                for p in face_landmarks
            ])

            # left and right pupil landmark points
            left_pupil = self._iris_landmarks[473]
            right_pupil = self._iris_landmarks[468]

            avg = (left_pupil + right_pupil) / 2

        return left_pupil, right_pupil, avg

    def draw_irises(self, frame):
        if self._iris_landmarks is not None:
            # Find the minimum enclosing circle for the left and right iris landmarks
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(self._iris_landmarks[self._left_iris_landmarks])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(self._iris_landmarks[self._right_iris_landmarks])
            # Convert the center points of the left and right iris to integer arrays
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            # Draw circles around the left and right iris on the frame
            cv2.circle(frame, center_left, int(l_radius), (0,255,0), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (0,255,0), 1, cv2.LINE_AA)

    def display_iris_coords(self, frame, left_iris, right_iris):
        # Display the coordinates of the left and right pupils on the frame
        cv2.putText(frame, "Left pupil:  " + str(left_iris), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,0), 1)
        cv2.putText(frame, "Right pupil: " + str(right_iris), (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,0), 1)
