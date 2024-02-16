import cv2
import numpy as np
import mediapipe as mp

class IrisDetection:
    def __init__(self):
        self._mp_drawing = mp.solutions.drawing_utils
        self._left_iris_landmarks_indexes = [6, 7, 8, 9] # 474-477
        self._right_iris_landmarks_indexes = [1, 2, 3, 4] # 469-472
        self._landmarks = None
        
    def landmarks(self, frame, face_mesh):
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
            # denormalize landmarks
            self._landmarks = np.array([
                np.multiply([p.x, p.y], [f_w, f_h]).astype(int)
                for p in face_landmarks
            ])
            return results.multi_face_landmarks[0]
    
    def iris_landmarks(self):
        self._iris_landmarks = self._landmarks[468:478]

        # left and right pupil landmark points
        left_pupil = self._iris_landmarks[5] # 473
        right_pupil = self._iris_landmarks[0] # 468

        # find the center coordinates between both pupils
        avg = (left_pupil + right_pupil) / 2

        return left_pupil, right_pupil, avg

    def show_right_eye_region(self, frame):
        eye_top = int(self._landmarks[386][1])
        eye_left = int(self._landmarks[362][0])
        eye_bottom = int(self._landmarks[374][1])
        eye_right = int(self._landmarks[263][0])

        cloned_image = frame.copy()
        cropped_right_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
        h, w, _ = cropped_right_eye.shape
        x = eye_left
        y = eye_top

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    def show_left_eye_region(self, frame):
        eye_top = int(self._landmarks[159][1])
        eye_left = int(self._landmarks[33][0])
        eye_bottom = int(self._landmarks[145][1])
        eye_right = int(self._landmarks[133][0])

        cloned_image = frame.copy()
        cropped_left_eye = cloned_image[eye_top:eye_bottom, eye_left:eye_right]
        h, w, _ = cropped_left_eye.shape
        x = eye_left
        y = eye_top

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    def draw_irises(self, frame):
        if self._iris_landmarks is not None:
            # Find the minimum enclosing circle for the left and right iris landmarks
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(self._iris_landmarks[self._left_iris_landmarks_indexes])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(self._iris_landmarks[self._right_iris_landmarks_indexes])
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
