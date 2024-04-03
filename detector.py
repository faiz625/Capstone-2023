import cv2
import threading
import queue
import mediapipe as mp
import numpy as np
import gaze
import os
import sys
from utils import focal_length, distance_finder, capture_image
from mouse_movement import MoveMouse

FACE_DIST = 48.4
FACE_WIDTH = 13.4
move_mouse = MoveMouse(frame_width=640, frame_height=480) 

class Detector:
    def __init__(self, loaded_dist=20, tvec=40, move=False):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self._save_folder = 'captured_images'
        self._d = 0
        self._loaded_distance = loaded_dist
        self._tvec = tvec
        self._move = move

        self.cap = cv2.VideoCapture(0)
        self.frame_queue = queue.Queue()
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()

        capture_image('captured_images', frame=self.frame_queue.get())
        os.makedirs(self._save_folder, exist_ok=True)
    
    def get_frame(self):
        frame = self.frame_queue.get()
        frame = cv2.flip(frame, 1)
        f_x, f_y = None, None
        with self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            face_landmarks = self.landmarks(frame, face_mesh)
            try:
                self.iris_landmarks()
                self.get_face_info(frame)
                self.get_distance_from_webcam(frame)
            except:
                pass
            
            # perform gaze estimation
            if face_landmarks is not None:
                try:
                    left_pupil_loc, right_pupil_loc, gaze_point = gaze.gaze(frame, face_landmarks, self._loaded_distance, self._tvec)
                    gaze.show_gaze(left_pupil_loc, right_pupil_loc, gaze_point, frame)
                    f_x, f_y = move_mouse.final_coords(gaze_point)
                    if self._move:
                        move_mouse.move_cursor(f_x, f_y)
                except:
                    pass

            # Display the frame
            #cv2.imshow("Frame", frame)
        return f_x, f_y, self._d

    def landmarks(self, frame, face_mesh):
        # convert the frame from RGB to BGR color space
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # process the face mesh to detect facial landmarks
        results = face_mesh.process(rgb_frame)

        # get the height and width of the frame
        f_h, f_w = frame.shape[:2]

        # check if there are multiple face landmarks detected
        if results.multi_face_landmarks:
            # extract the x,y coords for each face landmark
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
        self._eye_dist = np.linalg.norm(right_pupil - left_pupil)
        self._left_eye_center = left_pupil
        self._right_eye_center = right_pupil
    
    def get_distance_from_webcam(self, frame):
        ref_image = cv2.imread("captured_images/captured_image.jpg")
        ref_image_face_width, _, _, _ = self.get_face_info(ref_image)
        focal_length_value = focal_length(FACE_DIST, FACE_WIDTH, ref_image_face_width)
        face_width_in_frame, Faces, _, _ = self.get_face_info(frame)
        for (face_x, face_y, face_w, face_h) in Faces:
            if face_width_in_frame != 0:
                self._d = distance_finder(focal_length_value, FACE_WIDTH, face_width_in_frame)
    
    def get_face_info(self, frame):
        face_width = 0
        face_center_x = 0
        face_center_y = 0
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in faces:
            face_width = w
            face_center_x = int(w / 2) + x
            face_center_y = int(h / 2) + y

        return face_width, faces, face_center_x, face_center_y

    def capture_frames(self):
        if not self.cap.isOpened():
            print("Error: Unable to open webcam.")
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            if not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def close_cap(self):
        print("Closing face detector...")
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = Detector()
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        detector.get_frame()
    detector.close_cap()
