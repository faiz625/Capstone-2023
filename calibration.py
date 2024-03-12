import cv2
import mediapipe as mp
import numpy as np
from gaze import gaze
from iris_detection import IrisDetection
from datetime import datetime
import tkinter
from mouse_movement import MoveMouse  # Make sure to import your MoveMouse class

current_timestamp = datetime.now()

root = tkinter.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

move_mouse = MoveMouse(frame_width=1920, frame_height=1080)  # Initialize with your screen's resolution

def calibrate_gaze_estimation():
    # Initialize the webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)
    iris_detection = IrisDetection()
    # Initialize mediapipe face mesh
    mp_face_mesh = mp.solutions.face_mesh

    calibration_data = []  # List to store calibration data

    # Define the calibration points or targets in a 3x3 grid pattern
    # Adjusted to match your circles file's calculation for center positions
    num_rows = 5
    num_cols = 5
    horizontal_spacing = (screen_width - 2 * (screen_width // (2 * max(num_rows, num_cols)))) // (num_cols - 1)
    vertical_spacing = (screen_height - 2 * (screen_height // (2 * max(num_rows, num_cols)))) // (num_rows - 1)
    
    calibration_points = [
        (j * horizontal_spacing + (screen_width // (2 * max(num_rows, num_cols))), i * vertical_spacing + (screen_height // (2 * max(num_rows, num_cols))))
        for i in range(num_rows) for j in range(num_cols)
    ]

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        # Create a white background frame for calibration dots
        calibration_frame = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

        current_point_index = 0

        while current_point_index < len(calibration_points):

            # Draw the current calibration point on the calibration frame
            point = calibration_points[current_point_index]
            cv2.circle(calibration_frame, point, 10, (0, 255, 0), -1)

            while True:
                # Read a frame from the webcam
                ret, webcam_frame = cap.read()
                if not ret:
                    break

                # Flip the frame horizontally for a selfie-view display
                webcam_frame = cv2.flip(webcam_frame, 1)

                # Get face landmarks
                face_landmarks = iris_detection.landmarks(webcam_frame, face_mesh)

                # Perform gaze estimation
                if face_landmarks is not None:
                    gaze_L, gaze_R = gaze(webcam_frame, face_landmarks)
                    avg_gaze = ((gaze_L[0] + gaze_R[0]) / 2, (gaze_L[1] + gaze_R[1]) / 2)
                    move_mouse.move_cursor(avg_gaze)  # Move the cursor based on gaze

                # Display the webcam frame (to see yourself) and the calibration frame (with the circles)
                cv2.imshow("Webcam Frame", webcam_frame)
                cv2.imshow("Calibration Frame", calibration_frame)

                # Wait for the space bar key press to move to the next calibration point
                if cv2.waitKey(1) == ord(' '):
                    calibration_frame.fill(255)  # Clear the frame with a white background
                    break

            current_point_index += 1

    # Release the webcam and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_gaze_estimation()