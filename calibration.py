import cv2
import mediapipe as mp
import numpy as np
from iris_detection import IrisDetection
from datetime import datetime
import tkinter
import gaze

current_timestamp = datetime.now()

root = tkinter.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

def calibrate_gaze_estimation():
    # Initialize the webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)
    iris_detection = IrisDetection()
    # Initialize mediapipe face mesh
    mp_face_mesh = mp.solutions.face_mesh

    calibration_data = []  # List to store calibration data

    # Define the calibration points or targets in a 3x3 grid pattern
    calibration_points = [
        (108, 108),
        (534, 108),
        (960, 108),
        (1386, 108),
        (1812, 108),
        (108, 324),
        (534, 324),
        (960, 324),
        (1386, 324),
        (1812, 324),
        (108, 540),
        (534, 540),
        (960, 540),
        (1386, 540),
        (1812, 540),
        (108, 756),
        (534, 756),
        (960, 756),
        (1386, 756),
        (1812, 756),
        (108, 972),
        (534, 972),
        (960, 972),
        (1386, 972),
        (1812, 972)
    ]


    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        # Create a white background frame
        frame = np.ones((1080, 1920, 3), dtype=np.uint8) * 255  # White background frame

        current_point_index = 0  # Index of the current calibration point

        while current_point_index < len(calibration_points):

            # Draw the current calibration point on the frame
            point = calibration_points[current_point_index]
            cv2.circle(frame, point, 10, (0, 255, 0), -1)  # Draw a circle at the current calibration point position

            while True:
                # Read a frame from the webcam
                ret, webcam_frame = cap.read()
                if not ret:
                    break

                # Flip the frame horizontally for a selfie-view display
                webcam_frame = cv2.flip(webcam_frame, 1) 
                # get face landmarks
                face_landmarks = iris_detection.landmarks(webcam_frame, face_mesh)

                # perform gaze estimation
                if face_landmarks is not None:
                    gaze.gaze(webcam_frame, face_landmarks)

                # Get iris landmarks
                left_pupil, right_pupil, avg = iris_detection.iris_landmarks()


                #TODO calibration add data to list
                calibration_data.append({
                    'pupil coords': (left_pupil, right_pupil),
                    'points': point,
                    'timestamp': current_timestamp,
                    'screen size': (screen_width, screen_height),
                    # 'eye_landmarks': eye_landmarks,  # Assuming we can extract from iris detection
                    'calibration_accuracy': None  #calibration validation? TBD for now
                })

                # Display the frame
                cv2.imshow("Gaze", webcam_frame)
                cv2.imshow("Calibration", frame)

                # Wait for the space bar key press to move to the next calibration point
                if cv2.waitKey(1) == ord(' '):
                    # Clear the frame with a white background
                    frame.fill(255)
                    break

            current_point_index += 1

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    # Perform calibration data processing or storage here
    # You can save the calibration data to a file or process it further for calibration mapping

    print("Calibration phase completed.")
    print("Collected Calibration Data:")
    for data in calibration_data:
        print(data)

if __name__ == "__main__":
    calibrate_gaze_estimation()