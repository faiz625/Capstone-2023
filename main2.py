import cv2
import mediapipe as mp
from iris_detection import IrisDetection
from mouse_movement import MoveMouse
import time
import pyautogui
import datetime
import winsound

def main():
    cap = cv2.VideoCapture(0)
    iris_detection = IrisDetection()
    move_mouse = MoveMouse()
    # Initialize mediapipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        blink_start_time = None
        blink_duration = 0

        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Perform iris tracking on the frame using the face mesh 
            left_pupil, right_pupil, avg = iris_detection.iris_tracking(frame, face_mesh)

            # draw circles around the irises:
            iris_detection.draw_irises(frame)

            # Display the coordinates of the left and right pupils on the frame
            iris_detection.display_iris_coords(frame, left_pupil, right_pupil)

            # move mouse
            try:
                move_mouse.move_cursor(avg)
                time.sleep(move_mouse.update_frequency)
            except KeyboardInterrupt:
                break
            
            # Detect eye closure and measure duration
            left_eye_landmarks = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            if left_eye_landmarks:
                left_eye_landmarks = left_eye_landmarks[0].landmark
                left_eye_top = int(left_eye_landmarks[159].y * frame.shape[0])
                left_eye_bottom = int(left_eye_landmarks[145].y * frame.shape[0])

                if (left_eye_bottom - left_eye_top) < 10:  # Eye closed
                    if blink_start_time is None:
                        blink_start_time = time.time()
                    else:
                        blink_duration = time.time() - blink_start_time
                else:  # Eye open
                    if blink_duration > 1:
                        if blink_duration >= 4:
                            pyautogui.click(clicks=2)
                            print("Double Left click action performed")
                            winsound.Beep(2000, 100)
                            winsound.Beep(2000, 100)
                        elif blink_duration >= 3:
                            pyautogui.click(button='right')
                            print("Right click action performed")
                            winsound.Beep(3000, 200)
                        elif blink_duration >= 2:
                            pyautogui.click()
                            print("Left click action performed")
                            winsound.Beep(2000, 200)

                        blink_duration = 0
                        blink_start_time = None

            # Display the frame
            cv2.imshow("Frame", frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
