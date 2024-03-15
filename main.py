import cv2
import mediapipe as mp
from iris_detection import IrisDetection
from mouse_movement import MoveMouse
import time
import pyautogui
import datetime
import winsound

def main():
    # Initialize the webcam (0 is typically the built-in webcam)
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
        
        blink_count = 0
        start_time = time.time()

        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Perform iris tracking on the frame using the face mesh 
            left_pupil, right_pupil, avg = iris_detection.iris_tracking(frame, face_mesh)

            # draw circles around the irises:
            iris_detection.draw_irises(frame)

            # Display the coordinate    s of the left and right pupils on the frame
            iris_detection.display_iris_coords(frame, left_pupil, right_pupil)

            # move mouse
            try:
                move_mouse.move_cursor(avg)
                time.sleep(move_mouse.update_frequency)
            except KeyboardInterrupt:
                break
            
            # Detect left eye closure and perform click action
            left_eye_landmarks = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            right_eye_landmarks = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            if left_eye_landmarks:
                left_eye_landmarks = left_eye_landmarks[0].landmark
                left_eye_top = int(left_eye_landmarks[159].y * frame.shape[0])
                left_eye_bottom = int(left_eye_landmarks[145].y * frame.shape[0])

            if right_eye_landmarks:
                right_eye_landmarks = right_eye_landmarks[0].landmark
                right_eye_top = int(right_eye_landmarks[386].y * frame.shape[0])
                right_eye_bottom = int(right_eye_landmarks[374].y * frame.shape[0])

                if (left_eye_bottom - left_eye_top) < 10 and (right_eye_bottom - right_eye_top) < 10:
                    blink_count += 1
                    click_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Blink detected at {click_time}")

            # Reset blink count if more than 5 seconds have passed
            if time.time() - start_time > 5:
                if blink_count >= 2 and blink_count < 3:
                    pyautogui.click(button='right')
                    print("2 blinks detected in 5 seconds, performing L-click action")
                    winsound.Beep(2000, 200)
                elif blink_count >= 3 and blink_count < 4:
                    pyautogui.click()
                    print("3 blinks detected in 5 seconds, performing R-click action")
                    winsound.Beep(3000, 200)
                elif blink_count >= 4:
                    pyautogui.click()
                    pyautogui.click()
                    print("4 blinks detected in 5 seconds, performing double L-click action")
                    winsound.Beep(2000, 100)
                    winsound.Beep(2000, 100)
                blink_count = 0
                start_time = time.time()  
                winsound.Beep(1000, 200)   # every 5s timer beep

            
            # # Detect right eye closure and perform right click action
            # right_eye_landmarks = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            # if right_eye_landmarks:
            #     right_eye_landmarks = right_eye_landmarks[0].landmark
            #     right_eye_top = int(right_eye_landmarks[386].y * frame.shape[0])
            #     right_eye_bottom = int(right_eye_landmarks[374].y * frame.shape[0])
                
            #     if (right_eye_bottom - right_eye_top) < 10:
            #         pyautogui.click(button='right')
            #         click_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #         print(f"Right click registered at {click_time}")

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