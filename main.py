import cv2
import mediapipe as mp
from iris_detection import IrisDetection
from mouse_movement import MoveMouse
import time

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