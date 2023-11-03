import cv2
from gaze_detection import GazeTracking

def main():
    # Initialize the webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)
    gaze = GazeTracking()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,0), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (20, 95), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,0), 1)

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