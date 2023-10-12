import cv2
from eye_detection import detect_eyes_and_pupils

def main():
    '''
    Main function for eye and pupil detection in a webcam video stream.

    This function initializes the webcam, continuously captures frames, detects eyes and pupils, 
    and displays the results.
    
    Press 'q' to exit the application.
    
    '''
    # Initialize the webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Detect eyes and pupils
        frame = detect_eyes_and_pupils(frame)

        # Display the frame with detected eyes and pupils
        cv2.imshow('Pupil Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
