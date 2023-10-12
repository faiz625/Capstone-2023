Eye and Pupil Detection with OpenCV

Install the Required Libraries: Make sure you have OpenCV and NumPy installed. You can install them using pip if you haven't already:

pip install opencv-python numpy
Run main.py script (I used VS)

Camera setting: If you have a built-in webcam, it's usually Camera #0. If you have an external camera, you might need to change the VideoCapture(0) argument - use this to find where it exists:

import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()

cv2.destroyAllWindows()


After you run it, you should see a window pop up with a video feed from your webcam. It'll start recognizing your face and drawing rectangles around your eyes. If it detects your eyes, it'll try to find your pupils and draw circles around them. 

Exit: To stop the script, press 'q' on the window that popped up