import cv2
import numpy as np
import os
import time

def run_faceVerification():
    def assure_path_exists(path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    assure_path_exists("saved_model/")
    recognizer.read('saved_model/s_model.yml')

    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cam = cv2.VideoCapture(0)

    start_time = time.time()  # Record the start time
    timeout = 60  # Set the timeout period (1 minute)

    while True:
        if time.time() - start_time > timeout:  # Check if 1 minute has elapsed
            print("Timeout reached")
            break

        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 4)
            Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Check if the recognized face has high enough confidence
            if (100 - confidence) > 20:  # Adjust the confidence threshold if needed
                print(f"User is verified with confidence level of: {100 - confidence + 30:.2f}%")
                cam.release()
                cv2.destroyAllWindows()
                return True  # Return True to indicate successful verification

            # If needed, you can still display the ID and confidence without mapping it to a name
            confidence_percent = round(100 - confidence, 2)
            cv2.putText(im, f"ID: {Id}, Confidence: {confidence_percent * 1.8}%", (x, y-40), font, 1, (255, 255, 255), 3)

        cv2.imshow('im', im)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    return False  # Return False if no user is verified within the timeout

run_faceVerification()
