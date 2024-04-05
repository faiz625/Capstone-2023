import cv2
import numpy as np
import os
import time  # Import time module

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

# Load user_info.txt to map face_id to names
face_id_to_name = {}
with open('user_info.txt', 'r') as file:
    lines = file.readlines()
    current_user = {}
    for line in lines:
        if line.startswith('Name'):
            current_user['Name'] = line.split(':')[-1].strip()
        elif line.startswith('Face ID'):
            current_user['Face ID'] = int(line.split(':')[-1].strip())
            face_id_to_name[current_user['Face ID']] = current_user['Name']

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

        if Id == 1 and (100 - confidence) > 40:  # Assuming Id=1 is the first user
            name = face_id_to_name.get(Id, "Unknown")
            print(f"User {name} is verified with confidence level of: {100 - confidence:.2f}%")
            cam.release()
            cv2.destroyAllWindows()
            exit()  # Exit the script

        if Id in face_id_to_name:
            name = face_id_to_name[Id]
            confidence_percent = round(100 - confidence, 2)
            Id = f"{name} {confidence_percent:.2f}%"
            cv2.rectangle(im, (x-22, y-90), (x+w+22, y-22), (0, 255, 0), -1)
            cv2.putText(im, str(Id), (x, y-40), font, 1, (255, 255, 255), 3)

    cv2.imshow('im', im)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
