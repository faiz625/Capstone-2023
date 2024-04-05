import cv2
import os
from pymongo import MongoClient

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
collection = db['face_images']

def save_image_to_mongodb(gray_face, face_id, count):
    # Convert image to binary
    is_success, buffer = cv2.imencode(".jpg", gray_face)
    binary_image = buffer.tobytes()
    
    # Insert into MongoDB
    collection.insert_one({
        "face_id": face_id,
        "image_count": count,
        "image_data": binary_image
    })

# Read the latest face_id from user_info.txt
with open('user_info.txt', 'r') as file:
    lines = file.readlines()
    latest_face_id = 0
    for line in lines:
        if line.startswith('Face ID'):
            face_id = int(line.split(':')[-1].strip())
            latest_face_id = max(latest_face_id, face_id)

vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

assure_path_exists("training_data/")

while True:
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Save the face image to MongoDB instead of locally
        gray_face = gray[y:y + h, x:x + w]
        save_image_to_mongodb(gray_face, latest_face_id, count)
        cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count > 100:
        break

vid_cam.release()
cv2.destroyAllWindows()
