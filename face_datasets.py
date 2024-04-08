import cv2
import os
from pymongo import MongoClient
import sys  # Import sys to access command-line arguments

# Function to assure path exists
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['EyeTrackerTest']
collection = db['face_images']

def save_image_to_mongodb(gray_face, username, count):
    # Convert image to binary
    is_success, buffer = cv2.imencode(".jpg", gray_face)
    binary_image = buffer.tobytes()
    
    # Insert into MongoDB
    collection.insert_one({
        "username": username,
        "image_count": count,
        "image_data": binary_image
    })

# Check if username is provided as a command-line argument
if len(sys.argv) > 1:
    username = sys.argv[1]  # The first argument after the script name is assumed to be the username
else:
    print("Error: No username provided.")
    sys.exit(1)  # Exit the script if no username is provided

vid_cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count = 0

while True:
    _, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        # Save the face image to MongoDB instead of locally
        gray_face = gray[y:y + h, x:x + w]
        save_image_to_mongodb(gray_face, username, count)
        cv2.imshow('frame', image_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count >= 100:  # Changed to '>= 100' for clarity
        break

vid_cam.release()
cv2.destroyAllWindows()
