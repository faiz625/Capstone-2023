import cv2
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io
import os  

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
collection = db['face_images']

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# For detecting the faces, use Haarcascade Frontal Face default classifier
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Method for getting the images and label data from MongoDB
def getImagesAndLabelsFromMongoDB():
    faceSamples = []
    ids = []

    # Retrieve all documents from MongoDB
    for document in collection.find():
        # Convert binary image data back to numpy array image
        image_data = document["image_data"]
        image_id = document["face_id"]
        image = Image.open(io.BytesIO(image_data)).convert('L')
        img_numpy = np.array(image, 'uint8')

        # Detect face from the image
        faces = detector.detectMultiScale(img_numpy)

        # Looping for each face and appending it to their respective IDs
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(image_id)

    return faceSamples, ids

# Getting the faces and IDs from MongoDB
faces, ids = getImagesAndLabelsFromMongoDB()

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')
