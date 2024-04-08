import cv2
import numpy as np
from pymongo import MongoClient
from PIL import Image
import io
import os
import sys

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['EyeTrackerTest']
collection = db['face_images']

# Method for checking the existence of a path i.e., the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# For detecting the faces, use Haarcascade Frontal Face default classifier
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Assuming the default case where we use all data if no username is provided
username = None
if len(sys.argv) > 1:
    username = sys.argv[1]

# Method for getting the images and label data from MongoDB
def getImagesAndLabelsFromMongoDB(username=None):
    faceSamples = []
    ids = []
    users_to_ids = {}
    next_id = 0

    # Define the query based on whether a username is provided
    query = {"username": username} if username else {}

    # Retrieve documents from MongoDB based on the query
    for document in collection.find(query):
        # Convert binary image data back to numpy array image
        image_data = document["image_data"]
        user_id = document["username"]  # This is a string
        image = Image.open(io.BytesIO(image_data)).convert('L')
        img_numpy = np.array(image, 'uint8')

        # Map username to a unique integer, if not already done
        if user_id not in users_to_ids:
            users_to_ids[user_id] = next_id
            next_id += 1

        # Detect face from the image
        faces = detector.detectMultiScale(img_numpy)

        # Loop for each face and append it to their respective list
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(users_to_ids[user_id])  # Use mapped integer ID

    return faceSamples, ids

# Getting the faces and IDs from MongoDB
faces, ids = getImagesAndLabelsFromMongoDB(username)

# Train the model using the faces and IDs
recognizer.train(faces, np.array(ids))

# Save the model
assure_path_exists('saved_model/')
recognizer.write('saved_model/s_model.yml')
