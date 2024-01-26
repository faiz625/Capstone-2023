# Facial Recognition portion of Capstone

## Requirements

- Python 3.6
- OpenCV 3.3.0 or above
- Numpy 1.14.3 or above

## How does it work?

1. Gathers the dataset (pictures of user) using the webcam.
2.Trains a model from the acquired dataset and saves the model.
3.Use the trained model to classify faces in realtime.

## Steps to run:


**Step 1**. Run main.py and enter the name of your user. Once a name is entered, the program will take 100 pictures of the user and store it in the training_data folder. The information from the user will be stored to user_info.txt. If you want to add another user, run main.py again but enter a different username.

**Step 2**. Run training.py, this trains the model based on the images in the dataset and stores it in the saved_model folder.

**Step 3**: Run the face_recognition.py file to start detection.



