import cv2
import numpy as np
import mediapipe as mp
import pymongo
import time
from datetime import datetime

def iris_tracking(frame, face_mesh):
    # landmark points for left and right irises
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    # Convert the frame from RGB to BGR color space
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Get the height and width of the frame
    img_h, img_w = frame.shape[:2]
    # Process the face mesh to detect facial landmarks
    results = face_mesh.process(rgb_frame)
    # Check if there are multiple face landmarks detected
    if results.multi_face_landmarks:
        # Extract the mesh points of one face
        mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
        # Find the minimum enclosing circle for the left and right iris landmarks
        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        # Convert the center points of the left and right iris to integer arrays
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        # Draw circles around the left and right iris on the frame
        cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)
        # Return the center points of the left and right iris
        return center_left, center_right

def main():
    # Initialize the MongoDB client
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["iris_tracking_db"]
    
    # Drop the existing collection (if it exists) to clear the database
    if "iris_data" in db.list_collection_names():
        db.drop_collection("iris_data")

    collection = db["iris_data"]

    # Initialize a timer to send data every second
    timer_start = time.time()

    # Initialize the webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)

    # Create an instance of the FaceMesh class from Mediapipe
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Perform iris tracking on the frame using the face mesh
            try:
                left_pupil, right_pupil = iris_tracking(frame, face_mesh)
            except:
                left_pupil, right_pupil = None, None

            # Display the coordinates of the left and right pupils on the frame
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (20, 75), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 1)

            # Display the frame
            cv2.imshow("Frame", frame)

            # Send data to the MongoDB collection every second
            if time.time() - timer_start >= 1:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = {
                    "left_pupil": left_pupil.tolist() if left_pupil is not None else None,
                    "right_pupil": right_pupil.tolist() if right_pupil is not None else None,
                    "timestamp": current_time
                }
                collection.insert_one(data)
                print("Timestamp:", current_time)  # Print the timestamp to the console
                timer_start = time.time()

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the webcam, close the OpenCV window, and disconnect from MongoDB
    cap.release()
    cv2.destroyAllWindows()
    client.close()

if __name__ == "__main__":
    main()
