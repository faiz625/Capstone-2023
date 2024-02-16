import cv2
import numpy as np
import pickle

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def gaze(frame, points):
    """
    The gaze function gets an image and face landmarks from mediapipe framework.
    The function draws the gaze direction into the frame.
    """

    '''
    2D image points.
    relative takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y) format
    '''
    image_points = np.array([
        relative(points.landmark[4], frame.shape),  # Nose tip
        relative(points.landmark[152], frame.shape),  # Chin
        relative(points.landmark[263], frame.shape),  # Left eye left corner
        relative(points.landmark[33], frame.shape),  # Right eye right corner
        relative(points.landmark[287], frame.shape),  # Left Mouth corner
        relative(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    '''
    2D image points.
    relativeT takes mediapipe points that is normalized to [-1, 1] and returns image points
    at (x,y,0) format
    '''
    image_points1 = np.array([
        relativeT(points.landmark[4], frame.shape),  # Nose tip
        relativeT(points.landmark[152], frame.shape),  # Chin
        relativeT(points.landmark[263], frame.shape),  # Left eye, left corner
        relativeT(points.landmark[33], frame.shape),  # Right eye, right corner
        relativeT(points.landmark[287], frame.shape),  # Left Mouth corner
        relativeT(points.landmark[57], frame.shape)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26),  # Left eye, left corner
        (43.3, 32.7, -26),  # Right eye, right corner
        (-28.9, -28.9, -24.1),  # Left Mouth corner
        (28.9, -28.9, -24.1)  # Right mouth corner
    ])

    '''
    3D model eye points
    The center of the eye ball
    '''
    Eye_ball_center_right = np.array([[-29.05], [32.7], [-39.5]])
    Eye_ball_center_left = np.array([[29.05], [32.7], [-39.5]])  # the center of the left eyeball as a vector.

    d = open("dist.pkl", "rb")
    dt = pickle.load(d)
    m = open("cameraMatrix.pkl", "rb")
    mt = pickle.load(m)

    camera_matrix = mt

    dist_coeffs = dt
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D secsseded
        # project pupil image point into 3d world point 
        pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        pupil_world_cord_right = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T 

        # 3D gaze point
        S = Eye_ball_center_left + (pupil_world_cord - Eye_ball_center_left) * 20
        S_R = Eye_ball_center_right + (pupil_world_cord_right - Eye_ball_center_right) * 20

        # Project a 3D gaze direction onto the image plane.
        (eye_pupil2D, _) = cv2.projectPoints((int(S[0]), int(S[1]), int(S[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        (eye_pupil2D_R, _) = cv2.projectPoints((int(S_R[0]), int(S_R[1]), int(S_R[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        # project 3D head pose into the image plane
        (head_pose, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        (head_pose_R, _) = cv2.projectPoints((int(pupil_world_cord[0]), int(pupil_world_cord[1]), int(40)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        
        head_pose = np.mean([head_pose[0][0], head_pose_R[0][0]], axis=0)
        # correct gaze for head rotation
        #gaze = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose[0][0] - left_pupil)
        gaze_L = left_pupil + (eye_pupil2D[0][0] - left_pupil) - (head_pose - left_pupil)
        gaze_R = right_pupil + (eye_pupil2D_R[0][0] - right_pupil) - (head_pose - right_pupil)
        mean_gaze = ((gaze_L[0] + gaze_R[0]) / 2, (gaze_R[1] + gaze_L[1]) / 2)

        # Draw gaze line into screen
        left_pupil_loc = (int(left_pupil[0]), int(left_pupil[1]))
        right_pupil_loc = (int(right_pupil[0]), int(right_pupil[1]))
        p2 = (int(mean_gaze[0]), int(mean_gaze[1]))
        cv2.line(frame, left_pupil_loc, p2, (0, 0, 255), 2)
        cv2.line(frame, right_pupil_loc, p2, (0, 0, 255), 2)
        
        return gaze_L, gaze_R