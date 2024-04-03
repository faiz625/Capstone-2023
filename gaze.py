import cv2
import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)

def gaze(frame, points, distance_from_screen, tvec):
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

    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # 2d pupil location
    left_pupil = relative(points.landmark[468], frame.shape)
    right_pupil = relative(points.landmark[473], frame.shape)

    # Transformation between image point to world point
    _, transformation, _ = cv2.estimateAffine3D(image_points1, model_points, ransacThreshold=6)  # image to world transformation

    if transformation is not None:  # if estimateAffine3D sucesseded
        # project pupil image point into 3d world point 
        left_pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T
        right_pupil_world_cord = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

        # 3D gaze point
        L = Eye_ball_center_left + (left_pupil_world_cord - Eye_ball_center_left) * distance_from_screen
        R = Eye_ball_center_right + (right_pupil_world_cord - Eye_ball_center_right) * distance_from_screen

        # Project a 3D gaze direction onto the image plane.
        (left_eye_pupil2D, _) = cv2.projectPoints((int(L[0]), int(L[1]), int(L[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        (right_eye_pupil2D, _) = cv2.projectPoints((int(R[0]), int(R[1]), int(R[2])), rotation_vector,
                                             translation_vector, camera_matrix, dist_coeffs)
        # project 3D head pose into the image plane
        (left_head_pose, _) = cv2.projectPoints((int(left_pupil_world_cord[0]), int(left_pupil_world_cord[1]), int(tvec)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        (right_head_pose, _) = cv2.projectPoints((int(right_pupil_world_cord[0]), int(right_pupil_world_cord[1]), int(tvec)),
                                           rotation_vector,
                                           translation_vector, camera_matrix, dist_coeffs)
        
        gaze_left = left_pupil + (left_eye_pupil2D[0][0] - left_pupil) - (left_head_pose[0][0] - left_pupil)
        gaze_right = right_pupil + (right_eye_pupil2D[0][0] - right_pupil) - (right_head_pose[0][0] - right_pupil)
        gaze_point =  (int((gaze_left[0] + gaze_right[0]) / 2), int((gaze_left[1] + gaze_right[1]) / 2))

        left_pupil_loc = (int(left_pupil[0]), int(left_pupil[1]))
        right_pupil_loc = (int(right_pupil[0]), int(right_pupil[1]))
        
        return left_pupil_loc, right_pupil_loc, gaze_point
    
def show_gaze(left_pupil, right_pupil, gaze_point, frame):
    # Draw gaze line into screen
    left_pupil_loc = (int(left_pupil[0]), int(left_pupil[1]))
    right_pupil_loc = (int(right_pupil[0]), int(right_pupil[1]))
    p2 = (int(gaze_point[0]), int(gaze_point[1]))
    cv2.line(frame, left_pupil_loc, p2, (0, 0, 255), 2)
    cv2.line(frame, right_pupil_loc, p2, (0, 0, 255), 2)