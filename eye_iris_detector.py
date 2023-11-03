import cv2
import numpy as np

class EyeIrisDetector:
    """
    This class is responsible for detecting the iris of an eye and estimating
    the position of the pupil.

    Args:
        eye_frame (numpy.ndarray): A frame containing an eye and nothing else.
        threshold (int): Threshold value used to binarize the eye frame.
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def preprocess_eye_frame(eye_frame, threshold):
        """Preprocesses the eye frame to isolate the iris.

        Args:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else.
            threshold (int): Threshold value used to binarize the eye frame.

        Returns:
            A frame with a single element representing the iris.
        """

        # Apply bilateral filtering for noise reduction and detail preservation
        # Parameters: Diameter, Sigma values in color space, and sigma values in coordinate space
        processed_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)

        # Threshold the image to binarize it
        # Pixels with values greater than or equal to the threshold become 255, the rest become 0
        processed_frame = cv2.threshold(processed_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return processed_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the pupil by
        calculating the centroid.

        Args:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else.
        """
        self.iris_frame = self.preprocess_eye_frame(eye_frame, self.threshold)

        # Detect contours in the preprocessed iris frame
        # Using cv2.findContours to find the contours in the iris_frame
        # cv2.RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours
        # cv2.CHAIN_APPROX_NONE stores absolutely all the contour points, without any compression
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # Sort the detected contours by their areas (size)
        contours = sorted(contours, key=cv2.contourArea)

        # Try to calculate the moments of the second largest contour ([-2])
        try:
            # Calculate the moments (i.e., weighted average) of the contour
            moments = cv2.moments(contours[-2])

            # Calculate the x and y coordinates of the centroid by using the moments
            # The centroid coordinates are the weighted average of all pixel coordinates in the contour
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            # Handle potential errors, such as no contour or division by zero
            pass
