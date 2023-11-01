import cv2
from .EyeIrisDetector import EyeIrisDetector

LEFT_EYE = 0
RIGHT_EYE = 1

class PupilCalibration:
    """
    This class calibrates the pupil detection by finding the
    best binarization threshold value for the person and the webcam
    """
    def __init__(self):
        # number of frames to use for pupil calibration
        self.nb_frames = 20
        # lists to store threshold values for both eyes
        self.thresholds_left = []
        self.thresholds_right = []
    
    def calibration_completed(self):
        """
        Returns true if the calibration is completed
        """
        # comparing the number of captured thresholds for both eyes with the specified number of frames
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames
    
    def threshold(self):
        """
        Returns the threshold value for the given eye
        """
        if LEFT_EYE == 0:
            # calculates the threshold for the left eye by averaging the thresholds in thresholds_left list
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif RIGHT_EYE == 1:
            # calculates the threshold for the right eye by averaging the thresholds in thresholds_right list
            return int(sum(self.thresholds_right) / len(self.thresholds_right))
    
    @staticmethod
    def iris_size(frame):
        """
        Calculates the size of the iris in a given frame

        Parameters
        ----------
            frame (numpy.ndarray): Binarized iris frame
        """
        # crops the frame by removing 5 pixels from each side
        frame = frame[5:-5, 5:-5]
        # calculates the number of pixels in the cropped frame and counts the number of black pixels
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        # ratio of black pixels to the total number of pixels
        return nb_blacks / nb_pixels
    
    @staticmethod
    def find_best_threshold(eye_frame):
        """
        Finds the best threshold value for a given eye frame

        Parameters
        ----------
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        # average iris size
        average_iris_size = 0.48
        # dictionary to store the trial results
        trials = {}
        # thresholds from 5 to 100 with a step size of 5
        for threshold in range(5, 100, 5):
            # preprocesses the eye frame
            iris_frame = EyeIrisDetector.preprocess_eye_frame(eye_frame, threshold)
            # calculates the iris size
            trials[threshold] = PupilCalibration.iris_size(iris_frame)

        # threshold with the iris size closest to the average iris size
        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        # return the best threshold
        return best_threshold
    
    def evaluate(self, eye_frame):
        """
        Evaluates an eye frame and stores the best threshold value for the corresponding eye

        Parameters
        ----------
            eye_frame (numpy.ndarray): Frame of the eye
        """
        # determine the threshold for the given eye frame
        threshold = self.find_best_threshold(eye_frame)
        if LEFT_EYE == 0:
            self.thresholds_left.append(threshold)
        elif RIGHT_EYE == 1:
            self.thresholds_right.append(threshold)