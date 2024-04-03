import cv2
import os

def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    distance = (real_face_width * Focal_Length) / face_width_in_frame
    return distance

def clamp_value(x, max_value):
    if x < 0: return 0
    if x > max_value: return max_value
    return x

def capture_image(save_folder, frame):
    image_name = os.path.join(save_folder, "captured_image.jpg")
    cv2.imwrite(image_name, frame)