import cv2
import numpy as np

# Define frame dimensions
frame_width = 1920
frame_height = 1080

# Create a black frame
frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Define parameters for circles
num_rows = 5
num_cols = 5
circle_radius = min(frame_width, frame_height) // (2 * max(num_rows, num_cols))
circle_color = (0, 255, 0)  # Green color
circle_thickness = 2

# Calculate spacing between circles
horizontal_spacing = (frame_width - 2 * circle_radius) // (num_cols - 1)
vertical_spacing = (frame_height - 2 * circle_radius) // (num_rows - 1)

# Draw circles
for i in range(num_rows):
    for j in range(num_cols):
        center = (j * horizontal_spacing + circle_radius, i * vertical_spacing + circle_radius)
        print(center)
        cv2.circle(frame, center, circle_radius, circle_color, circle_thickness)

# Display the frame
cv2.imshow('Frame with Circles', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()