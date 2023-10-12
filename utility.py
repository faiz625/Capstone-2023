import cv2

def draw_rectangle(frame, point1, point2, color, thickness):
    '''
    Draw a rectangle on an image frame.

    Parameters:
        frame (numpy.ndarray): The image frame to draw on.
        point1 (tuple): The coordinates of the top-left corner of the rectangle.
        point2 (tuple): The coordinates of the bottom-right corner of the rectangle.
        color (tuple): The BGR color of the rectangle (e.g., (0, 0, 255) for red).
        thickness (int): The thickness of the rectangle border.
    '''
    cv2.rectangle(frame, point1, point2, color, thickness)

def draw_circle(frame, center, radius, color, thickness):
    '''
    Draw a circle on an image frame.

    Parameters:
        frame (numpy.ndarray): The image frame to draw on.
        center (tuple): The coordinates of the circle's center.
        radius (int): The radius of the circle.
        color (tuple): The BGR color of the circle (e.g., (0, 0, 255) for red).
        thickness (int): The thickness of the circle's border.
    '''
    cv2.circle(frame, center, radius, color, thickness)
