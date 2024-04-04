import mediapipe as mp
import pyautogui
import time
import winsound


class Clicker:
	def __init__(self):
		self._mp_face_mesh = mp.solutions.face_mesh
		self._blink_count = 0
		self._eye_closed_start_time = None
		self._reset_interval = 5  
		self._blink_threshold = 0.1 
		self._start_time = time.time()

	def perform_mouse_action(self):
		if self._blink_count == 2:
			print("Performing a left click")
			pyautogui.click(button='left')
			winsound.Beep(2000, 200)
		elif self._blink_count == 3:
			print("Performing a right click")
			pyautogui.click(button='right')
			winsound.Beep(3000, 200)
		elif self._blink_count == 4:
			print("Performing a double left click")
			pyautogui.doubleClick(button='left')
			winsound.Beep(2000, 100)
			winsound.Beep(2000, 100)

	def clickLoop(self, frame):
		with self._mp_face_mesh.FaceMesh(
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		) as face_mesh:
			output = face_mesh.process(frame)
			landmark_points = output.multi_face_landmarks

			if landmark_points:
				landmarks = landmark_points[0].landmark
				left = [landmarks[145], landmarks[159]]

				if (left[0].y - left[1].y) < 0.01:
					if self._eye_closed_start_time is None:
						self._eye_closed_start_time = time.time()  
				else:
					if self._eye_closed_start_time:
						if time.time() - self._eye_closed_start_time >= self._blink_threshold:
							self._blink_count += 1
							print(f"Blink detected. Current count: {self._blink_count}")
						self._eye_closed_start_time = None  

				if time.time() - self._start_time > self._reset_interval:
					self.perform_mouse_action()
					self._blink_count = 0 
					self._start_time = time.time()  
					winsound.Beep(1000, 200)   
