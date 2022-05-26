import cv2
import numpy as np

# FACE DETECTION OPENCV2
faceCascade = cv2.CascadeClassifier("C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml")
def find_faces(images):
	# images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
	coordinates = locate_faces(images)
	cropped_faces = [images[y:y + h, x:x + w] for (x, y, w, h) in coordinates]
	normalized_faces = [normalized_face(face) for face in cropped_faces]
	# normalized_faces = normalized_face(cropped_faces)
	return zip(normalized_faces, coordinates)

def normalized_face(face):
	# face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	face = cv2.resize(face, (48, 48))
	return face

def locate_faces(images):
	faces = faceCascade.detectMultiScale(
		images,
		scaleFactor = 1.1,
		minNeighbors =6,
		minSize = (30,30)
	)
	return faces

# END FACE DETECTION