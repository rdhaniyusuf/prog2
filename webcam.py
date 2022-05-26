import os
import cv2
import tensorflow
from tensorflow.keras.models import load_model
import joblib
import numpy as np

from lib.FaceDetection import find_faces
from lib.Prepocessing import video_equalizer
f = joblib.load('model/fisherface_training.pkl')
b = load_model('model/backprop_teman.h5')

emotions = ["Marah", "Jijik", "Takut", "Senang", "Sedih", "Terkejut", "Biasa saja"] 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
def proba(images=None):
	percent = np.array([i*0 for i in range(7)])
	predict = np.array([j for j in range(7)])
	
	if images is not None:
		images = video_equalizer(images)
		# images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
		_q = f.extract(images)
		_q = np.asarray(_q, dtype=np.float32)
		_p = b.predict(_q)

		percent = np.asarray([[_p[0][i] / np.sum(_p)] for i in range(len(_p[0]))]).reshape(1,-1)
		predict = np.argmax(_p, axis=1)
		return percent, predict
	
	return percent, predict

init = True

while init:
	rval, vFrame = cap.read()
	vFrame = cv2.flip(vFrame, 1)
	cv2image = cv2.cvtColor(vFrame, cv2.COLOR_RGB2GRAY)
	for normalized_face, (x,y, w, h) in find_faces(cv2image):
		_p, predict = proba(normalized_face)
		cv2.rectangle(cv2image, (x,y), (x+w, y+h), (255,255,255), 2)
		cv2.putText(cv2image,(str(emotions[predict[0]]) + str(": {:.2%}".format(float(_p[0][predict[0]])))), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2)		
	cv2.imshow('img', cv2image)
	if cv2.waitKey(30) & 0xff == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()