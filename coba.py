import os
from sys import flags
import cv2
import tensorflow
from tensorflow.keras.models import load_model
import joblib
import numpy as np

from lib.Prepocessing import video_equalizer
f = joblib.load('model/fisherface_training.pkl')
b = load_model('model/backprop_test.h5')

emotions = ["Marah", "Jijik", "Takut",
            "Senang", "Sedih", "Terkejut", "Biasa saja"]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceCascade = cv2.CascadeClassifier(
    "C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")


def proba(images=None):
    percent = np.array([i * 0 for i in range(7)])
    predict = np.array([j for j in range(7)])

    if images is not None:
        # images = video_equalizer(images)
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        _q = f.extract(images)
        _q = np.asarray(_q, dtype=np.float32)
        _p = b.predict(np.array(_q).reshape(len(_q, -1)))

        percent = np.asarray([[_p[0][i] / np.sum(_p)]
                             for i in range(len(_p[0]))]).reshape(1, -1)
        predict = np.argmax(_p, axis=1)
        return percent, predict

    return percent, predict


init = True

while init:
    rval, vFrame = cap.read()
    # vFrame = cv2.flip(vFrame, 1)
    cv2image = cv2.cvtColor(vFrame, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(
        cv2image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face = cv2image[y:y + h, x:x + w]
        face_rsz = cv2.resize(face, (48, 48))
        _p, predict = proba(face_rsz)
        cv2.rectangle(vFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(vFrame, (str(emotions[predict[0]]) + str(": {:.2%}".format(float(
            _p[0][predict[0]])))), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('img', vFrame)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
