import tensorflow as tf
tf.config.list_physical_devices('GPU')

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
hertz = cap.get(cv2.CAP_PROP_FPS)
print(width, height, hertz)