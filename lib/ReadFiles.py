import numpy as np
import pandas as pd
import cv2
import os
from os.path import abspath
from inspect import getsourcefile

# read File
def read_file(categories,path):
	paths = os.path.dirname(abspath(getsourcefile(lambda:0)))
	X, y = [],[]
	for i, category in enumerate(categories):
		samples = os.listdir(os.path.join(path, category))
		samples = [s for s in samples if 'png' or 'jpg' or 'jpeg' in s]
		for sample in samples:
			img = cv2.imread(os.path.join(path, category,sample), cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (48,48))
			img = img.reshape([1, 48*48])
			X.append(img)
			y.append(i)
	return X, y

def read_csv(path):
	df = pd.read_csv(path)
	pixels = df.loc[:, 'pixels'].values
	labels = df.loc[:, 'emotion'].values
	X = []
	try :
		for pixel in pixels:
			pixel = [int(t) for t in pixel.split(',')]
			X.append(pixel)
	except ValueError:
		for pixel in pixels:
			pixel = [int(t) for t in pixel.split(' ')]
			X.append(pixel)		
	try : 
		usage = df.loc[:, 'Usage'].values
	except KeyError:
		return X, labels
	return X, labels, usage
# end read files