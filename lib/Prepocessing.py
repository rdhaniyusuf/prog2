# # fungsi equalizer preprosessing
import numpy as np

def image_equalizer(images, labels):
    new_images = []
    for i in range(len(images)):
        img = np.asarray(images[i]).reshape(48, 48)
        histo = np.histogram(img, 256, (0, 255))[0]
        cdf = histo.cumsum()
        cdf_eq = (256-1)*(cdf/float(2304))
        cq = cdf_eq[img]
        new_images.append(np.asarray(cq, dtype=np.uint8).reshape(1, 2304))
    return new_images, labels

def video_equalizer(images):
	img = np.asarray(images).reshape(48, 48)
	histo = np.histogram(img, 256, (0, 255))[0]
	cdf = histo.cumsum()
	cdf_eq = (256-1)*(cdf/float(2304))
	cq = cdf_eq[img]
	return cq
