import cv2
from PIL import Image
import numpy as np
import scipy.misc 
from scipy.ndimage import label
from scipy.ndimage.morphology import distance_transform_bf
#from skimage.morphology import label


#main
opn_img = cv2.imread('/home/thaidy/Documents/ML/standard_test_images/mandril_color.tif')

a1 = cv2.cvtColor(opn_img, cv2.COLOR_BGR2GRAY)

#obtain cell pixels
thresh, b1 = cv2.threshold(a1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#erode the image cus otsu has over segmented the image

b2 = cv2.erode(b1, None, iterations=2)

#distance transform

dist_trans = cv2.distanceTransform(b2, 2, 3)

#thresholding the distance transform image to obtain pixels that are foreground

thresh, dt = cv2.threshold(dist_trans, 1, 255, cv2.THRESH_BINARY)

#labeling
##### labelled = label(b, background =0)
labelled, ncc = label(dt)
#convert 32bit integer
labelled.astype(np.int32)

#watershed
cv2.watershed(opn_img,labelled)
#convert ndarray to image
dt1 = Image.fromarray(labelled)
dt1.save('/home/thaidy/Desktop/output.png')
dt1.show()

