import scipy.misc
from PIL import Image
import numpy as np
from skimage import filters
from cv2 import *
import cv2

#main
opn_img = Image.open('/home/thaidy/Documents/ML/standard_test_images/livro.jpg').convert('L')

img_vec = np.asarray(opn_img)

c = cv2.adaptiveThreshold(img_vec, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 10)


#compare
c = Image.fromarray(c).convert('RGB')
c.save('/home/thaidy/Desktop/output.jpg')
c.show()

