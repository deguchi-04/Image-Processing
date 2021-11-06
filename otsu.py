from skimage.filters.thresholding import threshold_otsu
import scipy.misc
from PIL import Image
from numpy import asarray
import numpy as np

opn_img = Image.open('/home/thaidy/Documents/ML/standard_test_images/livro.jpg').convert('L')

img_vec = np.asarray(opn_img)

thresh = threshold_otsu(img_vec)

#compare
b = img_vec > thresh

vec_img = Image.fromarray(b)
vec_img.save('/home/thaidy/Desktop/output.jpg')
vec_img.show()