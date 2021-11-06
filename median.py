import scipy.misc
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
a = Image.open('/home/thaidy/Documents/ML/standard_test_images/lena_gray_256.tif').convert('L')

b = scipy.ndimage.filters.median_filter(a, size = 5, footprint= None, output= None, mode='reflect',cval=0.0,origin=0)
b = Image.fromarray(b)
b.save('/home/thaidy/Desktop/output.png')
b.show()
