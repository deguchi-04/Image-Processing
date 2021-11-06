import numpy as np
import scipy.misc, math
from PIL import Image


img = Image.open('/home/thaidy/Desktop/ex.jpg').convert('L')
#converting to ndarray
img1 = np.asarray(img)

#converting to 1D
fl = img1.flatten()

#histogram and the bins are computed
hist, bins = np.histogram(img1,256,[0,255],density=True)


#cdf computed
cdf = hist.cumsum()

#places where cdf = 0 is ignored
#rest stored in cdf_m
cdf_m = np.ma.masked_equal(cdf,0)

#histogram eq is performed
T = (255 * cdf).astype(np.int8)
#cdf values assigned in the flattened array
im2 = T[fl]
#transformin in 2D
im3 = np.reshape(im2,img1.shape)
im4 = Image.fromarray(im3).convert('RGB')
im4.save('/home/thaidy/Desktop/output.jpg')

im4.show()
