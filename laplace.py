import scipy.misc
import scipy.ndimage
from numpy import asarray
from skimage.feature import canny
from PIL import Image

a = Image.open('/home/thaidy/Desktop/easter.png').convert('L')
b = scipy.ndimage.filters.laplace(a,mode='reflect')
b = Image.fromarray(b)
b.save('/home/thaidy/Desktop/output.png')
b.show()
