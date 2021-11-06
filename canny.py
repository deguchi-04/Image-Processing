import scipy.misc
import scipy.misc, numpy
from numpy import asarray
from skimage.feature import canny
from PIL import Image

a = Image.open('/home/thaidy/Desktop/letra.jpg').convert('L')
a = asarray(a)
b = canny(a, sigma=1.0)
b = Image.fromarray(b)
b.save('/home/thaidy/Desktop/output.png')
b.show()
