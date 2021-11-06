import math, numpy
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

from numpy import asarray
from PIL import Image

a = Image.open('/home/thaidy/Documents/ML/standard_test_images/mandril_color.tif').convert('L')
b = asarray(a)



b1 = b.astype(float)
b2 = numpy.max(b1)
c = (255.0*numpy.log(1/b1))/numpy.log(1/b2)
c1 = c.astype(int)
d = Image.fromarray((c1 * 255).astype(numpy.uint8))
d.save('/home/thaidy/Desktop/output.jpg')
d.show()
