import math, numpy
from numpy import asarray
from PIL import Image

a = Image.open('/home/thaidy/Desktop/black.jpg').convert('L')
b = asarray(a)
gamma = 5
b1 = b.astype(float)
b3 = numpy.max(b1)
b2 = b1/b3
b3 = numpy.log(b2)*gamma
c = numpy.exp(b3)*255.0
c1 = c.astype(int)
d = Image.fromarray((c1 * 255).astype(numpy.uint8))
d.save('/home/thaidy/Desktop/output.jpg')
d.show()
