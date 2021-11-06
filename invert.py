import scipy.misc
import math
import numpy as np
from numpy import asarray
from PIL import Image

a = Image.open('/home/thaidy/Desktop/bird.jpg').convert('L')
a = asarray(a)
b = 255 - a
b = Image.fromarray(b)
b.save('/home/thaidy/Desktop/output.png')
b.show()
