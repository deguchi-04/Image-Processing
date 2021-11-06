import numpy as np
import scipy.misc, math
from PIL import Image


img = Image.open('/home/thaidy/Downloads/standard_test_images/woman_darkhair.tif').convert('L')
#converting to ndarray
im1 = np.asarray(img)

#converting to 1D
b = im1.max()
a = im1.min()
print(a,b)
c = im1.astype(float)


im2 = 255*(c-a)/(b-a)


im3 = Image.fromarray(im2).convert('RGB')
im3.save('/home/thaidy/Desktop/output.tif')
im3.show()
