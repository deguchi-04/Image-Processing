import scipy.misc
from PIL import Image
import numpy as np
from skimage import filters


#main
opn_img = Image.open('/home/thaidy/Documents/ML/standard_test_images/livro.jpg').convert('L')

img_vec = np.asarray(opn_img)

#The undocumented threshold_local doesn't return a binary image
b = filters.threshold_local(img_vec,39,offset=10)
#thats why i do that
c = img_vec > b

c = Image.fromarray(c).convert('RGB')
c.save('/home/thaidy/Desktop/output.jpg')
c.show()

