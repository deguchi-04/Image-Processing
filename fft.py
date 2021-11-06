import numpy as np
import scipy.fftpack as fftim
import math
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('/home/thaidy/Documents/ML/standard_test_images/to.jpg').convert('L')

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

img1 = np.asarray(img)
c = abs(fftim.fft2(img1))
d = fftim.fftshift(c)
e = fftim.ifftshift(d)
f = fftim.ifft2(e)

plt.subplot(151), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(152), plt.imshow(np.log(1+c), "gray"), plt.title("Spectrum")
plt.subplot(153), plt.imshow(np.log(1+d), "gray"), plt.title("Centered Spectrum")
plt.subplot(154), plt.imshow(np.log(1+e), "gray"), plt.title("Decentralized")
plt.subplot(155), plt.imshow(np.abs(f), "gray"), plt.title("Processed Image")
d.astype('float').tofile('/home/thaidy/Desktop/output.raw')

plt.show()