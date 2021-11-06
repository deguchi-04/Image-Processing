import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_otsu, threshold_local

from PIL import Image

image = Image.open('/home/thaidy/Documents/ML/standard_test_images/to.jpg').convert('L')


block_size = 35
binary_adaptive = threshold_local(image, block_size, offset=10)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')


ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()