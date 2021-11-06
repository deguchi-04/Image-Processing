from numpy.core.fromnumeric import cumprod
from skimage.filters.thresholding import threshold_otsu
import scipy.misc
import skimage.exposure as imexp
from PIL import Image
from numpy import asarray, histogram
import numpy as np
import matplotlib.pyplot as plt


def renyi_seg_fn(im, alpha):
    hist = imexp.histogram(im)
    #convert
    hist_float = [float(i) for i in hist[0]]
    #compute pdf
    pdf = hist_float/np.sum(hist_float)
    #compute cdf
    cumsum_pdf = np.cumsum(pdf)

    s = 0
    e = 255 #assuming 8 bit
    scalar = 1.0/(1-alpha)
    #a very small value to prevente division by 0
    eps = np.spacing(1)

    rr = e-s

    #need the 2nd parentheses cus the parameter is a tuple
    h1 = np.zeros((rr,1))
    h2 = np.zeros((rr,1))
    #the following loop computes h1 and h2 values used to compute entropy
    for ii in range(1,rr):
        iidash = ii+s
        temp1 = np.power(pdf[1:iidash]/cumsum_pdf[iidash], scalar)
        h1[ii] = np.log(np.sum(temp1)+eps)
        temp2 = np.power(pdf[iidash+1:255]/(1-cumsum_pdf[iidash]), scalar)
        h2[ii] = np.log(np.sum(temp2)+eps)
    
    T = h1 +h2
    #entropy value is calc
    T= -T*scalar

    #location where the max entropy occurs is when threshold for the renyi entropy 
    location = T.argmax(axis=0)
    #locatio value used as threshold

    thresh = location
    return thresh


#main
opn_img = Image.open('/home/thaidy/Documents/ML/standard_test_images/ex.jpg').convert('L')
img_vec = np.asarray(opn_img)

histogram, bin_edges = np.histogram(img_vec, bins=256, range=(-600, 600))
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([-10.0, 600.0])  # <- named arguments do not work here
plt.ylim([0.0, 200000.0])
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()


thresh = renyi_seg_fn(img_vec,3)

#compare
b = img_vec > thresh

vec_img = Image.fromarray(b)
vec_img.save('/home/thaidy/Desktop/output.jpg')
vec_img.show()

