import scipy.misc
import scipy.fftpack as fftim
import math, numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#open image 
img = Image.open('/home/thaidy/Documents/ML/standard_test_images/ex.jpg').convert('L')
#convert in ndarray
b = np.asarray(img)

#performing FFT
c = fftim.fft2(b)

#shifting frequency to center
d = fftim.fftshift(c)

#inicializating variable for convolution
M = d.shape[0]
N = d.shape[1]

#H is defined and inicilized in 1
H = np.ones((M,N))
#A MxN image have center in M/2 and N/2
center1 = M/2
center2 = N/2
cut_radius = 30.0 #cut off radius
#Defining convolution function for ILPF

for i in range(1,M):
    for j in range(1,N):
        #euclidian distance from the origin is computed
        r1 = math.sqrt((i-center1)**2+(j-center2)**2)
    #using cut off radius to eliminate high frequency
        if 0< r1 < cut_radius:
            H[i,j] = 0.0
#converting H to image
H = Image.fromarray(H)
#perforrming conv
conv = d * H
#computing mag of FFT
e = abs(fftim.ifft2(conv)).astype(np.uint8)
#convert e to image
f = Image.fromarray(e)
#saving
f.save('/home/thaidy/Desktop/output.jpg')
f.show()

