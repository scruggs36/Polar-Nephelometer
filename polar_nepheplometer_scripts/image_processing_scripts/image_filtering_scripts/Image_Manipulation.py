import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as mi
import pandas as pd
import numpy as np
from scipy.misc.pilutil import Image


#directory navigation i.e. path to image
path1 = '/home/austen/PycharmProjects/Polar Nephelometer/Data/05-25-2017/Sum25Images_Covered.BMP'
path2 = '/home/austen/PycharmProjects/Polar Nephelometer/Data/05-25-2017/Sum25Images_LightLeakage.BMP'

#Image.open reads the BMP as an image format
im1 = Image.open(path1)
im2 = Image.open(path2)

#imread reads the image as a Matrix, the other a 2D array
imArray1 = mi.fromimage(im1)
imArray2 = mi.fromimage(im2)
imArrayDiff = imArray2 - imArray1


f, ax = plt.subplots(1,2)
ax[0].imshow(imArray2, cmap=cm.Greys_r)
ax[0].set_title('Holes Uncovered')
ax[1].imshow(imArray1, cmap=cm.Greys_r)
ax[1].set_title('Holes Covered')
plt.show()



