import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as mi
import scipy.ndimage as ndimage
import skimage.filters as fs
import numpy as np
import numpy.polynomial.polynomial as poly
from skimage import feature
from scipy.misc import pilutil as pu
from scipy.misc.pilutil import Image
from skimage import exposure
from scipy.optimize import curve_fit
from skimage import color, io, img_as_float

#directory navigation i.e. path to image
path_windows = 'C:/Users/sm2/Documents/Github Repository Clone/Polar-Nephelometer/Data/06-19-2017/Image30s50mW.BMP'
path_linux = '/home/austen/PycharmProjects/TSI-3563-INeph/Data/06-19-2017/Image30s50mW.BMP'

#Image.open reads the BMP as an image format
im = Image.open(path_windows)

#imread reads the image as a Matrix, the other a 2D array
imMat = pu.imread(path_windows)
imArray = mi.fromimage(im)


# Otsu method for thresholding automatically finds the value that maximizes the variance between the background and foreground
# using skimages module we will conduct otsu thresholding
# I have included a variable to tune up or down the thresholding as I please
val_variable = -40
val = fs.threshold_otsu(imArray) + val_variable
hist, bins_center = exposure.histogram(imArray)
otsu_imArray = imArray > val
otsu_im = mi.toimage(otsu_imArray)


# Create figure showing otsu thresholding based upon raw image histogram
f, ax = plt.subplots(1,3)
ax[0].imshow(imArray, cmap='gray', interpolation='nearest')
ax[0].set_title('Raw Image')
ax[1].imshow(imArray < val, cmap='gray', interpolation='nearest')
ax[1].set_title('otsu filtered image')
ax[2].plot(bins_center, hist, lw=2)
ax[2].axvline(val, color='k', ls='--')
ax[2].set_title('Histogram: Number of Pixels that Fall into Intensity Bins 0-255')
plt.show()


# Can we find the contour pixel positions of the otsu filtered image edges?
# approach 1: Canny filtering the otsu filtered image
canny_filter_im_edge = feature.canny(otsu_imArray, sigma=1.0)
canny_filter_img = mi.toimage(canny_filter_im_edge)
#canny_filter_img.show()

f1, ax1 = plt.subplots()
ax1.imshow(canny_filter_img, cmap='gray')
ax1.set_title('Canny Filtered Otsu Threshold Image')
plt.show()

# approach 2 laplace filtering the otsu filtered image
radius = 10
distance_img = ndimage.distance_transform_edt(otsu_imArray)
morph_laplace_img = ndimage.morphological_laplace(distance_img, (radius, radius))
skeleton = morph_laplace_img < morph_laplace_img.min()/2

# figure showing effectiveness of Otsu-Laplace filtered image
f2, ax = plt.subplots(1, 3)
ax[0].imshow(imArray, cmap=cm.Greys_r)
ax[0].set_title('Raw Image')
ax[0].set_xlabel('X Pixel Coordinates')
ax[0].set_ylabel('Y Pixel Coordinates')
ax[1].imshow(otsu_imArray, cmap=cm.Greys_r)
ax[1].set_title('Otsu Filtered Image')
ax[1].set_xlabel('X Pixel Coordinates')
ax[1].set_ylabel('Y Pixel Coordinates')
ax[2].imshow(skeleton, cmap=cm.Greys_r)
ax[2].set_title('Otsu-Laplace Filtered Image')
ax[2].set_xlabel('X Pixel Coordinates')
ax[2].set_ylabel('Y Pixel Coordinates')
plt.tight_layout()
plt.show()


# Lets try some adaptive thresholding, its also called local thresholding


'''
#THIS IS ALL CURVE FITTING TO THE CENTER OF THE BEAM, THE APROACH IS TOTALLY WRONG
# The process below is concerned with obtaining the coordinates we care about
# 1. we divided the image in half to obtain seperate images of the otsu-laplace filtered top and bottom beams
skeleton1 = skeleton[0:511,:]
skeleton2 = skeleton[512:1023,:]
# check to see if we divided the image in half properly, success!
#plt.imshow(skeleton1, cmap=cm.Greys_r)
#plt.show()
#plt.imshow(skeleton2, cmap=cm.Greys_r)
#plt.show()

# 2. we wrote a double for loop to obtain the coordinates for the top and bottom images
ans1 = []
ans2 = []
delarray1 = []
delarray2 = []
for y in range(skeleton1.shape[0]):
    for x in range(skeleton1.shape[1]):
        if skeleton1[y, x] != 0:
            ans1 = ans1 + [[x, y]]

for y in range(skeleton2.shape[0]):
    for x in range(skeleton2.shape[1]):
        if skeleton2[y, x] != 0:
            ans2 = ans2 + [[x, y+512]]
# here we ended up shaving off points due to noise, we did this by
# setting a upper and lower bound on the points along the curve
for item in range(np.shape(ans1)[0]):
    if ans1[item][1] > 450:
        delarray1.append(item)

for item in range(np.shape(ans2)[0]):
    if ans2[item][1] > 900:
        delarray2.append(item)

for item in range(np.shape(ans2)[0]):
    if ans2[item][1] < 640:
        delarray2.append(item)

ans1 = np.delete(ans1, delarray1, 0)
ans2 = np.delete(ans2, delarray2, 0)



# This test shows that the coordinate information is preserved!
# The arrays it is numerically ordered by the column
ans1 = np.array(ans1)
ans1 = ans1[np.argsort(ans1[:,0])]
ans2 = np.array(ans2)
ans2 = ans2[np.argsort(ans2[:,0])]


yindx1 = ans1[:, 1]
xindx1 = ans1[:, 0]
yindx2 = ans2[:, 1]
xindx2 = ans2[:, 0]

# now, how can we fit a curve through these two dimensional coordinates?
# i think if we can fit them, we can model z, fix x to be an array range(1280) and solve for y
# doing this we can get the x,y coorinates for pixels we care about


# how do you fit data with 2 independent variables?

def func(y, a, b, c):
    return (a*y**2) + (b*y) + c

popt1, pcov1 = curve_fit(func, yindx1, xindx1)
popt2, pcov2 = curve_fit(func, yindx2, xindx2)
xarray1T = []
xarray1B = []
xarray2T = []
xarray2B = []

f3, ax = plt.subplots(1, 3)
ax[0].imshow(imArray, cmap=cm.Greys_r)
ax[0].set_title('Raw Image')
ax[0].set_xlabel('X Pixel Coordinates')
ax[0].set_ylabel('Y Pixel Coordinates')
ax[1].imshow(otsu_imArray, cmap=cm.Greys_r)
ax[1].set_title('Otsu Filtered Image')
ax[1].set_xlabel('X Pixel Coordinates')
ax[1].set_ylabel('Y Pixel Coordinates')
ax[1].plot(func(yindx1, *popt1), yindx1, 'r--', label='top beam fit')
ax[1].plot(func(yindx2, *popt2), yindx2, 'b--', label='bottom beam fit')
ax[2].plot(xindx1, yindx1, 'b-', label='top raw xy indices')
ax[2].plot(func(yindx1, *popt1), yindx1, color='orange', label='top beam fit')
ax[2].plot(xindx2, yindx2, 'g-', label='bottom raw xy indices')
ax[2].plot(func(yindx2, *popt2), yindx2, color='aqua', label='bottom beam fit')
ax[2].set_title('X and Y indices and fit')
ax[2].set_xlabel('X Pixel Coordinates')
ax[2].set_ylabel('Y Pixel Coordinates')
ax[2].invert_yaxis()
plt.tight_layout()
plt.show()

'''



