import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as mi
import scipy.ndimage as ndim
import scipy.ndimage.filters as filters
import skimage.filters as fs
import numpy as np
from skimage import feature
from scipy.misc import pilutil as pu
from scipy.misc.pilutil import Image
from skimage import data, color, exposure, img_as_float, io
from skimage.feature import hog

#directory navigation i.e. path to image
path = '/home/austen/PycharmProjects/Polar Nephelometer/Data/05-10-2017/plots/Sum5Images.BMP'

#Image.open reads the BMP as an image format
im = Image.open(path).convert('L')

#imread reads the image as a Matrix, the other a 2D array
imMat = pu.imread(path)
imArray = mi.fromimage(im)

#Sobel filtering the image, looks decent for edge detection
sobel_filter_im = fs.sobel(im)
sobel_filter_imh = fs.sobel_h(im)
sobel_filter_imv = fs.sobel_v(im)
sobel_filter_im = mi.toimage(sobel_filter_im)
sobel_filter_imh = mi.toimage(sobel_filter_imh)
sobel_filter_imv = mi.toimage(sobel_filter_imv)
#sobel_filter_im.show()
'''
#scipy sobel filter
sx = filters.sobel(im, axis=0, mode='constant')
sy = filters.sobel(im, axis=1, mode='constant')
sob = np.hypot(sx, sy)
sobint = sob.astype(int)
'''

f, axx = plt.subplots(1, 3)
axx[1].imshow(sobel_filter_imh, cmap='gray')
axx[1].set_title('Sobel filter along X axis')
axx[2].imshow(sobel_filter_imv, cmap='gray')
axx[2].set_title('Sobel filter along Y axis')
axx[0].imshow(sobel_filter_im, cmap='gray')
axx[0].set_title('Sobel filter along X and Y axes')
plt.show()

#Prewitt filtering the image, looks decent for edge detection
prewitt_filter_im = fs.prewitt(im)
prewitt_filter_imh = fs.prewitt_h(im)
prewitt_filter_imv = fs.prewitt_v(im)
prewitt_filter_im = mi.toimage(prewitt_filter_im)
prewitt_filter_imh = mi.toimage(prewitt_filter_imh)
prewitt_filter_imv = mi.toimage(prewitt_filter_imv)


'''
#scipy prewitt filter
pw0 = filters.prewitt(im, axis=0, mode='reflect')
pw1 = filters.prewitt(im, axis=1, mode='reflect')
pw0 = mi.toimage(pw0)
pw1 = mi.toimage(pw1)
'''
#cmap = cm.Greys_r also yields grey scale images
f2, axx2 = plt.subplots(1, 3)
axx2[0].set_title('Prewitt Filtered Image')
axx2[0].imshow(prewitt_filter_im, cmap='gray')
axx2[1].set_title('Prewitt Filtered Image along X axis')
axx2[1].imshow(prewitt_filter_imh, cmap='gray')
axx2[2].set_title('Prewitt Filtered Image along Y axis')
axx2[2].imshow(prewitt_filter_imv, cmap='gray')
plt.show()





# Laplace filtering the image, bad pick for edge detection
laplace_filter_im = ndim.filters.laplace(im, mode='reflect')
laplace_filter_im = mi.toimage(laplace_filter_im)
#laplace_filter_im.show()

# Laplace of Gaussian filtering the image, interesting
log_filter_im = ndim.filters.gaussian_laplace(im, 0.1, mode='reflect')
log_filter_im = mi.toimage(log_filter_im)
#log_filter_im.show()

f3, axx3 = plt.subplots(1, 2)
axx3[0].imshow(laplace_filter_im, cmap='gray')
axx3[0].set_title('Laplace Filtered Image')
axx3[1].imshow(log_filter_im, cmap='gray')
axx3[1].set_title('Laplace of Gaussian Filtered Image')
plt.show()

# Contrast stretching, image enhancement
b = imArray.max()
a = imArray.min()
# convert image to type float
c = imArray.astype(float)
# contrast stretching here, im_enhanced is 2d array
imArray_enhanced = 255 * (c-a)/(b-a)
# convert to image
im_enhanced = mi.toimage(imArray_enhanced)
#im_enhanced.show()

f4, axx4 = plt.subplots()
axx4.imshow(im_enhanced, cmap='gray')
axx4.set_title('Contrast enhancement')
plt.show()

# Histogram of Oriented Gradients, code works but the result is not helpful
fd, hog_image = hog(im, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(im, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
'''
# We will try by passing the otsu filtered image to a Canny filter
# Canny filtering the otsu filtered image
canny_filter_im_edge = feature.canny(otsu_imArray, sigma=1.0)
canny_filter_img = mi.toimage(canny_filter_im_edge)
canny_filter_img.show()


# this method of edge detection is far too tedious to work effectively and doesn't seem to be automatically reproducible
# this is due to the fact that i have to pick the bright points i want to use to fit
# i might as well fit two identical polynomials to the center part of each beam and then 
# adjust their y values up and down to bind the laser beam


# It looks like we found our edges in a rough manner, now we need to get these pixel locations!
ans = []
for y in range(canny_filter_im_edge.shape[0]):
    for x in range(canny_filter_im_edge.shape[1]):
        if canny_filter_im_edge[y, x] != 0:
            ans = ans + [[x, y]]

ans = np.array(ans)
print(ans.shape)

# now we have the coordinates for the edges in one big array, we need to divide this array into four arrays
# that bind the top and bottom laser beams, to do that we need to change the color at these pixels in a duplicate image
# and see what we cut out of the edges
alpha = 0.6
Edges = canny_filter_im_edge
Edges = img_as_float(Edges)
rows, cols = Edges.shape
color_mask = np.zeros((rows, cols, 3))

# gonna try to make a red edge mask, seems to be working
# here the x index governs the columns and thus acts as x indexer, and the y index governs rows, so it acts as a y indexer
xindx = ans[:, 1]
yindx = ans[:, 0]

# currently we are going to slince xindx and y indx until we get a better mask
# here is the top edge of the top beam
F_top_xindx = xindx[0:2500]
F_top_yindx = yindx[0:2500]
# this loop sets the mask color to be red
for counter, element in enumerate(F_top_xindx):
    X = F_top_xindx[counter]
    Y = F_top_yindx[counter]
    color_mask[X, Y] = [1, 0, 0]
# here is the bottom edge of the top beam
F_bot_xindx = np.concatenate((xindx[2890:3000], xindx[3300:3700], xindx[3800:5000]), axis=0)
F_bot_yindx = np.concatenate((yindx[2890:3000], yindx[3300:3700], yindx[3800:5000]), axis=0)
# this loop sets the mask color to be green
for counter, element in enumerate(F_bot_xindx):
    X = F_bot_xindx[counter]
    Y = F_bot_yindx[counter]
    color_mask[X, Y] = [0, 1, 0]
# here is the top edge of the bottom beam xindx[8750:8850], xindx[8950:9150], xindx[9250:9350]
B_top_xindx = np.concatenate((xindx[7000:7500], xindx[7900:8100], xindx[8800:8900]), axis=0)
B_top_yindx = np.concatenate((yindx[7000:7500], yindx[7900:8100], yindx[8800:8900]), axis=0)
# this loop sets the mask color to be blue
for counter, element in enumerate(B_top_xindx):
    X = B_top_xindx[counter]
    Y = B_top_yindx[counter]
    color_mask[X, Y] = [0, 0, 1]
# these loops just helped me figure out why i was getting some mask errors
# these loops are not important

#these for loops were tests, they are not important 
for counter, elementx in enumerate(xindx):
    if elementx > 1279:
        np.delete(xindx, counter)
        np.delete(yindx, counter)
for counter, elementy in enumerate(yindx):
    if elementy > 1024:
        np.delete(xindx, counter)
        np.delete(yindx, counter)


# make RBG version of grayscale image
Edges_color = np.dstack((Edges, Edges, Edges))

# convert input image and color mask to Hue Saturation Value
Edges_hsv = color.rgb2hsv(Edges_color)
color_mask_hsv = color.rgb2hsv(color_mask)

# replace hue and saturation of original image with that of the color mask
Edges_hsv[..., 0] = color_mask_hsv[..., 0]
Edges_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

Edges_masked = color.hsv2rgb(Edges_hsv)

# Display the output
f, (ax0, ax1, ax2) = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []})
ax0.imshow(Edges, cmap=plt.cm.gray)
ax1.imshow(color_mask)
ax2.imshow(Edges_masked)
plt.show()
'''
