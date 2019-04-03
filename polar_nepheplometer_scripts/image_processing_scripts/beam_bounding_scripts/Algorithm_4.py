from Neph_Functions import Image_Subtract, Image_Heatmaps, Profiles, Curve_Fit_Profiles, gaussian, quadratic, linear
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import curve_fit
from skimage.exposure import histogram
import math

'''
Algorithm written for forward and backward light scattering analysis
'''

# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_Image = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/03-22-2018/N2/Analysis/After/im_avg_N2_30s.png'
Path_BKG = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/03-22-2018/N2/Analysis/After/im_avg_N2_30s.png'
Path_PF = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/03-22-2018/N2/Analysis/After/PF_'
Path_HM = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/03-22-2018/N2/Analysis/After/HM.png'
# scipy.misc package mi.imread reads the BMP as an image format (could use cv2 as well, see version 1)
im = cv2.imread(Path_Image, 0)
plt.imshow(im, cmap='gray')
plt.show()
imBKG = cv2.imread(Path_BKG, 0)

# Subtract Background
#im = Image_Subtract(im, imBKG)

# Initial boundaries on the top beam in the image
col_boundary_1t = 200
col_boundary_2t = 700
row_boundary_1t = 300
row_boundary_2t = 575
numrowst = row_boundary_2t - row_boundary_1t
numcolst = col_boundary_2t - col_boundary_1t
Gap_Param = col_boundary_2t + (1279 - col_boundary_2t) + (1279 - col_boundary_2t)
# Initial boundaries on the bottom beam in the image
col_boundary_1b = 200
col_boundary_2b = 700
row_boundary_1b = 550
row_boundary_2b = 775
numrowsb = row_boundary_2b - row_boundary_1b
numcolsb = col_boundary_2b - col_boundary_1b


# calling function to draw vertical profiles
#Image_Heatmaps(imBKG, im, 255, Path_HM)

#top beam profiling and curve fitting
rows_DF_t, cols_DF_t, int_DF_t = Profiles(im, col_boundary_1t, col_boundary_2t, numrowst, numcolst, row_boundary_1t, row_boundary_2t)
fitxdata1_t, fitydata1_t, fyd1_t, ft_t, bt_t, mt_t, st_t, bkg_t, aug_array_t, aug_array_norm_t, tt_smooth_t, tb_smooth_t = Curve_Fit_Profiles(gaussian, linear, rows_DF_t, int_DF_t, cols_DF_t)
# bottom beam profiling and curve fitting
rows_DF_b, cols_DF_b, int_DF_b = Profiles(im, col_boundary_1b, col_boundary_2b, numrowsb, numcolsb, row_boundary_1b, row_boundary_2b)
fitxdata1_b, fitydata1_b, fyd1_b, ft_b, bt_b, mt_b, st_b, bkg_b, aug_array_b, aug_array_norm_b, tt_smooth_b, tb_smooth_b = Curve_Fit_Profiles(gaussian, linear, rows_DF_b, int_DF_b, cols_DF_b)
#print(mt_t)


# here we are plotting the fit data
f1, ax1 = plt.subplots(1, 3)
ax1[0].imshow(im, cmap='gray')
ax1[0].plot(cols_DF_t, tt_smooth_t, color='lime', marker='.', ms=0.2)
ax1[0].plot(cols_DF_t, tb_smooth_t, color='orange', marker='.', ms=0.2)
ax1[0].plot(cols_DF_t, mt_t, 'rX', ms=0.4)
ax1[0].plot(cols_DF_b, tt_smooth_b, color='#FF007F', marker='.', ms=0.2)
ax1[0].plot(cols_DF_b, tb_smooth_b, color='#0000FF', marker='.', ms=0.2)
ax1[0].plot(cols_DF_b, mt_b, 'rX', ms=0.4)
ax1[1].set_title('Fits to Top Profiles')
ax1[1].set_xlabel('Y Coordinate')
ax1[1].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1_t):
    ax1[1].plot(fitxdata1_t[counter], fyd1_t[counter])
ax1[2].set_title('Fits to Top Profiles')
ax1[2].set_xlabel('Y Coordinate')
ax1[2].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1_b):
    ax1[2].plot(fitxdata1_b[counter], fyd1_b[counter])
plt.show()

# image bounds solo
f1a, ax1a = plt.subplots()
ax1a.imshow(im, cmap='gray')
ax1a.plot(cols_DF_t, tt_smooth_t, color='lime', marker='.', ms=0.2)
ax1a.plot(cols_DF_t, tb_smooth_t, color='orange', marker='.', ms=0.2)
ax1a.plot(cols_DF_t, mt_t, 'rX', ms=0.4)
ax1a.plot(cols_DF_b, tt_smooth_b, color='#FF007F', marker='.', ms=0.2)
ax1a.plot(cols_DF_b, tb_smooth_b, color='#0000FF', marker='.', ms=0.2)
ax1a.plot(cols_DF_b, mt_b, 'rX', ms=0.4)
plt.show()

# image gaussian fits solo
f1b, ax1b = plt.subplots(1, 2)
ax1b[0].set_title('Fits to Top Profiles')
ax1b[0].set_xlabel('Y Coordinate')
ax1b[0].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1_t):
    ax1b[0].plot(fitxdata1_t[counter], fyd1_t[counter])
ax1b[1].set_title('Fits to Top Profiles')
ax1b[1].set_xlabel('Y Coordinate')
ax1b[1].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1_b):
    ax1b[1].plot(fitxdata1_b[counter], fyd1_b[counter])
plt.show()


# PF_N
f2, ax2 = plt.subplots()
# you would think this would be the x axis: np.append(cols_DF_t, cols_DF_b)
# however it cannot be that due to the fact the numbering restarts!
# thus we must make an array of coordinates for the x axis!
# in this plot we just use the transect number as the X axis... we need better treatment here
# [::-1] reverses the order of the array (flips the ordering)
ax2.plot(np.append(cols_DF_t, (np.asarray(cols_DF_b) + Gap_Param)), np.append(aug_array_norm_t, aug_array_norm_b[::-1]), 'r.', label='Phase Function')
ax2.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax2.set_xlabel('Column Profile Number')
ax2.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.show()

# PF_NN
f3, ax3 = plt.subplots()
# in this plot we just use the transect number as the X axis... we need a better treatment here
ax3.plot(np.append(cols_DF_t, np.asarray(cols_DF_b) + Gap_Param), np.append(aug_array_t, aug_array_b[::-1]), 'b.', label='Phase Function')
ax3.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax3.set_xlabel('Column Profile Number')
ax3.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.show()

#'''
#Save Phase Function
PhaseFunctionDF_top = pd.DataFrame()
PhaseFunctionDF_bot = pd.DataFrame()
# here we store the coordinates of each of the transects taken , using zip could also work but we used numpy to do it
PhaseFunctionDF_top['Top Beam Column Number'] = cols_DF_t
PhaseFunctionDF_top['Top Beam Row Numbers'] = rows_DF_t
PhaseFunctionDF_top['Top Beam Intensities'] = aug_array_t
PhaseFunctionDF_top['Top Beam Intensities Normalized'] = aug_array_norm_t
PhaseFunctionDF_top.to_csv(Path_PF +'top.txt')
PhaseFunctionDF_bot['Bottom Beam Column Number'] = cols_DF_b
PhaseFunctionDF_bot['Bottom Beam Row Numbers'] = rows_DF_b
PhaseFunctionDF_bot['Bottom Beam Intensities'] = aug_array_b[::-1]
PhaseFunctionDF_bot['Bottom Beam Intensities Normalized'] = aug_array_norm_b[::-1]
#print(PhaseFunctionDF)
PhaseFunctionDF_bot.to_csv(Path_PF+'bottom.txt')
#'''