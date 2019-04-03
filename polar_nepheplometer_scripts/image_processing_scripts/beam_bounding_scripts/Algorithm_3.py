from Neph_Functions import Image_Subtract, Image_Heatmaps, Profiles, Curve_Fit_Profiles, Curve_Fit_Profiles2, gaussian, quartic, quadratic, linear
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import curve_fit
from skimage.exposure import histogram
import math


'''
This code evaluates a single image between the rows given in the initial boundaries section
'''
# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_Image = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/06-12-2018/N2/N2_90s_Analysis/im_avg_N2_90s.png'
#Path_BKG = '/home/austen/mounts/chem/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/05-09-2018/N2_Analysis/im_avg_N2_6s.png'
Path_PF = '/home/austen/Documents/PF.txt'
Path_HM = '/home/austen/Documents/HM.png'
# scipy.misc package mi.imread reads the BMP as an image format (could use cv2 as well, see version 1)
im = cv2.imread(Path_Image, 0)
plt.imshow(im, cmap='gray')
plt.show()
#imBKG = cv2.imread(Path_BKG, 0)

# Subtract Background
#im = Image_Subtract(im, imBKG)

# Initial boundaries on the image
col_boundary_1 = 300
col_boundary_2 = 600
row_boundary_1 = 400
row_boundary_2 = 480
numrows = row_boundary_2 - row_boundary_1
numcols = col_boundary_2 - col_boundary_1


# calling function to draw vertical profiles
#Image_Heatmaps(imBKG, im, 255, Path_HM)


rows_DF, cols_DF, int_DF = Profiles(im, col_boundary_1, col_boundary_2, numrows, numcols, row_boundary_1, row_boundary_2)
fitxdata1, fitydata1, fyd1, ft, bt, mt, st, bkgt, aug_array, aug_array_norm, tt_smooth, tb_smooth, col_array = Curve_Fit_Profiles(gaussian, quartic, rows_DF, int_DF, cols_DF, 50)
#print(mt)


# here we are plotting the fit data
f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im, cmap='gray')
ax1[0].plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1[0].plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1[0].plot(cols_DF, mt, 'rX', ms=0.4)
ax1[1].set_title('Fits to Top Profiles')
ax1[1].set_xlabel('Y Coordinate')
ax1[1].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1):
    ax1[1].plot(fitxdata1[counter], fyd1[counter])
plt.show()


f1a, ax1a = plt.subplots()
ax1a.imshow(im, cmap='gray')
ax1a.plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1a.plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1a.plot(cols_DF, mt, 'rX', ms=0.4)
plt.show()

'''
f2, ax2 = plt.subplots()
ax2.plot(cols_DF, aug_array_norm, 'r-', label='Phase Function')
ax2.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax2.set_xlabel('Column Profile Number')
ax2.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.show()
'''

f3, ax3 = plt.subplots()
ax3.plot(cols_DF, aug_array, 'b-', label='Phase Function')
ax3.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax3.set_xlabel('Column Profile Number')
ax3.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.show()


#'''
#Save Phase Function
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Profile Number'] = cols_DF
PhaseFunctionDF['Intensity'] = aug_array
PhaseFunctionDF['Normalized Intensity'] = aug_array_norm
PhaseFunctionDF.to_csv(Path_PF)
#'''