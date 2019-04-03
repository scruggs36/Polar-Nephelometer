from Neph_Functions import Perp_Profiles, Image_Subtract, Image_Heatmaps, Profiles, Curve_Fit_Profiles, gaussian, quadratic, linear, loop_simps_riemann
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
Path_Image = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/06-12-2018/N2_3LPM_DMAOFF_Analysis/im_avg_N2_3LPM_DMAOFF_30s.png'
#Path_BKG = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/06-08-2018/N2_Analysis/im_avg_N2_30s.png'
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
col_boundary_2 = 1050
row_boundary_1 = 400
row_boundary_2 = 600
numrows = row_boundary_2 - row_boundary_1
numcols = col_boundary_2 - col_boundary_1


# calling function to draw vertical profiles
#Image_Heatmaps(imBKG, im, 255, Path_HM)


rows_DF, cols_DF, int_DF = Profiles(im, col_boundary_1, col_boundary_2, numrows, numcols, row_boundary_1, row_boundary_2)
fitxdata1, fitydata1, fyd1, ft, bt, mt, st, bkgt, aug_array, aug_array_norm, tt_smooth, tb_smooth, CDF = Curve_Fit_Profiles(gaussian, quadratic, rows_DF, int_DF, cols_DF)
# CDF is already a 1d array of column coordinates, sweet!
z, tt_coords, tb_coords, ts, BES, PES, x_pts, y_pts = Perp_Profiles(im, tt_smooth, tb_smooth, cols_DF, numcols)



# here we are plotting the fit data
f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im, cmap='gray')
ax1[0].plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1[0].plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1[0].plot(cols_DF, mt, 'rX', ms=0.4)
ax1[1].set_title('Fits to Top Profiles')
ax1[1].set_xlabel('Y Coordinate')
ax1[1].set_ylabel('Intensity')
ax1[1].grid(True)
for counter, row in enumerate(fitxdata1):
    ax1[1].plot(fitxdata1[counter], fyd1[counter])
plt.show()


f1a, ax1a = plt.subplots()
ax1a.imshow(im, cmap='gray')
ax1a.plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1a.plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1a.plot(cols_DF, mt, 'rX', ms=0.4)
plt.show()


f2, ax2 = plt.subplots()
ax2.plot(cols_DF, aug_array_norm, 'r-', label='Phase Function')
ax2.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax2.set_xlabel('Column Profile Number')
ax2.set_ylabel('Integrated Intensity')
ax2.grid(True)
plt.legend(loc=1)
plt.show()


f3, ax3 = plt.subplots()
ax3.plot(cols_DF, aug_array, 'b-', label='Phase Function')
ax3.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax3.set_xlabel('Column Profile Number')
ax3.set_ylabel('Integrated Intensity')
ax3.grid(True)
plt.legend(loc=1)
plt.show()
'''
# test case for perpendicular transects with limited resolution
f4, ax4 = plt.subplots(1, 2)
ax4[0].imshow(im, cmap='gray')
ax4[0].plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax4[0].plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax4[0].plot(cols_DF, mt, 'rX', ms=0.4)
ax4[1].set_title('Profiles')
ax4[1].set_xlabel('Y Coordinate')
ax4[1].set_ylabel('Intensity')
ax4[1].grid(True)
for element in np.linspace(0, 1200, 100):
        ax4[0].plot(x_pts[int(element)], y_pts[int(element)], color='yellow', marker='.', ms=0.2)
        ax4[1].plot(z[int(element)])
plt.show()


# test phase function for perpendicular transects
z_trunc = []
for element in np.linspace(0, 1200, 100):
    z_trunc.append(z[int(element)])
simps, riemann = loop_simps_riemann(z_trunc)
f5, ax5 = plt.subplots()
ax5.plot(simps, 'r-', label='simpsion method')
ax5.plot(riemann, 'b-', label='riemann sum')
ax5.grid(True)
ax5.set_title('Integrated Intensity as a Function of Profile Number')
ax5.set_ylabel('Integrated Intensity')
ax5.set_xlabel('Profile Number')
plt.show()


# full resolution for perpendicular transects
f6, ax6 = plt.subplots(1, 2)
ax6[0].imshow(im, cmap='gray')
ax6[0].plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax6[0].plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax6[0].plot(cols_DF, mt, 'rX', ms=0.4)
ax6[1].set_title('Profiles')
ax6[1].set_xlabel('Y Coordinate')
ax6[1].set_ylabel('Intensity')
ax6[1].grid(True)
for counter, elements in enumerate(x_pts):
    if counter < numcols:
        ax6[0].plot(x_pts[counter], y_pts[counter], color='yellow', marker='.', ms=0.2)
for element in z:
    ax6[1].plot(element)
plt.show()

# full resolution phase function for perpendicular transects
simps2, riemann2 = loop_simps_riemann(z)
f7, ax7 = plt.subplots()
ax7.plot(simps2, 'r-', label='simpsion method')
ax7.plot(riemann2, 'b-', label='riemann sum')
ax7.grid(True)
ax7.set_title('Integrated Intensity as a Function of Profile Number')
ax7.set_ylabel('Integrated Intensity')
ax7.set_xlabel('Profile Number')
plt.show()


'''
#Save Phase Function
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Profile Number'] = cols_DF
PhaseFunctionDF['Intensity'] = aug_array
PhaseFunctionDF['Normalized Intensity'] = aug_array_norm
PhaseFunctionDF.to_csv(Path_PF)
#'''


