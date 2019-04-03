from Neph_Functions import Loop_Image_Average, Image_Heatmaps, Profiles2, Curve_Fit_Profiles, Curve_Fit_Profiles2, gaussian, quartic, quadratic, linear
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
Path_ParticleDir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/07-18-2018/Polarizer After Half Waveplate/100/Images'
Path_BKGDir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/07-26-2018/He_Pol'
Path_Save = '/home/austen/Documents'

# file names
P_FN_1 = '/im_avg_N2_HeSub_30s_HW_100Degrees_coord.png'
P_FN_2 = '/im_avg_N2_HeSub_30s_HW_100Degrees.png'
BKG_FN_1 = '/im_avg_He_30s_HW_34Degrees_coord.png'
BKG_FN_2 = '/im_avg_He_30s_HW_34Degrees.png'
Path_PF = '/SD_N2_HeSub_HW_100Degrees.txt'



# averaging background images
im_BKGAvg = Loop_Image_Average(Path_BKGDir)
plt.imshow(im_BKGAvg, cmap='gray')
# saves image plot
plt.colorbar()
#plt.savefig(Path_Save + BKG_FN_1)
#plt.show()
plt.clf()

# writes image to new file
cv2.imwrite(Path_Save + BKG_FN_2, im_BKGAvg)
imBKG = cv2.imread(Path_Save + BKG_FN_2, 0)
im = cv2.imread(Path_Save + BKG_FN_2, 0)

# subtracts images from background and averages them
im_Avg = Loop_Image_Average(Path_ParticleDir, Path_Save + BKG_FN_2)
plt.imshow(im_Avg, cmap='gray')
# saves image plot
plt.colorbar()
plt.savefig(Path_Save + P_FN_1)
#plt.show()
plt.clf()
# writes image to new file
cv2.imwrite(Path_Save + P_FN_2, im_Avg)
im = cv2.imread(Path_Save + P_FN_2, 0)


# Initial boundaries on the image
col_boundary_1 = 252
col_boundary_2 = 1059
row_boundary_1 = 500
row_boundary_2 = 600
numrows = row_boundary_2 - row_boundary_1
numcols = col_boundary_2 - col_boundary_1



rows_DF, cols_DF, tt_smooth, tb_smooth, mid_smooth, int_DF = Profiles2(im, col_boundary_1, col_boundary_2, numrows, numcols, row_boundary_1, row_boundary_2, quartic, 20)
fitxdata1, fitydata1, fyd1, ft, bt, mt, col_array, st, bkgt, aug_array, aug_array_norm = Curve_Fit_Profiles2(gaussian, rows_DF, cols_DF, tt_smooth, tb_smooth, mid_smooth, int_DF)
#print(mid_smooth)


# here we are plotting the fit data
f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im, cmap='gray')
ax1[0].plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1[0].plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1[0].plot(cols_DF, mid_smooth, color='red', marker='.', ms=0.2)
ax1[1].set_title('Fits to Top Profiles')
ax1[1].set_xlabel('Y Coordinate')
ax1[1].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1):
    ax1[1].plot(fitxdata1[counter], fyd1[counter])
#plt.show()
plt.savefig(Path_Save + '/Bounds_&_Fits.pdf', format='pdf')
plt.clf()

f1a, ax1a = plt.subplots()
ax1a.imshow(im, cmap='gray')
ax1a.plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1a.plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1a.plot(cols_DF, mt, 'rX', ms=0.4)
#plt.show()
plt.savefig(Path_Save + '/Bounds.pdf', format='pdf')
plt.clf()

f2, ax2 = plt.subplots()
ax2.plot(cols_DF, aug_array_norm, 'r-', label='Phase Function')
ax2.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax2.set_xlabel('Column Profile Number')
ax2.set_ylabel('Integrated Intensity')
ax2.legend(loc=1)
#plt.show()
plt.savefig(Path_Save + '/SD_Norm.pdf', format='pdf')
plt.clf()

f3, ax3 = plt.subplots()
ax3.plot(cols_DF, aug_array, 'b-', label='Phase Function')
ax3.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax3.set_xlabel('Column Profile Number')
ax3.set_ylabel('Integrated Intensity')
ax3.legend(loc=1)
plt.yscale('log')
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig(Path_Save + '/SD_Log.pdf', format='pdf')
plt.clf()


#'''
#Save Phase Function
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Columns'] = cols_DF
PhaseFunctionDF['Intensity'] = aug_array
PhaseFunctionDF.to_csv(Path_Save + Path_PF)
#'''