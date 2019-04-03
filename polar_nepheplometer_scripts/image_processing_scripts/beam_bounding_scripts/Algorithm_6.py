from Neph_Functions import Loop_Image_Average, Image_Subtract, Image_Heatmaps, Profiles, Curve_Fit_Profiles, gaussian, quadratic, linear
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from scipy.optimize import curve_fit
#from skimage.exposure import histogram
#import math


'''
This code evaluates the average of all images in a file directory between the rows given in the initial boundaries section
'''
# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_ParticleDir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/05-11-2018/PSL_701nm'
Path_MediumImage = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/05-11-2018/N2_Analysis/im_avg_N2_10s.png'
Path_HeDir = ''
Path_SaveDir = '/home/austen/Documents'
FN_1 = '/im_avg_PSL701nm_10s_coord.png'
FN_2 = '/im_avg_PSL701nm_10s.png'

# averaging images
im_Avg = Loop_Image_Average(Path_ParticleDir, Path_MediumImage)




# Initial boundaries on the image
col_boundary_1 = 200
col_boundary_2 = 1000
row_boundary_1 = 200
row_boundary_2 = 600
numrows = row_boundary_2 - row_boundary_1
numcols = col_boundary_2 - col_boundary_1


# calling function to draw vertical profiles
#Image_Heatmaps(imBKG, im, 255, Path_HM)


rows_DF, cols_DF, int_DF = Profiles(im_Avg, col_boundary_1, col_boundary_2, numrows, numcols, row_boundary_1, row_boundary_2)
fitxdata1, fitydata1, fyd1, ft, bt, mt, st, bkgt, aug_array, aug_array_norm, tt_smooth, tb_smooth, col_array = Curve_Fit_Profiles(gaussian, quadratic, rows_DF, int_DF, cols_DF)
#print(mt)


# here we are plotting the fit data
f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im_Avg, cmap='gray')
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
ax1a.imshow(im_Avg, cmap='gray')
ax1a.plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
ax1a.plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
ax1a.plot(cols_DF, mt, 'rX', ms=0.4)
plt.show()


f2, ax2 = plt.subplots()
ax2.plot(cols_DF, aug_array_norm, 'r-', label='Phase Function')
ax2.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax2.set_xlabel('Column Profile Number')
ax2.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.grid(True)
plt.show()


f3, ax3 = plt.subplots()
ax3.plot(cols_DF, aug_array_norm, 'r-', label='Phase Function')
ax3.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax3.set_xlabel('Column Profile Number')
ax3.set_ylabel('Integrated Intensity')
plt.yscale('log')
plt.legend(loc=1)
plt.grid(True)
plt.show()


f4, ax4 = plt.subplots()
ax4.plot(cols_DF, aug_array, 'b-', label='Phase Function')
ax4.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax4.set_xlabel('Column Profile Number')
ax4.set_ylabel('Integrated Intensity')
plt.yscale('log')
plt.legend(loc=1)
plt.grid(True)
plt.show()


f5, ax5 = plt.subplots()
ax5.plot(cols_DF, aug_array, 'b-', label='Phase Function')
ax5.set_title('Integrated Intensity as a Function of Vertical Profile Number')
ax5.set_xlabel('Column Profile Number')
ax5.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.grid(True)
plt.show()


#'''
#Save Phase Function
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Profile Number'] = cols_DF
PhaseFunctionDF['Intensity'] = aug_array
PhaseFunctionDF['Normalized Intensity'] = aug_array_norm
PhaseFunctionDF.to_csv(Path_SaveDir + '/PF.txt')
#'''

