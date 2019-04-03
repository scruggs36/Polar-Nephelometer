import PIL
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from Neph_Functions import Loop_Image_Sum, Profiles, Curve_Fit_Profiles, gaussian, linear, quadratic

'''
This code is used to evaluate a single pass of a summed image!
'''
# network file path
Path_ParticleDir = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/04-16-2018/PSL_701nm'
Path_MediumImage = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/04-16-2018/Analysis_N2/Average/im_avg_N2_60s.png'
Path_HeDir = ''
Path_SaveDir = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/04-16-2018/Analysis_PSL_701nm/Sum'
FN_1 = '/im_sum_PSL701nm_60s_coord.png'
FN_2 = '/im_sum_PSL701nm_60s.png'



# creating different types of the summed array
im_summed = Loop_Image_Sum(Path_ParticleDir, Path_MediumImage)
#im_summed_8bit = im_summed.astype('uint8')
im_summed_int = im_summed.astype('int')

# plot summed image
plt.imshow(im_summed_int, cmap='inferno')
plt.colorbar()
plt.show()
#plt.savefig(Path_Save + FN_1)

# plot histogram of summed image
histogram, bin_edges = np.histogram(im_summed_int, bins=np.arange(0, im_summed_int.flatten().max(), 10))
plt.hist(histogram, bin_edges)
plt.title('Image Histogram')
plt.xlabel('Summed Digital Number Bins')
plt.ylabel('$Log_{10}(Counts)$')
plt.yscale('log')
plt.xscale('linear')
plt.grid(True)
plt.show()

# writes image to new file
#cv2.imwrite(Path_Save + FN_2, im_summed_int)

# Initial boundaries on the image
col_boundary_1 = 100
col_boundary_2 = 700
row_boundary_1 = 300
row_boundary_2 = 600
numrows = row_boundary_2 - row_boundary_1
numcols = col_boundary_2 - col_boundary_1


rows_DF, cols_DF, int_DF = Profiles(im_summed_int, col_boundary_1, col_boundary_2, numrows, numcols, row_boundary_1, row_boundary_2)
fitxdata1, fitydata1, fyd1, ft, bt, mt, st, bkgt, aug_array, aug_array_norm, tt_smooth, tb_smooth = Curve_Fit_Profiles(gaussian, linear, rows_DF, int_DF, cols_DF)
#print(mt)

# here we are plotting the fit data
f1, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im_summed_int, cmap='gray')
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
ax1a.imshow(im_summed_int, cmap='gray')
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

