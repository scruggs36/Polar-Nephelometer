'''
Austen K. Scruggs
10-10-2018
Desctription: This code averages background and sample images, then subtracts the image to get a resultant background
subtracted sample image. The image then gets evaluated in the same way that the labview code evaluates the images!
'''

from Neph_Functions import Loop_Image_Average,  Image_Subtract
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, pchip_interpolate
from math import pi, cos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_PSL_Dir = '/home/austen/Documents/03-11-2019_Analysis/SD_3s_Offline.txt'
Path_CO2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/03-11-2019/CO2/3s'
Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/03-11-2019/N2/3s'
Path_He_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/03-11-2019/He/3s'
Path_BKG_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/03-11-2019/BKG/3s'
#Path_Dark_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/01-23-2019/900/BKG/T1'
Path_Save = '/home/austen/Documents/'

# import PSL Data
PSL_Data = pd.read_csv(Path_PSL_Dir, sep=',', header=0)
PSL_C = np.array(PSL_Data['Sample Columns'])
PSL_I = np.array(PSL_Data['Sample Intensity'])
N_C = np.array(PSL_Data['Nitrogen Columns'])
N_I = np.array(PSL_Data['Nitrogen Intensity'])
# averaging background images
im_CO2Avg = Loop_Image_Average(Path_CO2_Dir)
im_N2Avg = Loop_Image_Average(Path_N2_Dir)
im_HeAvg = Loop_Image_Average(Path_He_Dir)
im_BKGAvg = Loop_Image_Average(Path_BKG_Dir)
#im_DarkAvg = Loop_Image_Average(Path_Dark_Dir)

# writes image to new file
cv2.imwrite(Path_Save + '/CO2_Avg.BMP', im_CO2Avg)
cv2.imwrite(Path_Save + '/N2_Avg.BMP', im_N2Avg)
cv2.imwrite(Path_Save + '/He_Avg.BMP', im_HeAvg)
cv2.imwrite(Path_Save + '/BKG_Avg.BMP', im_BKGAvg)

# start subtractions for N2 Rayleigh scattering and sample scattering, this is predominantly for pretty pictures only!
im_Helium_Corrected = Image_Subtract(im_HeAvg, im_BKGAvg) # pure helium scattering and light hitting walls; in theory the thermal background is subtracted
im_N2Rayleigh = Image_Subtract(im_N2Avg, im_HeAvg) # this subtracts the helium scattering, the light hitting walls, and thermal background; in theory leaving pure nitrogen scattering
im_CO2Rayleigh = Image_Subtract(im_CO2Avg, im_HeAvg)

# writes pretty images to new file
cv2.imwrite(Path_Save + '/He_Avg_Corrected.BMP', im_Helium_Corrected)
cv2.imwrite(Path_Save + '/N2_Avg_Corrected.BMP', im_N2Rayleigh)
cv2.imwrite(Path_Save + '/CO2_Avg_Corrected.BMP', im_CO2Rayleigh)

# Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
rows = [200, 600]
cols = [250, 1050]
cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
#ROI = im[rows[0]:rows[1], cols[0]:cols[1]]

# find coordinates based on sample - N2 scattering averaged image (without corrections)
row_max_index_array = []
for element in cols_array:
    arr = np.arange(rows[0], rows[1], 1).astype(int)
    im_transect = im_CO2Rayleigh[arr, element]
    index_nosub = np.argmax(im_transect)
    row_max_index_array.append(index_nosub + rows[0])

# polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
polynomial_fit = np.poly1d(np.polyfit(cols_array, row_max_index_array, deg=2))
sigma_pixels = 30
mid = polynomial_fit(cols_array)
top = polynomial_fit(cols_array) - sigma_pixels
bot = polynomial_fit(cols_array) + sigma_pixels

# pretty picture plots for background signal corrections
# plots of all averaged images and profile coordinates
# plot of averaged background images
f0, ax0 = plt.subplots(1, 2, figsize=(12, 7))
im_f0a = ax0[0].imshow(im_BKGAvg, cmap='gray')
im_f0b = ax0[1].imshow(im_BKGAvg, cmap='gray')
ax0[1].plot(cols_array, top, ls='-', color='lawngreen')
ax0[1].plot(cols_array, mid, ls='-', color='red')
ax0[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
ax0[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(ax0[0])
divider_b = make_axes_locatable(ax0[1])
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
f0.colorbar(im_f0a, cax=cax_a)
f0.colorbar(im_f0b, cax=cax_b)
ax0[0].set_title('Background Image')
ax0[1].set_title('Background Image & \n SD Coordinates')
plt.savefig(Path_Save + '/BKG.pdf', format='pdf')
plt.show()

# pretty picture plots for background signal corrections
# plot of averaged Helium images
f1, ax1 = plt.subplots(1, 2, figsize=(12, 7))
im_f1a = ax1[0].imshow(im_HeAvg, cmap='gray')
im_f1b = ax1[1].imshow(im_Helium_Corrected, cmap='gray')
ax1[1].plot(cols_array, top, ls='-', color='lawngreen')
ax1[1].plot(cols_array, mid, ls='-', color='red')
ax1[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
ax1[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(ax1[0])
divider_b = make_axes_locatable(ax1[1])
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
f1.colorbar(im_f1a, cax=cax_a)
f1.colorbar(im_f1b, cax=cax_b)
ax1[0].set_title('Helium Image')
ax1[1].set_title('Helium Image Background Corrected')
plt.savefig(Path_Save + '/He.pdf', format='pdf')
plt.show()

# pretty picture plots for background signal corrections
# plot of the N2 corrected (helium and bkg) subtracted data
f2, ax2 = plt.subplots(1, 2, figsize=(12, 7))
im_f2a = ax2[0].imshow(im_N2Avg, cmap='gray')
im_f2b = ax2[1].imshow(im_N2Rayleigh, cmap='gray')
ax2[1].plot(cols_array, top, ls='-', color='lawngreen')
ax2[1].plot(cols_array, mid, ls='-', color='red')
ax2[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
ax2[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(ax2[0])
divider_b = make_axes_locatable(ax2[1])
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
f2.colorbar(im_f2a, cax=cax_a)
f2.colorbar(im_f2a, cax=cax_b)
ax2[0].set_title('Nitrogen Image')
ax2[1].set_title('Helium Corrected Nitrogen Image')
plt.savefig(Path_Save + '/N2.pdf', format='pdf')
plt.show()

# pretty picture plots for background signal corrections
# plot of averaged sample images
f3, ax3 = plt.subplots(1, 2, figsize=(12, 7))
im_f3a = ax3[0].imshow(im_CO2Avg, cmap='gray')
im_f3b = ax3[1].imshow(im_CO2Rayleigh, cmap='gray')
ax3[1].plot(cols_array, top, ls='-', color='lawngreen')
ax3[1].plot(cols_array, mid, ls='-', color='red')
ax3[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
ax3[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(ax3[0])
divider_b = make_axes_locatable(ax3[1])
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
f3.colorbar(im_f3a, cax=cax_a)
f3.colorbar(im_f3b, cax=cax_b)
ax3[0].set_title('$CO_2$ Image')
ax3[1].set_title('Helium Corrected $CO_2$ Image')
plt.savefig(Path_Save + '/Sample.pdf', format='pdf')
plt.show()


# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_N2 = []
SD_N2_imsub = []
arr_ndarray_N2 = []
bound_transect_ndarray_N2 = []
bound_transect_ndarray_N2_imsub = []
background_N2 = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_N2.append(arr)
    bound_transect = im_N2Avg[arr, element]
    transect_summed = np.sum(bound_transect)
    SD_N2.append(transect_summed)
    bound_transect_imsub = im_N2Rayleigh[arr, element]
    bound_transect_ndarray_N2.append(bound_transect)
    bound_transect_imsub_corr = np.array(bound_transect_imsub).astype('int') - int(round(np.mean(bound_transect_imsub[0:10])))
    for counter, val in enumerate(bound_transect_imsub_corr):
        if val < 0:
            bound_transect_imsub_corr[counter] = 0
    bound_transect_ndarray_N2_imsub.append(bound_transect_imsub_corr)
    transect_summed_imsub = np.sum(bound_transect_imsub_corr)
    SD_N2_imsub.append(transect_summed_imsub)
SD_N2_imsub = np.array(SD_N2_imsub).astype('int')

# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_CO2 = []
SD_CO2_imsub = []
arr_ndarray_CO2 = []
bound_transect_ndarray_CO2 = []
bound_transect_ndarray_CO2_imsub = []
background_CO2 = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_CO2.append(arr)
    bound_transect = im_CO2Avg[arr, element]
    bound_transect_ndarray_CO2.append(bound_transect)
    transect_summed = np.sum(bound_transect)
    SD_CO2.append(transect_summed)
    bound_transect_imsub = im_CO2Rayleigh[arr, element]
    bound_transect_imsub_corr = np.array(bound_transect_imsub).astype('int') - int(round(np.mean(bound_transect_imsub[0:10])))
    for counter, val in enumerate(bound_transect_imsub_corr):
        if val < 0:
            bound_transect_imsub_corr[counter] = 0
    bound_transect_ndarray_CO2_imsub.append(bound_transect_imsub_corr)
    transect_summed_imsub = np.sum(bound_transect_imsub_corr)
    SD_CO2_imsub.append(transect_summed_imsub)
SD_CO2_imsub = np.array(SD_CO2_imsub).astype('int')

# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_He = []
SD_He_imsub = []
arr_ndarray_He = []
bound_transect_ndarray_He = []
bound_transect_ndarray_He_imsub = []
background_He = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_He.append(arr)
    bound_transect = im_HeAvg[arr, element]
    transect_summed = np.sum(bound_transect)
    SD_He.append(transect_summed)
    bound_transect_imsub = im_Helium_Corrected[arr, element]
    bound_transect_imsub_corr = np.array(bound_transect_imsub).astype('int') - int(round(np.mean(bound_transect_imsub[0:10])))
    for counter, val in enumerate(bound_transect_imsub_corr):
        if val < 0:
            bound_transect_imsub_corr[counter] = 0
    transect_summed_imsub = np.sum(bound_transect_imsub_corr)
    SD_He_imsub.append(transect_summed_imsub)
SD_He_imsub = np.array(SD_He_imsub).astype('int')

# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_BKG = []
SD_BKG_Corr = []
arr_ndarray_BKG = []
bound_transect_ndarray_BKG = []
bound_transect_ndarray_BKG_Corr = []
background_BKG = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_BKG.append(arr)
    bound_transect = im_BKGAvg[arr, element]
    transect_summed = np.sum(bound_transect)
    SD_BKG.append(transect_summed)
    bound_transect_Corr = np.array(bound_transect).astype('int') - int(round(np.mean(bound_transect[0:len(bound_transect)])))
    for counter, val in enumerate(bound_transect_Corr):
        if val < 0:
            bound_transect_Corr[counter] = 0
    transect_summed_Corr = np.sum(bound_transect_Corr)
    SD_BKG_Corr.append(transect_summed_Corr)
SD_BKG_Corr = np.array(SD_BKG_Corr)

'''
# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_Dark = []
arr_ndarray_Dark = []
bound_transect_ndarray_Dark = []
background_Dark = []
SD_Dark_ccd_corrected = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_Dark.append(arr)
    bound_transect = im_DarkAvg[arr, element]
    bound_transect_ndarray_Dark.append(bound_transect)
    transect_summed = np.sum(bound_transect)
    SD_Dark.append(transect_summed)
'''

# plot of the Sample nitrogen subtracted data with bounds
f4, ax4 = plt.subplots(2, 2, figsize=(12, 6))
im_f4 = ax4[0, 0].imshow(im_CO2Rayleigh, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f4.colorbar(im_f4, cax=cax)
ax4[0, 0].set_title('Averaged $CO_2$ Image')
ax4[0, 1].imshow(im_CO2Rayleigh, cmap='gray')
ax4[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax4[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].set_xlabel('Columns')
ax4[0, 1].set_ylabel('Rows')
ax4[0, 1].set_title('Averaged $CO_2$ Image \n He Subtracted')
for counter, element in enumerate(arr_ndarray_CO2):
    #ax4[1, 0].plot(element, bound_transect_ndarray_CO2[counter], linestyle='-')
    ax4[1, 0].plot(element, bound_transect_ndarray_CO2_imsub[counter], linestyle='-')
ax4[1, 0].set_xlabel('Rows')
ax4[1, 0].set_ylabel('Intensity (DN)')
ax4[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax4[1, 0].grid(True)
ax4[1, 1].plot(cols_array, SD_CO2, linestyle='-', color='red', label='SD: $CO_2$')
ax4[1, 1].plot(cols_array, SD_CO2_imsub, linestyle='-', color='blue', label='SD: $CO_2$ - He')
#ax4[1, 1].plot(cols_array, SD_CO2_imsub_corrected, linestyle='-', color='green', label='SD: $CO_2$ - He - Edge Corrected')
ax4[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax4[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax4[1, 1].set_title('Scattering Diagram')
ax4[1, 1].grid(True)
ax4[1, 1].set_yscale('log')
ax4[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F4_Sample.pdf', format='pdf')
plt.show()


# plot of the backgound subtracted data with bounds
f5, ax5 = plt.subplots(2, 2, figsize=(12, 6))
im_f5 = ax5[0, 0].imshow(im_N2Avg, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax5[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f5.colorbar(im_f5, cax=cax)
ax5[0, 0].set_title('Averaged Nitrogen Image')
ax5[0, 1].imshow(im_N2Rayleigh, cmap='gray')
ax5[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax5[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax5[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax5[0, 1].set_xlabel('Columns')
ax5[0, 1].set_ylabel('Rows')
ax5[0, 1].set_title('Averaged $N_2$ Image\n Helium Subtracted')
for counter, element in enumerate(arr_ndarray_N2):
    #ax5[1, 0].plot(element, bound_transect_ndarray_N2[counter], linestyle='-')
    ax5[1, 0].plot(element, bound_transect_ndarray_N2_imsub[counter], linestyle='-')
ax5[1, 0].set_xlabel('Rows')
ax5[1, 0].set_ylabel('Intensity (DN)')
ax5[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax5[1, 0].grid(True)
ax5[1, 1].plot(cols_array, SD_N2, linestyle='-', color='red', label='SD: $N_2$')
ax5[1, 1].plot(cols_array, SD_N2_imsub, linestyle='-', color='blue', label='SD: $N_2$ - He')
#ax5[1, 1].plot(cols_array, SD_N2_imsub_corrected, linestyle='-', color='green', label='SD: N2 - He - Edge Correction')
ax5[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax5[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax5[1, 1].set_title('Scattering Diagram')
ax5[1, 1].grid(True)
ax5[1, 1].set_yscale('log')
ax5[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F5_N2.pdf', format='pdf')
plt.show()

# Save Phase Function, the data saved here has no subtractions/corrections applied to them, each is raw signal
# note the CCD Noise cannot be backed out, as we would have to cover the lens to do it, if at some point we take
# covered images we could do it...
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Columns'] = cols_array
PhaseFunctionDF['Background Intensity'] = SD_BKG
PhaseFunctionDF['Helium Intensity'] = SD_He
PhaseFunctionDF['Nitrogen Intensity'] = SD_N2
PhaseFunctionDF['CO2 Intensity'] = SD_CO2
PhaseFunctionDF['Helium Intensity Corr.'] = SD_He_imsub
PhaseFunctionDF['Nitrogen Intensity Corr.'] = SD_N2_imsub
PhaseFunctionDF['CO2 Intensity Corr.'] = SD_CO2_imsub
PhaseFunctionDF.to_csv(Path_Save + '/SD_Offline.txt')



f6, ax6 = plt.subplots(2, 2, figsize=(12, 7))
ax6[0, 0].plot(arr_ndarray_CO2[50], bound_transect_ndarray_CO2[50], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[50][:10])*len(bound_transect_ndarray_CO2[50])))
ax6[0, 0].plot(arr_ndarray_CO2[50], bound_transect_ndarray_CO2_imsub[50], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2_imsub[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2_imsub[50][:10])*len(bound_transect_ndarray_CO2_imsub[50])))
ax6[0, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 0].set_title('Profiles Compared')
ax6[0, 0].grid(True)
ax6[0, 0].legend(loc=1)
ax6[0, 1].plot(arr_ndarray_CO2[200], bound_transect_ndarray_CO2[200], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[200][:10])*len(bound_transect_ndarray_CO2[200])))
ax6[0, 1].plot(arr_ndarray_CO2[200], bound_transect_ndarray_CO2_imsub[200], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2_imsub[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2_imsub[200][:10])*len(bound_transect_ndarray_CO2_imsub[200])))
ax6[0, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 1].set_title('Profiles Compared')
ax6[0, 1].grid(True)
ax6[0, 1].legend(loc=1)
ax6[1, 0].plot(arr_ndarray_CO2[300], bound_transect_ndarray_CO2[300], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[300][:10])*len(bound_transect_ndarray_CO2[300])))
ax6[1, 0].plot(arr_ndarray_CO2[300], bound_transect_ndarray_CO2_imsub[300], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2_imsub[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2_imsub[300][:10])*len(bound_transect_ndarray_CO2_imsub[300])))
ax6[1, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 0].set_title('Profiles Compared')
ax6[1, 0].grid(True)
ax6[1, 0].legend(loc=1)
ax6[1, 1].plot(arr_ndarray_CO2[350], bound_transect_ndarray_CO2[350], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[350][:10])*len(bound_transect_ndarray_CO2[350])))
ax6[1, 1].plot(arr_ndarray_CO2[350], bound_transect_ndarray_CO2_imsub[350], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2_imsub[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2_imsub[350][:10])*len(bound_transect_ndarray_CO2_imsub[350])))
ax6[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 1].set_title('Profiles Compared')
ax6[1, 1].grid(True)
ax6[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F6_Profiles.pdf', format='pdf')
plt.show()

# columns to theta
slope = 0.2112
intercept = -47.972
theta = (np.array(cols_array) * slope) + intercept
rads = theta * pi/180.0

# fit to 1 + cos^2(theta)
def cos_fn(x, a, b):
    return b + (a * np.square(np.cos(x)))

popt_co2, pcov_co2 = curve_fit(cos_fn, rads, SD_CO2)
popt_N2, pcov_N2 = curve_fit(cos_fn, rads, SD_N2)
popt_co2_imsub, pcov_co2_imsub = curve_fit(cos_fn, rads, SD_CO2_imsub)
popt_N2_imsub, pcov_N2_imsub = curve_fit(cos_fn, rads, SD_N2_imsub)

f7, ax7 = plt.subplots(1, 3, figsize=(20, 7))
ax7[0].plot(theta, SD_CO2, linestyle='-', color='black', label='$CO_2$ Scattering')
ax7[0].plot(theta, cos_fn(rads, *popt_co2), linestyle='--', color='purple', label='$CO_2$ Scattering fit')
ax7[0].plot(theta, SD_N2, linestyle='-', color='green', label='$N_2$ Scattering')
ax7[0].plot(theta, cos_fn(rads, *popt_N2), linestyle='--', color='orange', label='$N_2$ Scattering fit')
ax7[0].plot(theta, SD_He, linestyle='-', color='cyan', label='He Scattering')
# something is wrong with the background when saving a PNG, I plan on comparing the 2darray to the png image directly
#ax7[0].plot(theta, SD_BKG, linestyle='-', color='red', label='Background')
ax7[1].plot(theta, SD_CO2_imsub, linestyle='-', color='black', label='$CO_2$ Scattering')
ax7[1].plot(theta, cos_fn(rads, *popt_co2_imsub), linestyle='--', color='purple', label='$CO_2$ Scattering fit')
ax7[1].plot(theta, SD_N2_imsub, linestyle='-', color='green', label='$N_2$ Scattering')
ax7[1].plot(theta, cos_fn(rads, *popt_N2_imsub), linestyle='--', color='orange', label='$N_2$ Scattering fit')
ax7[1].plot(theta, SD_He_imsub, linestyle='-', color='cyan', label='He Scattering')
# something is wrong with the background when saving a PNG, I plan on comparing the 2darray to the png image directly
#ax7[1].plot(theta, SD_BKG_Corr, linestyle='-', color='red', label='Background')
ax7[2].plot(theta, PSL_I/N_I, linestyle='-', color='red', label='PSL/Nitrogen Scattering')
ax7[0].set_title('Scattering Contributions to Raw Sample Scattering Diagram')
ax7[0].set_ylabel('Intensity (DN)')
ax7[0].set_xlabel('\u0398')
ax7[0].grid(True)
ax7[0].legend(loc=1)
ax7[1].set_title('Scattering Contributions to Corrected Sample Scattering Diagram')
ax7[1].set_ylabel('Intensity (DN)')
ax7[1].set_xlabel('\u0398')
ax7[1].grid(True)
ax7[1].legend(loc=1)
ax7[2].set_title('PSL/N2 Ratio Scattering Diagram')
ax7[2].set_ylabel('Ratio')
ax7[2].set_xlabel('\u0398')
ax7[2].grid(True)
ax7[2].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F7_Contributions.pdf', format='pdf')
plt.show()


rayleigh_cos = np.array([(1 + (cos(rad))**2) for rad in rads])
ratio_CO2 = np.array(SD_CO2_imsub) / rayleigh_cos
ratio_N2 = np.array(SD_N2_imsub) / rayleigh_cos
ratio_CO2_min = ratio_CO2 / np.amin(ratio_CO2)
ratio_N2_min = ratio_N2 / np.amin(ratio_N2)

print(ratio_N2_min)
ratio_CO2_min_savgol = savgol_filter(ratio_CO2_min, window_length=151, polyorder=2, deriv=0)
ratio_N2_min_savgol = savgol_filter(ratio_N2_min, window_length=151, polyorder=2, deriv=0)
ratio_CO2_min_pchip = pchip_interpolate(theta, ratio_CO2_min_savgol, theta, der=0, axis=0)
ratio_N2_min_pchip = pchip_interpolate(theta, ratio_N2_min_savgol, theta, der=0, axis=0)

# we did not do a semilogy plot here, y data cannot have zeros in it! log of zero is not defined!!!
f8, ax8 = plt.subplots(1, 3, figsize=(20, 7))
ax8[0].plot(theta, SD_CO2_imsub, linestyle='-', color='red', label='$CO_2$ Scattering - He Scattering')
ax8[0].plot(theta, SD_N2_imsub, linestyle='-', color='blue', label='$N_2$ Scattering - He Scattering')
ax8[0].plot(theta, SD_CO2_imsub/SD_N2_imsub, linestyle='-', color='green', label='$CO_2$/$N_2$ Ratio')
ax8[0].plot(theta, SD_He_imsub, linestyle='-', color='purple', label='He Scattering - Background')
# something is wrong with the background when saving a PNG, I plan on comparing the 2darray to the png image directly
#ax8[0].plot(theta, SD_BKG_Corr, linestyle='-', color='black', label='Background')
ax8[0].set_title('Rayleigh Scattering Diagrams')
ax8[0].set_ylabel('Intensity (DN)')
ax8[0].set_xlabel('\u0398')
ax8[0].grid(True)
ax8[0].legend(loc=1)
ax8[1].plot(theta, ratio_CO2_min, linestyle='-', color='red', label='$CO_2$ Scattering/(1 + $cos^2(\u0398)$')
ax8[1].plot(theta, ratio_CO2_min_pchip, linestyle='-', color='orange', label='$CO_2$ Scattering/(1 + $cos^2(\u0398)$ Pchip')
ax8[1].plot(theta, ratio_N2_min, linestyle='-', color='blue', label='$N_2$ Scattering/(1 + $cos^2(\u0398)$')
ax8[1].plot(theta, ratio_N2_min_pchip, linestyle='-', color='aqua', label='$N_2$ Scattering /(1 + $cos^2(\u0398)$ Pchip')
ax8[1].plot(theta, SD_CO2_imsub/SD_N2_imsub, linestyle='-', color='green', label='$CO_2$/$N_2$ Ratio')
#ax8[1].plot(theta, np.array(SD_He_imsub) / np.array([(1 + (cos(rad))**2) for rad in rads]), linestyle='-', color='cyan', label='He Scattering - Background')
#ax8[1].plot(theta, SD_BKG_Corr, linestyle='-', color='red', label='Background')
ax8[1].set_title('Rayleigh Ratio')
ax8[1].set_ylabel('Ratio')
ax8[1].set_xlabel('\u0398')
ax8[1].grid(True)
ax8[1].legend(loc=1)
ax8[2].plot(theta, PSL_I/ratio_N2_min, linestyle='-', color='lawngreen', label='PSL Intensity/($N_2$ Scattering/(1 + $cos^2(\u0398)$)')
ax8[2].set_title('PSL to Rayleigh Ratio')
ax8[2].set_ylabel('Ratio')
ax8[2].set_xlabel('\u0398')
ax8[2].grid(True)
ax8[2].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F8_Contributions_Corr.pdf', format='pdf')
plt.show()

