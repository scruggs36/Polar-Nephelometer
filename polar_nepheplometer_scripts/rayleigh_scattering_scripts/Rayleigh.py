'''
Austen K. Scruggs
10-10-2018
Desctription: This code averages background and sample images, then subtracts the image to get a resultant background
subtracted sample image. The image then gets evaluated in the same way that the labview code evaluates the images!
'''

from Neph_Functions import Loop_Image_Average,  Image_Subtract
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# Path
Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/11-26-2018/N2/15s'
Path_He_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/11-19-2018/He/1s'
Path_BKG_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/11-26-2018/BKG'
Path_Save = '/home/austen/Documents'

# averaging background images
im_N2Avg = Loop_Image_Average(Path_N2_Dir)
im_HeAvg = Loop_Image_Average(Path_He_Dir)
im_BKGAvg = Loop_Image_Average(Path_BKG_Dir)

# writes image to new file
cv2.imwrite(Path_Save + '/BKG_Avg.BMP', im_BKGAvg)
cv2.imwrite(Path_Save + '/He_Avg.BMP', im_HeAvg)
cv2.imwrite(Path_Save + '/N2_Avg.BMP', im_N2Avg)

# start subtractions for N2 Rayleigh scattering and sample scattering
im_Helium_Corrected = Image_Subtract(im_HeAvg, im_BKGAvg) # pure helium scattering and light hitting walls; in theory the thermal background is subtracted
im_N2Rayleigh = Image_Subtract(im_N2Avg, im_HeAvg) # this subtracts the helium scattering, the light hitting walls, and thermal background; in theory leaving pure nitrogen scattering

# writes image to new file
cv2.imwrite(Path_Save + '/He_Avg_Corrected.BMP', im_Helium_Corrected)
cv2.imwrite(Path_Save + '/N2_Avg_Corrected.BMP', im_N2Rayleigh)

# plot of averaged background images
f0, ax0 = plt.subplots()
im_f0 = ax0.imshow(im_BKGAvg, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
f0.colorbar(im_f0, cax=cax)
ax0.set_title('Background Image')
plt.savefig(Path_Save + '/BKG.pdf', format='pdf')
plt.show()

# plot of averaged bkg images
f1, ax1 = plt.subplots(1, 2, figsize=(12, 7))
im_f1a = ax1[0].imshow(im_HeAvg, cmap='gray')
im_f1b = ax1[1].imshow(im_Helium_Corrected, cmap='gray')
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

# plot of the N2 corrected (helium and bkg) subtracted data
f2, ax2 = plt.subplots(1, 2, figsize=(12, 7))
im_f2a = ax2[0].imshow(im_N2Avg, cmap='gray')
im_f2b = ax2[1].imshow(im_N2Rayleigh, cmap='gray')
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

# Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
rows = [400, 600]
cols = [260, 1050]
cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
#ROI = im[rows[0]:rows[1], cols[0]:cols[1]]

# find coordinates based on N2 rayleigh scattering averaged image (without corrections)
row_max_index_array_N2 = []
for element in cols_array:
    arr = np.arange(rows[0], rows[1], 1).astype(int)
    im_transect = im_N2Avg[arr, element]
    index_nosub = np.argmax(im_transect)
    row_max_index_array_N2.append(index_nosub + rows[0])

# polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
polynomial_fit = np.poly1d(np.polyfit(cols_array, row_max_index_array_N2, deg=2))
sigma_pixels = 20
mid = polynomial_fit(cols_array)
top = polynomial_fit(cols_array) - sigma_pixels
bot = polynomial_fit(cols_array) + sigma_pixels

prof_coords_ndarray = []
for counter, entry in enumerate(polynomial_fit):
    prof_coords_ndarray.append([top[counter], mid[counter], bot[counter]])

# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_N2 = []
SD_N2_imsub = []
arr_ndarray_N2 = []
bound_transect_ndarray_N2 = []
bound_transect_ndarray_N2_imsub = []
background_N2 = []
SD_N2_imsub_corrected = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_N2.append(arr)
    bound_transect = im_N2Avg[arr, element]
    bound_transect_imsub = im_N2Rayleigh[arr, element]
    bkg_pts = np.average(bound_transect[:10])
    bkg_pts_imsub = np.average(bound_transect[:10])
    bound_transect_ndarray_N2.append(bound_transect)
    bound_transect_ndarray_N2_imsub.append(bound_transect_imsub)
    transect_summed = np.sum(bound_transect)
    transect_summed_imsub = np.sum(bound_transect_imsub)
    SD_N2.append(transect_summed)
    SD_N2_imsub.append(transect_summed_imsub)
    background_N2.append(bkg_pts)
    SD_N2_imsub_corrected.append(np.asarray(transect_summed_imsub) - bkg_pts)

# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_He = []
SD_He_imsub = []
arr_ndarray_He = []
bound_transect_ndarray_He = []
bound_transect_ndarray_He_imsub = []
background_He = []
SD_He_imsub_corrected = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_He.append(arr)
    bound_transect = im_HeAvg[arr, element]
    bound_transect_imsub = im_Helium_Corrected[arr, element]
    bkg_pts = np.average(bound_transect[:10])
    bound_transect_ndarray_He.append(bound_transect)
    bound_transect_ndarray_He_imsub.append(bound_transect_imsub)
    transect_summed = np.sum(bound_transect)
    transect_summed_imsub = np.sum(bound_transect_imsub)
    SD_He.append(transect_summed)
    SD_He_imsub.append(transect_summed_imsub)
    background_He.append(bkg_pts)
    SD_He_imsub_corrected.append(np.asarray(transect_summed) - bkg_pts)


# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_BKG = []
arr_ndarray_BKG = []
bound_transect_ndarray_BKG = []
background_BKG = []
SD_BKG_ccd_corrected = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    arr_ndarray_BKG.append(arr)
    bound_transect = im_BKGAvg[arr, element]
    bkg_pts = np.average(bound_transect[:10]) * len(bound_transect)
    bound_transect_ndarray_BKG.append(bound_transect)
    transect_summed = np.sum(bound_transect)
    SD_BKG.append(transect_summed)
    background_BKG.append(bkg_pts)
    SD_BKG_ccd_corrected.append(transect_summed - bkg_pts)

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
ax5[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax5[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='blue')
ax5[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='yellow')
ax5[0, 1].set_xlabel('Columns')
ax5[0, 1].set_ylabel('Rows')
ax5[0, 1].set_title('Averaged Nitrogen Image\n Helium Subtracted')
for counter, element in enumerate(arr_ndarray_N2):
    ax5[1, 0].plot(element, bound_transect_ndarray_N2[counter], linestyle='-')
ax5[1, 0].set_xlabel('Rows')
ax5[1, 0].set_ylabel('Intensity (DN)')
ax5[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax5[1, 0].grid(True)
ax5[1, 1].plot(cols_array, SD_N2, linestyle='-', color='red', label='SD: N2')
ax5[1, 1].plot(cols_array, SD_N2_imsub, linestyle='-', color='blue', label='SD: N2 - He')
ax5[1, 1].plot(cols_array, SD_N2_imsub_corrected, linestyle='-', color='green', label='SD: N2 - He - Edge Correction')
ax5[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax5[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax5[1, 1].set_title('Scattering Diagram')
ax5[1, 1].grid(True)
ax5[1, 1].set_yscale('log')
ax5[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F5_N2.pdf', format='pdf')
plt.show()

slope = 0.2056
intercept = -45.2769
Theta = (slope * np.array(cols_array)) + intercept
Rads = (Theta * pi/180)
N_N2 = (1 * 10)/ (0.821 * 298)
a_par=15.00
a_perp = 9.79
Alpha = (1/3) * (a_par + 2 * a_perp)
wav = 663E-9
print(Alpha)
#Save Phase Function

PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Columns'] = cols_array
PhaseFunctionDF['Theta'] = Theta
PhaseFunctionDF['Nitrogen Intensity Raw'] = SD_N2
PhaseFunctionDF['Nitrogen Intensity He Sub'] = SD_N2_imsub
PhaseFunctionDF['Nitrogen Intensity He Sub & Edge Corr.'] = SD_N2_imsub_corrected
PhaseFunctionDF.to_csv(Path_Save + '/SD_Rayleigh.txt')

# 1 + cos^2(theta) function defined

def circular_pol_rayleigh(rads, a, b):
    return a + b * np.cos(rads)**2

def circular_pol_rayleigh2(rads, R, constant):
    return  ((8 * pi**4 *N_N2 * Alpha**2)/(wav**4 * R**2)) * (1 + np.cos(rads)**2) + constant

popt_N2raw, pcov_N2raw = curve_fit(circular_pol_rayleigh, Rads[100:700], SD_N2[100:700])
popt_N2raw2, pcov_N2raw2 = curve_fit(circular_pol_rayleigh2, Rads[100:700], SD_N2[100:700])
popt_N2cor, pcov_N2cor = curve_fit(circular_pol_rayleigh, Rads[100:700], SD_N2_imsub_corrected[100:700])
popt_N2cor2, pcov_N2cor2 = curve_fit(circular_pol_rayleigh2, Rads[100:700], SD_N2_imsub_corrected[100:700])

f6, ax6 = plt.subplots(2, 1, figsize=(10, 7))
ax6[0].plot(Theta, SD_N2, color='green', ls='-', label='Raw Rayleigh N2')
ax6[0].plot(Theta[100:700], SD_N2[100:700], color='yellow', ls='-', label='Subset Rayleigh N2')
ax6[0].plot(Theta, circular_pol_rayleigh(Rads, *popt_N2raw), color='red', ls='-', label='Raw Rayleigh N2 Fit \n y = ' + str('{:.2f}'.format(popt_N2raw[1])) + '$cos(\u0398)^{2}$ + ' + str('{:.2f}'.format(popt_N2raw[0])))
ax6[0].plot(Theta, circular_pol_rayleigh2(Rads, *popt_N2raw2), color='blue', ls='-', label='Raw Rayleigh N2 Theory Fit')
ax6[0].set_xlabel('\u0398')
ax6[0].set_ylabel('Intensity')
ax6[0].set_title('Nitrogen Rayleigh Scattering Diagram for Circularly Polarized Light')
ax6[0].grid(True)
ax6[0].legend(loc=1)
ax6[1].plot(Theta, SD_N2_imsub_corrected, color='green', ls='-', label='Corrected Rayleigh N2')
ax6[1].plot(Theta[100:700], SD_N2_imsub_corrected[100:700], color='yellow', ls='-', label='Corrected Subset Rayleigh N2')
ax6[1].plot(Theta, circular_pol_rayleigh(Rads, *popt_N2cor), color='red', ls='-', label='Corrected Rayleigh N2 Fit \n y = ' + str('{:.2f}'.format(popt_N2cor[1])) + '$cos(\u0398)^{2}$ + ' + str('{:.2f}'.format(popt_N2cor[0])))
ax6[1].plot(Theta, circular_pol_rayleigh2(Rads, *popt_N2cor2), color='blue', ls='-', label='Corrected Rayleigh N2 Theory Fit')
ax6[1].set_xlabel('\u0398')
ax6[1].set_ylabel('Intensity')
ax6[1].set_title('Nitrogen Rayleigh Scattering Diagram for Circularly Polarized Light')
ax6[1].grid(True)
ax6[1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F6_N2Fit.pdf', format='pdf')
plt.show()

print('I_0/R raw^2:', popt_N2raw2[0])
print('I_0/R cor^2:', popt_N2cor2[0])