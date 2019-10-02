'''
Austen K. Scruggs
10-10-2018
Desctription: This code averages and 12bit 2darrays (containing 12bit image data) , then subtracts the averaged 2darrays
from their corresponding background 2darrays. This is the update that was necessary to apply from the Mono12 update. This is for PSL samples
not an analysis of Rayleigh scattering images.
'''

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from PIL import Image
from scipy.interpolate import pchip_interpolate
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2
import os


# Beam finding images directories
#Path_Bright_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-08-2019/CO2/txt'

# Sample images directory
Path_Samp_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/2019-09-28/PSL600/3s/PSL600_3s_0.5lamda_0_AVG_.txt'
# Rayleigh images directories
Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/2019-09-28/N2/3s/N2_3s_0.5lamda_0_AVG_.txt'
# save directory
Path_Save = '/home/austen/Desktop/2019-09-26_Analysis'



# averaging nd-arrays function
def Ndarray_Average(directory):
    nd_array = []
    file_list = os.listdir(directory)
    for file in file_list:
        print(file)
        nd_arr_element = np.loadtxt(directory + '/' + file, dtype='int', delimiter='\t')
        nd_array.append(nd_arr_element)
    nd_np_array = np.array(nd_array)
    nd_arr_avg = np.transpose(np.squeeze(nd_np_array.mean(axis=0, dtype='int', keepdims=True), axis=0))
    print(nd_arr_avg.shape)
    return nd_arr_avg


Raw_Sample = np.loadtxt(Path_Samp_Dir, delimiter='\t')
Raw_N2 = np.loadtxt(Path_N2_Dir, delimiter='\t')
Corrected_Sample = Raw_Sample - Raw_N2
Corrected_Sample[Corrected_Sample < 0] = 0


# averaging sample, n2, he, bkg
#Bright = Ndarray_Average(Path_Bright_Dir)
#np.savetxt(Path_Save + '/' + 'Bright.txt', Bright)
'''
Raw_Sample = Ndarray_Average(Path_Samp_Dir)
np.savetxt(Path_Save + '/' + 'Raw_Sample.txt', Raw_Sample, delimiter=',')

Raw_N2 = Ndarray_Average(Path_N2_Dir)
np.savetxt(Path_Save + '/' + 'Raw_N2.txt', Raw_N2, delimiter=',')


# sample - n2 - bkg, and n2 - he - bkg
#Corrected_Sample = np.subtract(Raw_Sample, Raw_N2)
Corrected_Sample = Raw_Sample - Raw_N2
Corrected_Sample[Corrected_Sample < 0] = 0
np.savetxt(Path_Save + '/' + 'Corrected_Sample.txt', Corrected_Sample)
'''

'''
# this saves the corrected ndarray as a txt file, and saves it as a jpg
Corrected_Sample_im = np.loadtxt(Path_Save + '/' + 'Corrected_Sample.txt').astype(dtype=np.uint16)
Raw_N2_im = np.loadtxt(Path_Save + '/' + 'Raw_N2.txt').astype(dtype=np.uint16)
'''
# cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
'''
Raw_BKG_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Raw_BKG.png'), cv2.COLOR_BGR2GRAY)
Corrected_Sample_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Corrected_Sample.jpg'), cv2.COLOR_BGR2GRAY)
Corrected_Bright_CO2_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Corrected_Bright_CO2.png'), cv2.COLOR_BGR2GRAY)
Corrected_CO2_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Corrected_CO2.png'), cv2.COLOR_BGR2GRAY)
Corrected_N2_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Corrected_N2.png'), cv2.COLOR_BGR2GRAY)
Corrected_Bright_N2_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Corrected_Bright_N2.png'), cv2.COLOR_BGR2GRAY)
Corrected_He_PNG = cv2.cvtColor(cv2.imread(Path_Save + '/' + 'Corrected_He.png'), cv2.COLOR_BGR2GRAY)
'''


# Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
rows = [400, 600]
cols = [230, 1060]
cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
#ROI = im[rows[0]:rows[1], cols[0]:cols[1]]

def gaussian(x, a, b, c, d):
    return d + (abs(a) * np.exp((-1 * (x - b) ** 2) / (2 * c ** 2)))


def rayleigh_scattering (x, a, b):
    return b + (a * np.square(np.cos(x)))

# find coordinates based on sample - N2 scattering averaged image (without corrections)
row_max_index_array = []
for element in cols_array:
    arr = np.arange(rows[0], rows[1], 1).astype(int)
    im_transect = Corrected_Sample[arr, element]
    index_nosub = np.argmax(im_transect)
    row_max_index_array.append(index_nosub + rows[0])

# polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
tuner = len(cols_array)
iterator = round(len(cols_array)/tuner)
# based on the division in iterator, sometimes it needs an extra iteration to capture the rest of the coord points
#iterator = round(len(cols_array)/tuner) + 1
print(iterator)
mid = []
top = []
bot = []
sigma_pixels = 30
for counter, element in enumerate(range(iterator)):
    if counter < iterator:
        print(counter)
        x = cols_array[(counter) * tuner: (counter + 1) * tuner]
        y = row_max_index_array[(counter) * tuner: (counter + 1) * tuner]
        print(x)
        #print(y)
        polynomial_fit = np.poly1d(np.polyfit(x, y, deg=6))
        #sigma_pixels = 20
        [mid.append(polynomial_fit(element)) for element in x]
        [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
        [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]
    if counter == iterator:
        print(counter)
        x = cols_array[(counter) * tuner: len(cols_array)]
        y = row_max_index_array[(counter) * tuner: len(row_max_index_array)]
        print(x)
        # print(y)
        polynomial_fit = np.poly1d(np.polyfit(x, y, deg=6))
        #sigma_pixels = 20
        [mid.append(polynomial_fit(element)) for element in x]
        [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
        [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]

#print(len(mid))
#print(mid)
#top = np.array(top).ravel()
#print(top.shape)
#print(top)
#bot = np.array(bot).ravel()
#print(bot.shape)
#print(bot)


# pretty picture plots for background signal corrections
# plots of all averaged images and profile coordinates
# plot of averaged background images
fcal, axcal = plt.subplots(figsize=(12, 7))
im_fcal = axcal.pcolormesh(Corrected_Sample, cmap='gray')
axcal.plot(cols_array, top, ls='-', color='lawngreen')
axcal.plot(cols_array, mid, ls='-', color='red')
axcal.plot(cols_array, row_max_index_array, ls='-', color='purple')
axcal.plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(axcal)
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
fcal.colorbar(im_fcal, cax=cax_a)
axcal.set_title('Sample: 900nm Polystyrene Latex Spheres \n at 0\u00b0 Retardance')
plt.savefig(Path_Save + '/Sample.png', format='png')
#plt.savefig(Path_Save + '/Sample.pdf', format='pdf')
plt.show()


f0, ax0 = plt.subplots(figsize=(12, 7))
im_f0a = ax0.pcolormesh(Raw_N2, cmap='gray')
ax0.plot(cols_array, top, ls='-', color='lawngreen')
ax0.plot(cols_array, mid, ls='-', color='red')
ax0.plot(cols_array, row_max_index_array, ls='-', color='purple')
ax0.plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(ax0)
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
f0.colorbar(im_f0a, cax=cax_a)
ax0.set_title('Background: Nitrogen Rayleigh Scattering \n at 0\u00b0 Retardance')
plt.savefig(Path_Save + '/N2.png', format='png')
#plt.savefig(Path_Save + '/N2.pdf', format='pdf')
plt.show()


# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
N2_PN = []
SD_N2 = []
arr_ndarray_N2 = []
bound_transect_ndarray_N2 = []
bound_transect_ndarray_gfit_N2 = []
bound_transect_ndarray_gfit_bc_N2 = []
bound_transect_aoc_array_N2 = []
background_N2 = []
SD_N2_gfit = []
SD_N2_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_N2[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        N2_PN.append(element)
        # raw data wrangling
        arr_ndarray_N2.append(arr)
        bound_transect_ndarray_N2.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_N2.append(transect_summed)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect, p0=[bound_transect[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            #print(popt)
            bound_transect_ndarray_gfit_N2.append(gfit)
            gfit_sum_N2 = np.sum(gfit)
            SD_N2_gfit.append(gfit_sum_N2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_N2.append(gfit - popt[3])
            gfit_sum_N2_bc = np.sum(gfit - popt[3])
            SD_N2_gfit_bkg_corr.append(gfit_sum_N2_bc)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_N2.append(gfit)
            gfit_sum_N2 = np.nan
            SD_N2_gfit.append(gfit_sum_N2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_N2.append(gfit)
            gfit_sum_N2_bc = np.nan
            SD_N2_gfit_bkg_corr.append(gfit_sum_N2_bc)

# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
Samp_PN = []
SD_Samp = []
arr_ndarray_Samp = []
bound_transect_ndarray_Samp = []
bound_transect_ndarray_gfit_Samp = []
bound_transect_ndarray_gfit_bc_Samp = []
bound_transect_aoc_array_Samp = []
background_Samp = []
SD_Samp_gfit = []
SD_Samp_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Corrected_Sample[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        Samp_PN.append(element)
        # data wrangling
        arr_ndarray_Samp.append(arr)
        bound_transect_ndarray_Samp.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_Samp.append(transect_summed)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect, p0=[bound_transect[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            #print(popt)
            bound_transect_ndarray_gfit_Samp.append(gfit)
            gfit_sum_Samp = np.sum(gfit)
            SD_Samp_gfit.append(gfit_sum_Samp)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_Samp.append(gfit - popt[3])
            gfit_sum_bc_Samp = np.sum(gfit - popt[3])
            SD_Samp_gfit_bkg_corr.append(gfit_sum_bc_Samp)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_Samp.append(gfit)
            gfit_sum_Samp = np.nan
            SD_Samp_gfit.append(gfit_sum_Samp)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_Samp.append(gfit)
            gfit_sum_bc_Samp = np.nan
            SD_Samp_gfit_bkg_corr.append(gfit_sum_bc_Samp)


# plot of the Sample nitrogen subtracted data with bounds
f4, ax4 = plt.subplots(2, 2, figsize=(12, 6))
im_f4 = ax4[0, 0].pcolormesh(Corrected_Sample, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f4.colorbar(im_f4, cax=cax)
ax4[0, 0].set_title('Averaged Sample Image')
ax4[0, 1].pcolormesh(Corrected_Sample, cmap='gray')
ax4[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax4[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].set_xlabel('Columns')
ax4[0, 1].set_ylabel('Rows')
ax4[0, 1].set_title('Averaged Sample Image \n Nitrogen Subtracted')
for counter, element in enumerate(arr_ndarray_Samp):
    ax4[1, 0].plot(element, bound_transect_ndarray_Samp[counter], linestyle='-')
ax4[1, 0].set_xlabel('Rows')
ax4[1, 0].set_ylabel('Intensity (DN)')
ax4[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax4[1, 0].grid(True)
ax4[1, 1].plot(Samp_PN, SD_Samp, linestyle='-', color='red', label='SD: Sample')
ax4[1, 1].plot(Samp_PN, SD_Samp_gfit_bkg_corr, linestyle='-', color='blue', label='SD: Sample - N2')
#ax4[1, 1].plot(cols_array, SD_Samp_imsub_corrected, linestyle='-', color='green', label='SD: Sample - N2 - Edge Corrected')
ax4[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax4[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax4[1, 1].set_title('Scattering Diagram')
ax4[1, 1].grid(True)
ax4[1, 1].set_yscale('log')
ax4[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/Sample_PF.png', format='png')
plt.show()



# plot of the backgound subtracted data with bounds
f5, ax5 = plt.subplots(2, 2, figsize=(12, 6))
im_f5 = ax5[0, 0].pcolormesh(Raw_N2, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax5[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f5.colorbar(im_f5, cax=cax)
ax5[0, 0].set_title('Averaged Nitrogen Image')
ax5[0, 1].pcolormesh(Raw_N2, cmap='gray')
ax5[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax5[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax5[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax5[0, 1].set_xlabel('Columns')
ax5[0, 1].set_ylabel('Rows')
ax5[0, 1].set_title('Averaged Nitrogen Image')
for counter, element in enumerate(arr_ndarray_N2):
    ax5[1, 0].plot(element, bound_transect_ndarray_N2[counter], linestyle='-')
ax5[1, 0].set_xlabel('Rows')
ax5[1, 0].set_ylabel('Intensity (DN)')
ax5[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax5[1, 0].grid(True)
ax5[1, 1].plot(N2_PN, SD_N2, linestyle='-', color='red', label='SD: N2')
ax5[1, 1].plot(N2_PN, SD_N2_gfit_bkg_corr, linestyle='-', color='blue', label='SD: N2 BKG Correction')
#ax5[1, 1].plot(cols_array, SD_N2_imsub_corrected, linestyle='-', color='green', label='SD: N2 - He - Edge Correction')
ax5[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax5[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax5[1, 1].set_title('Scattering Diagram')
ax5[1, 1].grid(True)
ax5[1, 1].set_yscale('log')
ax5[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/N2_PF.png', format='png')
plt.show()
'''
# columns to theta
slope = 0.2077
intercept = -45.0818
# columns to theta
theta_N2 = (np.array(N2_PN) * slope) + intercept
print('N2 angular range:', [theta_N2[0], theta_N2[-1]])
rads_N2 = theta_N2 * pi/180.0
rads_N2 = rads_N2
print('ROI range', [(slope * cols[0]) + intercept, (slope * cols[1]) + intercept])
'''
# Save Phase Function, the data saved here has no subtractions/corrections applied to them, each is raw signal
# note the CCD Noise cannot be backed out, as we would have to cover the lens to do it, if at some point we take
# covered images we could do it...
DF_Headers = ['Sample Columns', 'N2 Columns', 'Sample Intensity', 'N2 Intensity']
DF_S_C = pd.DataFrame(Samp_PN)
DF_N2_C = pd.DataFrame(N2_PN)
#DF_PF_S = pd.DataFrame(SD_Samp)
DF_PF_S = pd.DataFrame(SD_Samp)
DF_PF_N2 = pd.DataFrame(SD_N2)
PhaseFunctionDF = pd.concat([DF_S_C, DF_N2_C, DF_PF_S, DF_PF_N2], ignore_index=False, axis=1)
PhaseFunctionDF.columns = DF_Headers
PhaseFunctionDF.to_csv(Path_Save + '/SD_Particle.txt')



f6, ax6 = plt.subplots(2, 2, figsize=(12, 7))
ax6[0, 0].plot(arr_ndarray_Samp[50], bound_transect_ndarray_Samp[50], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[50][:10])*len(bound_transect_ndarray_Samp[50])))
ax6[0, 0].plot(arr_ndarray_Samp[50], bound_transect_ndarray_gfit_bc_Samp[50], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_Samp[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_Samp[50][:10])*len(bound_transect_ndarray_gfit_bc_Samp[50])))
ax6[0, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 0].set_title('Profiles Compared')
ax6[0, 0].grid(True)
ax6[0, 0].legend(loc=1)
ax6[0, 1].plot(arr_ndarray_Samp[200], bound_transect_ndarray_Samp[200], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[200][:10])*len(bound_transect_ndarray_Samp[200])))
ax6[0, 1].plot(arr_ndarray_Samp[200], bound_transect_ndarray_gfit_bc_Samp[200], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_Samp[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_Samp[200][:10])*len(bound_transect_ndarray_gfit_bc_Samp[200])))
ax6[0, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 1].set_title('Profiles Compared')
ax6[0, 1].grid(True)
ax6[0, 1].legend(loc=1)
ax6[1, 0].plot(arr_ndarray_Samp[300], bound_transect_ndarray_Samp[300], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[300][:10])*len(bound_transect_ndarray_Samp[300])))
ax6[1, 0].plot(arr_ndarray_Samp[300], bound_transect_ndarray_gfit_bc_Samp[300], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_Samp[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_Samp[300][:10])*len(bound_transect_ndarray_gfit_bc_Samp[300])))
ax6[1, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 0].set_title('Profiles Compared')
ax6[1, 0].grid(True)
ax6[1, 0].legend(loc=1)
ax6[1, 1].plot(arr_ndarray_N2[350], bound_transect_ndarray_Samp[350], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[350][:10])*len(bound_transect_ndarray_N2[350])))
ax6[1, 1].plot(arr_ndarray_N2[350], bound_transect_ndarray_gfit_bc_Samp[350], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_Samp[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_Samp[350][:10])*len(bound_transect_ndarray_gfit_bc_Samp[350])))
ax6[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 1].set_title('Profiles Compared')
ax6[1, 1].grid(True)
ax6[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/Profiles.png', format='png')
plt.show()


'''
f7, ax7 = plt.subplots(1, 3, figsize=(36, 7))
ax7[0].plot(Samp_PN, SD_Samp, linestyle='-', color='black', label='Raw 900nm PSL Scattering')
ax7[0].plot(N2_PN, SD_N2, linestyle='-', color='blue', label='N2 Scattering')
ax7[0].set_title('Scattering Contributions to Raw Sample Scattering Diagram')
ax7[0].set_ylabel('Intensity (DN)')
ax7[0].set_xlabel('Profile Number (Column Number)')
ax7[0].grid(True)
ax7[0].legend(loc=1)
ax7[1].plot(Samp_PN, SD_Samp_gfit, linestyle='-', color='black', label='Raw 900nm PSL Scattering')
ax7[1].plot(N2_PN, SD_N2_gfit, linestyle='-', color='blue', label='N2 Scattering')
ax7[1].set_title('Scattering Contributions to Sample Scattering Diagram via Gaussian Fitting')
ax7[1].set_ylabel('Intensity (DN)')
ax7[1].set_xlabel('Profile Number (Column Number)')
ax7[1].grid(True)
ax7[1].legend(loc=1)
ax7[2].plot(Samp_PN, SD_Samp_gfit_bkg_corr, linestyle='-', color='black', label='PSL Scattering - N2 Scattering')
ax7[2].plot(N2_PN, SD_N2_gfit_bkg_corr, linestyle='-', color='blue', label='$N_2$ Scattering - He Scattering')
ax7[2].set_title('Scattering Contributions to Sample \n Scattering Diagram via Gaussian Fitting and Edge Correction ')
ax7[2].set_ylabel('Intensity (DN)')
ax7[2].set_xlabel('Profile Number (Column Number)')
ax7[2].grid(True)
ax7[2].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/Contributions.png', format='png')
plt.show()
'''

