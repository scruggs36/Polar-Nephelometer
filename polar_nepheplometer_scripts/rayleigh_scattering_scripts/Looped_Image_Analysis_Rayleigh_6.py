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
from scipy.interpolate import pchip_interpolate, PchipInterpolator
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2
import os


# Beam finding images directories
#Path_Bright_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-08-2019/CO2/txt'
# Rayleigh images directories
Path_CO2_Dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/CO2/300s/2darray/CO2_300s_0.5R_Average_Sat Feb 8 2020 4_31_12 PM.txt'
#Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/11.19.19/N2/400s/N2_400s_0.5lamda_0_AVG_.txt'
Path_He_Dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/He/300s/2darray/He_300s_0.5R_Average_Sat Feb 8 2020 5_00_15 PM.txt'
# save directory
Path_Save = '/home/sm3/Desktop/Recent'



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


Raw_CO2 = np.loadtxt(Path_CO2_Dir, delimiter='\t')
#Raw_N2 = np.loadtxt(Path_N2_Dir, delimiter='\t')
Raw_He = np.loadtxt(Path_He_Dir, delimiter='\t')
Corrected_CO2 = Raw_CO2 - Raw_He
#Corrected_CO2 = Raw_CO2
Corrected_CO2[Corrected_CO2 < 0] = 0
#Corrected_N2 = Raw_N2 - Raw_He
#Corrected_N2[Corrected_N2 < 0] = 0

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



# Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
rows = [150, 225]
cols = [50, 800]
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
    im_transect = Corrected_CO2[arr, element]
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
sigma_pixels = 5
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

coords_df = pd.DataFrame()
coords_df['Top'] = top
coords_df['Middle'] = mid
coords_df['Bottom'] = bot
coords_df.to_csv(Path_Save + '/image_coordinates.txt')

# pretty picture plots for background signal corrections
# plots of all averaged images and profile coordinates
# plot of averaged background images
fcal, axcal = plt.subplots(2, 1, figsize=(12, 12))
im_fcala = axcal[0].pcolormesh(Corrected_CO2, cmap='gray')
divider_cala = make_axes_locatable(axcal[0])
cax_cala = divider_cala.append_axes("right", size="5%", pad=0.05)
fcal.colorbar(im_fcala, cax=cax_cala)
axcal[0].set_title('Carbon Dioxide Rayleigh Scattering \nHelium Subtracted')
im_fcalb = axcal[1].pcolormesh(Corrected_CO2, cmap='gray')
axcal[1].plot(cols_array, top, ls='-', color='lawngreen')
axcal[1].plot(cols_array, mid, ls='-', color='red')
axcal[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
axcal[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_calb = make_axes_locatable(axcal[1])
cax_calb = divider_calb.append_axes("right", size="5%", pad=0.05)
fcal.colorbar(im_fcalb, cax=cax_calb)
axcal[1].set_title('Carbon Dioxide Rayleigh Scattering \nHelium Subtracted')
plt.savefig(Path_Save + '/CO2.png', format='png')
#plt.savefig(Path_Save + '/Sample.pdf', format='pdf')
plt.show()

'''
fcal, axcal = plt.subplots(2, 1, figsize=(12, 12))
im_fcala = axcal[0].pcolormesh(Corrected_N2, cmap='gray')
divider_cala = make_axes_locatable(axcal[0])
cax_cala = divider_cala.append_axes("right", size="5%", pad=0.05)
fcal.colorbar(im_fcala, cax=cax_cala)
axcal[0].set_title('Nitrogen Rayleigh Scattering \nHelium Subtracted')
im_fcalb = axcal[1].pcolormesh(Corrected_N2, cmap='gray')
axcal[1].plot(cols_array, top, ls='-', color='lawngreen')
axcal[1].plot(cols_array, mid, ls='-', color='red')
axcal[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
axcal[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_calb = make_axes_locatable(axcal[1])
cax_calb = divider_calb.append_axes("right", size="5%", pad=0.05)
fcal.colorbar(im_fcalb, cax=cax_calb)
axcal[1].set_title('Nitrogen Rayleigh Scattering \nHelium Subtracted')
plt.savefig(Path_Save + '/N2.png', format='png')
#plt.savefig(Path_Save + '/Sample.pdf', format='pdf')
plt.show()
'''


f0, ax0 = plt.subplots(2, 1, figsize=(12, 12))
im_f0a = ax0[0].pcolormesh(Raw_He, cmap='gray')
divider_a = make_axes_locatable(ax0[0])
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
f0.colorbar(im_f0a, cax=cax_a)
ax0[0].set_title('Background: Helium Rayleigh Scattering')
im_f0b = ax0[1].pcolormesh(Raw_He, cmap='gray')
ax0[1].plot(cols_array, top, ls='-', color='lawngreen')
ax0[1].plot(cols_array, mid, ls='-', color='red')
ax0[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
ax0[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_b = make_axes_locatable(ax0[1])
cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
f0.colorbar(im_f0b, cax=cax_b)
ax0[1].set_title('Background: Helium Rayleigh Scattering')
plt.savefig(Path_Save + '/He.png', format='png')
plt.show()

'''
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
    bound_transect = np.array(Corrected_N2[arr, element]).astype('int')
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
'''


# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
He_PN = []
SD_He = []
arr_ndarray_He = []
bound_transect_ndarray_He = []
bound_transect_ndarray_gfit_He = []
bound_transect_ndarray_gfit_bc_He = []
bound_transect_aoc_array_He = []
background_He = []
SD_He_gfit = []
SD_He_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_He[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        He_PN.append(element)
        # raw data wrangling
        arr_ndarray_He.append(arr)
        bound_transect_ndarray_He.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_He.append(transect_summed)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect, p0=[bound_transect[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            #print(popt)
            bound_transect_ndarray_gfit_He.append(gfit)
            gfit_sum_He = np.sum(gfit)
            SD_He_gfit.append(gfit_sum_He)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_He.append(gfit - popt[3])
            gfit_sum_He_bc = np.sum(gfit - popt[3])
            SD_He_gfit_bkg_corr.append(gfit_sum_He_bc)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_He.append(gfit)
            gfit_sum_He = np.nan
            SD_He_gfit.append(gfit_sum_He)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_He.append(gfit)
            gfit_sum_He_bc = np.nan
            SD_He_gfit_bkg_corr.append(gfit_sum_He_bc)



# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
CO2_PN = []
SD_CO2 = []
arr_ndarray_CO2 = []
bound_transect_ndarray_CO2 = []
bound_transect_ndarray_gfit_CO2 = []
bound_transect_ndarray_gfit_bc_CO2 = []
bound_transect_aoc_array_CO2 = []
background_CO2 = []
SD_CO2_gfit = []
SD_CO2_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Corrected_CO2[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        CO2_PN.append(element)
        # data wrangling
        arr_ndarray_CO2.append(arr)
        bound_transect_ndarray_CO2.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_CO2.append(transect_summed)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect, p0=[bound_transect[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            #print(popt)
            bound_transect_ndarray_gfit_CO2.append(gfit)
            gfit_sum_CO2 = np.sum(gfit)
            SD_CO2_gfit.append(gfit_sum_CO2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_CO2.append(gfit - popt[3])
            gfit_sum_bc_CO2 = np.sum(gfit - popt[3])
            SD_CO2_gfit_bkg_corr.append(gfit_sum_bc_CO2)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_CO2.append(gfit)
            gfit_sum_CO2 = np.nan
            SD_CO2_gfit.append(gfit_sum_CO2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_bc_CO2.append(gfit)
            gfit_sum_bc_CO2 = np.nan
            SD_CO2_gfit_bkg_corr.append(gfit_sum_bc_CO2)

SD_He = np.array(SD_He)
SD_He_gfit = np.array(SD_He_gfit)
SD_He_gfit_bkg_corr = np.array(SD_He_gfit_bkg_corr)
#SD_N2 = np.array(SD_N2)
#SD_N2_gfit = np.array(SD_N2_gfit)
#SD_N2_gfit_bkg_corr = np.array(SD_N2_gfit_bkg_corr)
SD_CO2 = np.array(SD_CO2)
SD_CO2_gfit = np.array(SD_CO2_gfit)
SD_CO2_gfit_bkg_corr = np.array(SD_CO2_gfit_bkg_corr)


# plot of the Sample nitrogen subtracted data with bounds
f4, ax4 = plt.subplots(2, 2, figsize=(12, 6))
im_f4 = ax4[0, 0].pcolormesh(Corrected_CO2, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f4.colorbar(im_f4, cax=cax)
ax4[0, 0].set_title('Averaged CO2 Image')
ax4[0, 1].pcolormesh(Corrected_CO2, cmap='gray')
ax4[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax4[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].set_xlabel('Columns')
ax4[0, 1].set_ylabel('Rows')
ax4[0, 1].set_title('Averaged CO2 Image \n Helium Subtracted')
for counter, element in enumerate(arr_ndarray_CO2):
    ax4[1, 0].plot(element, bound_transect_ndarray_CO2[counter], linestyle='-')
ax4[1, 0].set_xlabel('Rows')
ax4[1, 0].set_ylabel('Intensity (DN)')
ax4[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax4[1, 0].grid(True)
ax4[1, 1].plot(CO2_PN, SD_CO2, linestyle='-', color='red', label='SD: CO2 - He (Riemann)')
ax4[1, 1].plot(CO2_PN, SD_CO2_gfit, linestyle='-', color='blue', label='SD: CO2 - He (Gaussian Fit)')
ax4[1, 1].plot(CO2_PN, SD_CO2_gfit_bkg_corr, linestyle='-', color='green', label='SD: CO2 - He (Gaussian Fit -  Bkg Constant)')
ax4[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax4[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax4[1, 1].set_title('Scattering Diagram')
ax4[1, 1].grid(True)
ax4[1, 1].set_yscale('log')
ax4[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/CO2_PF.png', format='png')
plt.show()

'''
# plot of the backgound subtracted data with bounds
f5, ax5 = plt.subplots(2, 2, figsize=(12, 6))
im_f5 = ax5[0, 0].pcolormesh(Corrected_N2, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax5[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f5.colorbar(im_f5, cax=cax)
ax5[0, 0].set_title('Averaged Nitrogen Image')
ax5[0, 1].pcolormesh(Corrected_N2, cmap='gray')
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
ax5[1, 1].plot(N2_PN, SD_N2, linestyle='-', color='red', label='SD: N2 - He (Riemann)')
ax5[1, 1].plot(N2_PN, SD_N2_gfit, linestyle='-', color='blue', label='SD: N2 - He (Gaussian Fit)')
ax5[1, 1].plot(N2_PN, SD_N2_gfit_bkg_corr, linestyle='-', color='green', label='SD: N2 - He (Gaussian Fit - Bkg Constant)')
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
slope = 0.2049
intercept = -2.7594
# columns to theta
theta_N2 = (np.array(CO2_PN) * slope) + intercept
print('N2 angular range:', [theta_N2[0], theta_N2[-1]])
rads_N2 = theta_N2 * pi/180.0
rads_N2 = rads_N2
print('ROI range', [(slope * cols[0]) + intercept, (slope * cols[1]) + intercept])

# Save Phase Function, the data saved here has no subtractions/corrections applied to them, each is raw signal
# note the CCD Noise cannot be backed out, as we would have to cover the lens to do it, if at some point we take
# covered images we could do it...
DF_Headers = ['CO2 Columns', 'He Columns', 'CO2 Theta', 'He Theta', 'CO2 Intensity', 'CO2 Intensity gfit', 'CO2 Intensity gfit corr', 'He Intensity', 'He Intensity gfit', 'He Intensity gfit corr', 'N2 Intensity', 'N2 Intensity gfit', 'N2 Intensity gfit corr']
DF_CO2_C = pd.DataFrame(CO2_PN)
DF_He_C = pd.DataFrame(He_PN)
DF_CO2_Theta = pd.DataFrame([(x * slope) + intercept for x in CO2_PN])
DF_He_Theta = pd.DataFrame([(x * slope) + intercept for x in He_PN])
DF_PF_CO2 = pd.DataFrame(SD_CO2)
DF_PF_CO2_gfit = pd.DataFrame(SD_CO2_gfit)
DF_PF_CO2_gfit_bkg_corr = pd.DataFrame(SD_CO2_gfit_bkg_corr)
DF_PF_He = pd.DataFrame(SD_He)
DF_PF_He_gfit = pd.DataFrame(SD_He_gfit)
DF_PF_He_gfit_bkg_corr = pd.DataFrame(SD_He_gfit_bkg_corr)
#DF_PF_N2 = pd.DataFrame(SD_N2)
#DF_PF_N2_gfit = pd.DataFrame(SD_N2_gfit)
#DF_PF_N2_gfit_bkg_corr = pd.DataFrame(SD_N2_gfit_bkg_corr)
DF_PF_N2 = pd.DataFrame(np.full(shape=len(SD_CO2), fill_value=np.nan))
DF_PF_N2_gfit = pd.DataFrame(np.full(shape=len(SD_CO2_gfit), fill_value=np.nan))
DF_PF_N2_gfit_bkg_corr = pd.DataFrame(np.full(shape=len(SD_CO2_gfit_bkg_corr), fill_value=np.nan))
PhaseFunctionDF = pd.concat([DF_CO2_C, DF_He_C, DF_CO2_Theta, DF_He_Theta, DF_PF_CO2, DF_PF_CO2_gfit, DF_PF_CO2_gfit_bkg_corr, DF_PF_He, DF_PF_He_gfit, DF_PF_He_gfit_bkg_corr, DF_PF_N2, DF_PF_N2_gfit, DF_PF_N2_gfit_bkg_corr], ignore_index=False, axis=1)
PhaseFunctionDF.columns = DF_Headers
PhaseFunctionDF.to_csv(Path_Save + '/SD_Rayleigh.txt')



f6, ax6 = plt.subplots(2, 2, figsize=(12, 7))
ax6[0, 0].plot(arr_ndarray_CO2[50], bound_transect_ndarray_CO2[50], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[50][:10])*len(bound_transect_ndarray_CO2[50])))
ax6[0, 0].plot(arr_ndarray_CO2[50], bound_transect_ndarray_gfit_bc_CO2[50], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_CO2[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_CO2[50][:10])*len(bound_transect_ndarray_gfit_bc_CO2[50])))
ax6[0, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 0].set_title('Profiles Compared')
ax6[0, 0].grid(True)
ax6[0, 0].legend(loc=1)
ax6[0, 1].plot(arr_ndarray_CO2[200], bound_transect_ndarray_CO2[200], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[200][:10])*len(bound_transect_ndarray_CO2[200])))
ax6[0, 1].plot(arr_ndarray_CO2[200], bound_transect_ndarray_gfit_bc_CO2[200], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_CO2[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_CO2[200][:10])*len(bound_transect_ndarray_gfit_bc_CO2[200])))
ax6[0, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 1].set_title('Profiles Compared')
ax6[0, 1].grid(True)
ax6[0, 1].legend(loc=1)
ax6[1, 0].plot(arr_ndarray_CO2[300], bound_transect_ndarray_CO2[300], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[300][:10])*len(bound_transect_ndarray_CO2[300])))
ax6[1, 0].plot(arr_ndarray_CO2[300], bound_transect_ndarray_gfit_bc_CO2[300], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_CO2[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_CO2[300][:10])*len(bound_transect_ndarray_gfit_bc_CO2[300])))
ax6[1, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 0].set_title('Profiles Compared')
ax6[1, 0].grid(True)
ax6[1, 0].legend(loc=1)
ax6[1, 1].plot(arr_ndarray_CO2[350], bound_transect_ndarray_CO2[350], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[350][:10])*len(bound_transect_ndarray_CO2[350])))
ax6[1, 1].plot(arr_ndarray_CO2[350], bound_transect_ndarray_gfit_bc_CO2[350], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_gfit_bc_CO2[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_gfit_bc_CO2[350][:10])*len(bound_transect_ndarray_gfit_bc_CO2[350])))
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

'''
vertical_path = '/home/austen/Desktop/2019-10-11_Analysis/CO2/60s/lamda_0/SD_Rayleigh.txt'
horizontal_path = '/home/austen/Desktop/2019-10-11_Analysis/CO2/60s/lamda_0.5/SD_Rayleigh.txt'

CO2_V_DF = pd.read_csv(vertical_path, delimiter=',', header=0)
CO2_H_DF = pd.read_csv(horizontal_path, delimiter=',', header=0)
CO2_V_Theta = np.array(CO2_V_DF['CO2 Theta'])
CO2_H_Theta = np.array(CO2_H_DF['CO2 Theta'])
He_V_Theta = np.array(CO2_V_DF['He Theta'])
He_H_Theta = np.array(CO2_H_DF['He Theta'])
CO2_V_PF = np.array(CO2_V_DF['CO2 Intensity'])
CO2_H_PF = np.array(CO2_H_DF['CO2 Intensity'])
He_V_PF = np.array(CO2_V_DF['He Intensity'])
He_H_PF = np.array(CO2_H_DF['He Intensity'])

del_array_He = []
for counter, element in enumerate(np.isnan(He_H_PF)):
    if element == True:
        del_array_He.append(counter)


del_array_CO2 = []
for counter, element in enumerate(np.isnan(CO2_H_PF)):
    if element == True:
        del_array_CO2.append(counter)

He_H_PF2 = np.delete(He_H_PF, del_array_He, axis=0)
He_H_Theta2 = np.delete(He_H_Theta, del_array_He, axis=0)
CO2_H_PF2 = np.delete(CO2_H_PF, del_array_CO2, axis=0)
CO2_H_Theta2 = np.delete(CO2_H_Theta, del_array_CO2, axis=0)

del_array_He = []
for counter, element in enumerate(np.isnan(He_V_PF)):
    if element == True:
        del_array_He.append(counter)


del_array_CO2 = []
for counter, element in enumerate(np.isnan(CO2_V_PF)):
    if element == True:
        del_array_CO2.append(counter)

He_V_PF2 = np.delete(He_V_PF, del_array_He, axis=0)
He_V_Theta2 = np.delete(He_V_Theta, del_array_He, axis=0)
CO2_V_PF2 = np.delete(CO2_V_PF, del_array_CO2, axis=0)
CO2_V_Theta2 = np.delete(CO2_V_Theta, del_array_CO2, axis=0)

f8, ax8 = plt.subplots(1, 3, figsize=(12, 6))
ax8[0].plot(CO2_H_Theta2, CO2_H_PF2, label='Horizontal: Raw PF')
ax8[0].plot(CO2_V_Theta2, CO2_V_PF2, label='Vertical: Raw PF')
ax8[0].set_title('Raw Phase Functions')
ax8[0].set_xlabel('\u00b0')
ax8[0].set_ylabel('Intensity')
ax8[0].legend(loc=1)
ax8[0].grid(True)

ax8[1].plot(CO2_H_Theta, CO2_H_PF - He_H_PF, label='Horizontal: CO2 - He')
ax8[1].plot(CO2_V_Theta, CO2_V_PF - He_V_PF, label='Vertical: CO2 - He')
ax8[1].set_title('Corrected Phase Functions')
ax8[1].set_xlabel('\u00b0')
ax8[1].set_ylabel('Intensity')
ax8[1].legend(loc=1)
ax8[1].grid(True)

ax8[2].plot(He_H_Theta2,  He_H_PF2, label='Horizontal: Helium')
ax8[2].plot(He_V_Theta2,  He_V_PF2, label='Vertical: Helium')
ax8[2].set_title('Helium Phase Functions')
ax8[2].set_xlabel('\u00b0')
ax8[2].set_ylabel('Intensity')
ax8[2].legend(loc=1)
ax8[2].grid(True)

plt.tight_layout()
plt.show()
'''