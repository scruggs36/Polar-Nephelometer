'''
Austen K. Scruggs
10-10-2018
Desctription: This code averages and 12bit 2darrays (containing 12bit image data) , then subtracts the averaged 2darrays
from their corresponding background 2darrays. This is the update that was necessary to apply from the Mono12 update.
'''

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_Samp_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-16-2019/600nm_PSL/txt/3s'
Path_CO2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-16-2019/CO2/txt/3s'
Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-16-2019/N2/txt/3s'
Path_He_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-16-2019/He/txt/3s'
Path_BKG_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-16-2019/BKG/txt/3s'
#Path_Dark_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/01-23-2019/900/BKG/T1'
Path_Save = '/home/austen/Documents'

# averaging nd-arrays function
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



# averaging sample, n2, he, bkg
Raw_Sample = Ndarray_Average(Path_Samp_Dir)
Raw_CO2 = Ndarray_Average(Path_CO2_Dir)
Raw_N2 = Ndarray_Average(Path_N2_Dir)
Raw_He = Ndarray_Average(Path_He_Dir)
Raw_BKG = Ndarray_Average(Path_BKG_Dir)



# sample - n2 - bkg, and n2 - he - bkg
Corrected_Sample = np.subtract(Raw_Sample, Raw_N2)
Corrected_Sample[Corrected_Sample < 0] = 0
Corrected_CO2 = np.subtract(Raw_CO2, Raw_He)
Corrected_CO2[Corrected_CO2 < 0] = 0
Corrected_N2 = np.subtract(Raw_N2, Raw_He)
Corrected_N2[Corrected_N2 < 0] = 0
Corrected_He = np.subtract(Raw_He, Raw_BKG)
Corrected_He[Corrected_He < 0] = 0


# Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
rows = [550, 700]
cols = [245, 1050]
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
    im_transect = Corrected_N2[arr, element]
    index_nosub = np.argmax(im_transect)
    row_max_index_array.append(index_nosub + rows[0])

# polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
polynomial_fit = np.poly1d(np.polyfit(cols_array, row_max_index_array, deg=2))
sigma_pixels = 20
mid = polynomial_fit(cols_array)
top = polynomial_fit(cols_array) - sigma_pixels
bot = polynomial_fit(cols_array) + sigma_pixels

# pretty picture plots for background signal corrections
# plots of all averaged images and profile coordinates
# plot of averaged background images
f0, ax0 = plt.subplots(1, 2, figsize=(12, 7))
im_f0a = ax0[0].pcolormesh(Raw_BKG, cmap='gray')
im_f0b = ax0[1].pcolormesh(Raw_BKG, cmap='gray')
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
plt.savefig(Path_Save + '/BKG.png', format='png')
plt.show()

# pretty picture plots for background signal corrections
# plot of averaged Helium images
f1, ax1 = plt.subplots(1, 2, figsize=(12, 7))
im_f1a = ax1[0].pcolormesh(Raw_He, cmap='gray')
im_f1b = ax1[1].pcolormesh(Corrected_He, cmap='gray')
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
plt.savefig(Path_Save + '/He.png', format='png')
plt.show()

# pretty picture plots for background signal corrections
# plot of the N2 corrected (helium and bkg) subtracted data
f2, ax2 = plt.subplots(1, 2, figsize=(12, 7))
im_f2a = ax2[0].pcolormesh(Raw_N2, cmap='gray')
im_f2b = ax2[1].pcolormesh(Corrected_N2, cmap='gray')
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
plt.savefig(Path_Save + '/N2.png', format='png')
plt.show()

# pretty picture plots for background signal corrections
# plot of averaged sample images
f3, ax3 = plt.subplots(1, 2, figsize=(12, 7))
im_f3a = ax3[0].pcolormesh(Raw_Sample, cmap='gray')
im_f3b = ax3[1].pcolormesh(Corrected_Sample, cmap='gray')
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
ax3[0].set_title('Particle Image')
ax3[1].set_title('Nitrogen Corrected Particle Image')
plt.savefig(Path_Save + '/Sample.png', format='png')
plt.show()

# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
CO2_PN = []
CO2_PN_imsub = []
SD_CO2 = []
SD_CO2_imsub = []
arr_ndarray_CO2 = []
bound_transect_ndarray_CO2 = []
bound_transect_ndarray_CO2_imsub = []
bound_transect_ndarray_gfit_CO2 = []
bound_transect_ndarray_gfit_CO2_imsub = []
bound_transect_aoc_array_CO2 = []
bound_transect_aoc_array_CO2_imsub = []
background_CO2 = []
SD_CO2_gfit = []
SD_CO2_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_CO2[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        CO2_PN.append(element)
        CO2_PN_imsub.append(element)
        # raw data wrangling
        arr_ndarray_CO2.append(arr)
        bound_transect_ndarray_CO2.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_CO2.append(transect_summed)
        # raw data wrangling with background subtraction
        bound_transect_imsub = np.array(Corrected_CO2[arr, element]).astype('int') - int(round(np.mean(np.array(Corrected_CO2[arr, element])[-5:])))
        bound_transect_ndarray_CO2_imsub.append(bound_transect_imsub)
        transect_summed_imsub = np.sum(bound_transect_imsub)
        SD_CO2_imsub.append(transect_summed_imsub)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect_imsub, p0=[bound_transect_imsub[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            #print(popt)
            bound_transect_ndarray_gfit_CO2.append(gfit)
            gfit_sum_CO2 = np.sum(gfit)
            SD_CO2_gfit.append(gfit_sum_CO2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_CO2_imsub.append(gfit - popt[3])
            gfit_sum_CO2_imsub = np.sum(gfit - popt[3])
            SD_CO2_gfit_bkg_corr.append(gfit_sum_CO2_imsub)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_CO2.append(gfit)
            gfit_sum_CO2 = np.nan
            SD_CO2_gfit.append(gfit_sum_CO2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_CO2_imsub.append(gfit)
            gfit_sum_CO2_imsub = np.nan
            SD_CO2_gfit_bkg_corr.append(gfit_sum_CO2_imsub)


# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
N2_PN = []
N2_PN_imsub = []
SD_N2 = []
SD_N2_imsub = []
arr_ndarray_N2 = []
bound_transect_ndarray_N2 = []
bound_transect_ndarray_N2_imsub = []
bound_transect_ndarray_gfit_N2 = []
bound_transect_ndarray_gfit_N2_imsub = []
bound_transect_aoc_array_N2 = []
bound_transect_aoc_array_N2_imsub = []
background_N2 = []
SD_N2_gfit = []
SD_N2_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_N2[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        N2_PN_imsub.append(element)
        N2_PN.append(element)
        # raw data wrangling
        arr_ndarray_N2.append(arr)
        bound_transect_ndarray_N2.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_N2.append(transect_summed)
        # raw data wrangling with background subtraction
        bound_transect_imsub = np.array(Corrected_N2[arr, element]).astype('int') - int(round(np.mean(np.array(Corrected_N2[arr, element])[-5:])))
        bound_transect_ndarray_N2_imsub.append(bound_transect_imsub)
        transect_summed_imsub = np.sum(bound_transect_imsub)
        SD_N2_imsub.append(transect_summed_imsub)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect_imsub, p0=[bound_transect_imsub[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            bound_transect_ndarray_gfit_N2.append(gfit)
            gfit_sum_N2 = np.sum(gfit)
            SD_N2_gfit.append(gfit_sum_N2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_N2_imsub.append(gfit - popt[3])
            gfit_sum_N2_imsub = np.sum(gfit - popt[3])
            SD_N2_gfit_bkg_corr.append(gfit_sum_N2_imsub)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_N2.append(gfit)
            gfit_sum_N2 = np.nan
            SD_N2_gfit.append(gfit_sum_N2)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_N2_imsub.append(gfit)
            gfit_sum_N2_imsub = np.nan
            SD_N2_gfit_bkg_corr.append(gfit_sum_N2_imsub)
# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers


# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
He_PN = []
He_PN_imsub = []
SD_He = []
SD_He_imsub = []
arr_ndarray_He = []
bound_transect_ndarray_He = []
bound_transect_ndarray_He_imsub = []
background_He = []
SD_He_imsub_corrected = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_He[arr, element])
    if np.amax(bound_transect) < 4095:
        arr_ndarray_He.append(arr)
        He_PN.append(element)
        bound_transect_ndarray_He.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_He.append(transect_summed)
        He_PN_imsub.append(element)
        bound_transect_imsub = np.array(Corrected_He[arr, element]).astype('int') - int(round(np.mean(np.array(Corrected_He[arr, element])[0:10])))
        bound_transect_imsub_z = bound_transect_imsub[bound_transect_imsub < 0] = 0
        bound_transect_ndarray_He_imsub.append(bound_transect_imsub_z)
        transect_summed_imsub = np.sum(bound_transect_imsub_z)
        SD_He_imsub.append(transect_summed_imsub)


# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
BKG_PN = []
BKG_PN_imsub = []
SD_BKG = []
arr_ndarray_BKG = []
bound_transect_ndarray_BKG = []
bound_transect_ndarray_BKG_imsub = []
background_BKG = []
SD_BKG_imsub = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_BKG[arr, element])
    if np.amax(bound_transect) < 4095:
        arr_ndarray_BKG.append(arr)
        BKG_PN.append(element)
        bound_transect_ndarray_BKG.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_BKG.append(transect_summed)
        BKG_PN_imsub.append(element)
        bound_transect_imsub = np.array(Raw_BKG[arr, element]).astype('int') - int(round(np.mean(np.array(Raw_BKG[arr, element]))))
        bound_transect_ndarray_BKG_imsub.append(bound_transect_imsub)
        transect_summed_imsub = np.sum(bound_transect_imsub)
        SD_BKG_imsub.append(transect_summed_imsub)
# this is important for evaluating profiles along transects between the bounds
# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers

Samp_PN = []
Samp_PN_imsub = []
SD_Samp = []
SD_Samp_imsub = []
arr_ndarray_Samp = []
bound_transect_ndarray_Samp = []
bound_transect_ndarray_Samp_imsub = []
bound_transect_ndarray_gfit_Samp = []
bound_transect_ndarray_gfit_Samp_imsub = []
bound_transect_aoc_array_Samp = []
bound_transect_aoc_array_Samp_imsub = []
background_Samp = []
SD_Samp_gfit = []
SD_Samp_gfit_bkg_corr = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_Sample[arr, element]).astype('int')
    if np.amax(bound_transect) < 4095:
        idx_max = np.argmax(bound_transect)
        Samp_PN.append(element)
        Samp_PN_imsub.append(element)
        # raw data wrangling
        arr_ndarray_Samp.append(arr)
        bound_transect_ndarray_Samp.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_Samp.append(transect_summed)
        # raw data wrangling with background subtraction
        bound_transect_imsub = np.array(Corrected_Sample[arr, element]).astype('int') - int(round(np.mean(np.array(Corrected_Sample[arr, element])[-5:])))
        bound_transect_ndarray_Samp_imsub.append(bound_transect_imsub)
        transect_summed_imsub = np.sum(bound_transect_imsub)
        SD_Samp_imsub.append(transect_summed_imsub)
        # gaussian fitting of raw data
        try:
            popt, pcov = curve_fit(gaussian, arr, bound_transect_imsub, p0=[bound_transect_imsub[idx_max], arr[idx_max], 5.0, 5.0])
            gfit = [gaussian(x, *popt) for x in arr]
            #print(popt)
            bound_transect_ndarray_gfit_Samp.append(gfit)
            gfit_sum_Samp = np.sum(gfit)
            SD_Samp_gfit.append(gfit_sum_Samp)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_Samp_imsub.append(gfit - popt[3])
            gfit_sum_Samp_imsub = np.sum(gfit - popt[3])
            SD_Samp_gfit_bkg_corr.append(gfit_sum_Samp_imsub)
        except RuntimeError:
            gfit = np.empty(len(arr))
            gfit[:] = np.nan
            bound_transect_ndarray_gfit_Samp.append(gfit)
            gfit_sum_Samp = np.nan
            SD_Samp_gfit.append(gfit_sum_Samp)
            # gaussian fitting of raw data with background correction
            bound_transect_ndarray_gfit_Samp_imsub.append(gfit)
            gfit_sum_Samp_imsub = np.nan
            SD_Samp_gfit_bkg_corr.append(gfit_sum_Samp_imsub)

'''
Samp_PN = []
Samp_PN_imsub = []
SD_Samp = []
SD_Samp_imsub = []
arr_ndarray_Samp = []
bound_transect_ndarray_Samp = []
bound_transect_ndarray_Samp_imsub = []
background_Samp = []
SD_Samp_imsub_corrected = []
for counter, element in enumerate(cols_array):
    arr = np.arange(top[counter], bot[counter], 1).astype(int)
    bound_transect = np.array(Raw_Sample[arr, element])
    if np.amax(bound_transect) < 4095:
        arr_ndarray_Samp.append(arr)
        Samp_PN.append(element)
        bound_transect_ndarray_Samp.append(bound_transect)
        transect_summed = np.sum(bound_transect)
        SD_Samp.append(transect_summed)
        bound_transect_imsub = np.array(Corrected_Sample[arr, element]).astype('int') - int(round(np.mean(np.array(Corrected_Sample[arr, element])[0:10])))
        Samp_PN_imsub.append(element)
        bound_transect_ndarray_Samp_imsub.append(bound_transect_imsub)
        transect_summed_imsub = np.sum(bound_transect_imsub)
        SD_Samp_imsub.append(transect_summed_imsub)
'''


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
im_f4 = ax4[0, 0].pcolormesh(Raw_Sample, cmap='gray')
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
ax4[1, 1].plot(Samp_PN_imsub, SD_Samp_imsub, linestyle='-', color='blue', label='SD: Sample - N2')
#ax4[1, 1].plot(cols_array, SD_Samp_imsub_corrected, linestyle='-', color='green', label='SD: Sample - N2 - Edge Corrected')
ax4[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax4[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax4[1, 1].set_title('Scattering Diagram')
ax4[1, 1].grid(True)
ax4[1, 1].set_yscale('log')
ax4[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F4_Sample.png', format='png')
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
ax5[0, 1].pcolormesh(Corrected_N2, cmap='gray')
ax5[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax5[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax5[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax5[0, 1].set_xlabel('Columns')
ax5[0, 1].set_ylabel('Rows')
ax5[0, 1].set_title('Averaged Nitrogen Image\n Helium Subtracted')
for counter, element in enumerate(arr_ndarray_N2):
    ax5[1, 0].plot(element, bound_transect_ndarray_N2[counter], linestyle='-')
ax5[1, 0].set_xlabel('Rows')
ax5[1, 0].set_ylabel('Intensity (DN)')
ax5[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax5[1, 0].grid(True)
ax5[1, 1].plot(N2_PN, SD_N2, linestyle='-', color='red', label='SD: N2')
ax5[1, 1].plot(N2_PN_imsub, SD_N2_imsub, linestyle='-', color='blue', label='SD: N2 - He')
#ax5[1, 1].plot(cols_array, SD_N2_imsub_corrected, linestyle='-', color='green', label='SD: N2 - He - Edge Correction')
ax5[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax5[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax5[1, 1].set_title('Scattering Diagram')
ax5[1, 1].grid(True)
ax5[1, 1].set_yscale('log')
ax5[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F5_N2.png', format='png')
plt.show()

# columns to theta
slope = 0.2112
intercept = -47.972
# columns to theta
theta_N2 = (np.array(N2_PN) * slope) + intercept
print('N2 angular range:', [theta_N2[0], theta_N2[-1]])
rads_N2 = theta_N2 * pi/180.0
rads_N2 = rads_N2
theta_CO2 = (np.array(CO2_PN) * slope) + intercept
print('CO2 angular range:', [theta_CO2[0], theta_CO2[-1]])
rads_CO2 = theta_CO2 * pi/180.0
rads_CO2 = rads_CO2
print('ROI range', [(slope * cols[0]) + intercept, (slope * cols[1]) + intercept])

# Save Phase Function, the data saved here has no subtractions/corrections applied to them, each is raw signal
# note the CCD Noise cannot be backed out, as we would have to cover the lens to do it, if at some point we take
# covered images we could do it...
DF_Headers = ['Sample Columns', 'CO2 Columns', 'N2 Columns', 'He Columns', 'BKG Columns', 'Theta', 'Sample Intensity', 'Sample Intensity Corrected', 'CO2 Intensity', 'CO2 Intensity Corrected', 'N2 Intensity', 'N2 Intensity Corrected', 'He Intensity', 'BKG Intensity']
DF_S_C = pd.DataFrame(Samp_PN)
DF_CO2_C = pd.DataFrame(CO2_PN)
DF_N2_C = pd.DataFrame(N2_PN)
DF_He_C = pd.DataFrame(He_PN)
DF_BKG_C = pd.DataFrame(BKG_PN)
DF_Theta = pd.DataFrame(theta_CO2)
DF_SD_S = pd.DataFrame(SD_Samp)
DF_SD_S_C = pd.DataFrame(SD_Samp_imsub)
DF_SD_CO2 = pd.DataFrame(SD_CO2)
DF_SD_CO2_C = pd.DataFrame(SD_CO2_imsub)
DF_SD_N2 = pd.DataFrame(SD_N2)
DF_SD_N2_C = pd.DataFrame(SD_N2_imsub)
DF_SD_He = pd.DataFrame(SD_He)
DF_SD_BKG = pd.DataFrame(SD_BKG)
PhaseFunctionDF = pd.concat([DF_S_C, DF_CO2_C, DF_N2_C, DF_He_C, DF_BKG_C, DF_Theta, DF_SD_S, DF_SD_S_C, DF_SD_CO2, DF_SD_CO2_C, DF_SD_N2, DF_SD_N2_C, DF_SD_He, DF_SD_BKG], ignore_index=False, axis=1)
PhaseFunctionDF.columns = DF_Headers
PhaseFunctionDF.to_csv(Path_Save + '/SD_Particle.txt')



f6, ax6 = plt.subplots(2, 2, figsize=(12, 7))
ax6[0, 0].plot(arr_ndarray_Samp[50], bound_transect_ndarray_Samp[50], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[50][:10])*len(bound_transect_ndarray_Samp[50])))
ax6[0, 0].plot(arr_ndarray_Samp[50], bound_transect_ndarray_Samp_imsub[50], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp_imsub[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp_imsub[50][:10])*len(bound_transect_ndarray_Samp_imsub[50])))
ax6[0, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 0].set_title('Profiles Compared')
ax6[0, 0].grid(True)
ax6[0, 0].legend(loc=1)
ax6[0, 1].plot(arr_ndarray_Samp[200], bound_transect_ndarray_Samp[200], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[200][:10])*len(bound_transect_ndarray_Samp[200])))
ax6[0, 1].plot(arr_ndarray_Samp[200], bound_transect_ndarray_Samp_imsub[200], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp_imsub[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp_imsub[200][:10])*len(bound_transect_ndarray_Samp_imsub[200])))
ax6[0, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 1].set_title('Profiles Compared')
ax6[0, 1].grid(True)
ax6[0, 1].legend(loc=1)
ax6[1, 0].plot(arr_ndarray_Samp[300], bound_transect_ndarray_Samp[300], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp[300][:10])*len(bound_transect_ndarray_Samp[300])))
ax6[1, 0].plot(arr_ndarray_Samp[300], bound_transect_ndarray_Samp_imsub[300], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_Samp_imsub[300])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_Samp_imsub[300][:10])*len(bound_transect_ndarray_Samp_imsub[300])))
ax6[1, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 0].set_title('Profiles Compared')
ax6[1, 0].grid(True)
ax6[1, 0].legend(loc=1)
ax6[1, 1].plot(arr_ndarray_N2[350], bound_transect_ndarray_N2[350], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_N2[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_N2[350][:10])*len(bound_transect_ndarray_N2[350])))
ax6[1, 1].plot(arr_ndarray_N2[350], bound_transect_ndarray_N2_imsub[350], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_N2_imsub[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_N2_imsub[350][:10])*len(bound_transect_ndarray_N2_imsub[350])))
ax6[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 1].set_title('Profiles Compared')
ax6[1, 1].grid(True)
ax6[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F6_Profiles.png', format='png')
plt.show()


f7, ax7 = plt.subplots(1, 2, figsize=(12, 7))
ax7[0].plot(Samp_PN, SD_Samp, linestyle='-', color='black', label='Raw 900nm PSL Scattering')
ax7[0].plot(CO2_PN, SD_CO2, linestyle='-', color='red', label='CO2 Scattering')
ax7[0].plot(N2_PN, SD_N2, linestyle='-', color='blue', label='N2 Scattering')
ax7[0].plot(He_PN, SD_He, linestyle='-', color='cyan', label='He Scattering')
ax7[0].plot(BKG_PN, SD_BKG, linestyle='-', color='green', label='Background Scattering')
ax7[0].set_title('Scattering Contributions to Raw Sample Scattering Diagram')
ax7[0].set_ylabel('Intensity (DN)')
ax7[0].set_xlabel('Profile Number (Column Number)')
ax7[0].grid(True)
ax7[0].legend(loc=1)
ax7[1].semilogy(Samp_PN, SD_Samp, linestyle='-', color='black', label='Raw 900nm PSL Scattering')
ax7[1].semilogy(CO2_PN, SD_CO2, linestyle='-', color='red', label='CO2 Scattering')
ax7[1].semilogy(N2_PN, SD_N2, linestyle='-', color='blue', label='N2 Scattering')
ax7[1].semilogy(He_PN, SD_He, linestyle='-', color='cyan', label='He Scattering')
ax7[1].semilogy(BKG_PN, SD_BKG, linestyle='-', color='green', label='Background Scattering')
ax7[1].set_title('Scattering Contributions to Raw Sample Scattering Diagram')
ax7[1].set_ylabel('Intensity (DN)')
ax7[1].set_xlabel('Profile Number (Column Number)')
ax7[1].grid(True)
ax7[1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F7_Contributions.png', format='png')
plt.show()

# we did not do a semilogy plot here, y data cannot have zeros in it! log of zero is not defined!!!
f8, ax8 = plt.subplots(figsize=(20, 7))
ax8.plot(Samp_PN_imsub, SD_Samp_imsub, linestyle='-', color='black', label='PSL Scattering - N2 Scattering')
ax8.plot(CO2_PN_imsub, SD_CO2_imsub, linestyle='-', color='red', label='$N_2$ Scattering - He Scattering')
ax8.plot(N2_PN_imsub, SD_N2_imsub, linestyle='-', color='blue', label='$N_2$ Scattering - He Scattering')
ax8.plot(He_PN_imsub, SD_He_imsub, linestyle='-', color='cyan', label='He Scattering - Background')
ax8.plot(BKG_PN_imsub, SD_BKG_imsub, linestyle='-', color='green', label='Background')
ax8.set_title('Corrected Scattering Diagram')
ax8.set_ylabel('Intensity (DN)')
ax8.set_xlabel('Profile Number (Column Number)')
ax8.grid(True)
ax8.legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F8_Contributions_Corr.png', format='png')
plt.show()
