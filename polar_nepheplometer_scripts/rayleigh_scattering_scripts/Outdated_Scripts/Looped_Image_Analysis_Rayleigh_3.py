'''
Austen K. Scruggs
10-10-2018
Desctription: This code averages and 12bit 2darrays (containing 12bit image data) , then subtracts the averaged 2darrays
from their corresponding background 2darrays. This is the update that was necessary to apply from the Mono12 update.
'''

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelmin
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.misc import derivative
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from math import pi



# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_CO2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-08-2019/CO2/txt'
Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-08-2019/N2/txt'
Path_He_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-08-2019/He/txt'
Path_BKG_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/04-08-2019/BKG/txt'
#Path_Dark_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/01-23-2019/900/BKG/T1'
Path_Save = '/home/austen/Documents'

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
Raw_CO2 = Ndarray_Average(Path_CO2_Dir)
Raw_N2 = Ndarray_Average(Path_N2_Dir)
Raw_He = Ndarray_Average(Path_He_Dir)
Raw_BKG = Ndarray_Average(Path_BKG_Dir)


# sample - n2 - bkg, and n2 - he - bkg
Corrected_CO2 = np.subtract(Raw_CO2, Raw_He)
Corrected_CO2[Corrected_CO2 < 0] = 0
Corrected_N2 = np.subtract(Raw_N2, Raw_He)
Corrected_N2[Corrected_N2 < 0] = 0
Corrected_He = np.subtract(Raw_He, Raw_BKG)
Corrected_He[Corrected_He < 0] = 0

# Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
rows = [550, 650]
cols = [245, 1050]
cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
# these arrays are made trying to do a boostrap beam finding method
cols_mid_start = np.flip(np.arange(cols_array[0], round(len(cols_array)/2), 1).astype(int))
cols_mid_end = np.arange(round(len(cols_array)/2), cols_array[-1], 1).astype(int)
#ROI = im[rows[0]:rows[1], cols[0]:cols[1]]

# functions
def gaussian(x, a, b, c, d):
    return d + (abs(a) * np.exp((-1 * (x - b) ** 2) / (2 * c ** 2)))


def rayleigh_scattering(x, a, b):
    return a * b * (1 + np.square(np.cos(x)))
'''
# create a figure to test the process,
fd0, axd0 = plt.subplots(1, 2)
# find coordinates based on sample - N2 scattering averaged image (without corrections)
row_max_index_array_mid_start = []
for counter, element in enumerate(cols_mid_start):
    print(counter)
    if counter == 0:
        a = rows[0]
        b = rows[1]
        arr = np.array(np.arange(a, b, 1).astype(int))
        n = len(arr)
        im_transect = np.array(Corrected_CO2[arr, element])
        #print(arr)
        axd0[0].plot(arr, im_transect, ls='-')
        try:
            mean = sum(arr * im_transect) / n  # note this correction
            sig = sum(im_transect * (arr - mean) ** 2) / n
            popt_coords, pcov_coords = curve_fit(gaussian, arr, im_transect, p0=[1.0, mean, sig, 1.0])
            gauss = [gaussian(x, *popt_coords) for x in arr]
            axd0[1].plot(arr, gauss, ls='-')
            index = arr[np.argmax(gauss)]
            print('we tried: ', index)
            row_max_index_array_mid_start.append(index)
        except RuntimeError:
            index = arr[np.argmax(im_transect)]
            print('RuntimeError Exception: ', index)
            row_max_index_array_mid_start.append(index)
    if counter > 0:
        dpix = 10
        c = rows[0]
        d = rows[1]
        arr = np.array(np.arange(c, d, 1).astype(int))
        n = len(arr)
        #print(arr)
        im_transect = np.array(Corrected_CO2[arr, element])
        axd0[0].plot(arr, im_transect, ls='-')
        try:
            mean = sum(arr * im_transect) / n  # note this correction
            sig = sum(im_transect * (arr - mean) ** 2) / n
            popt_coords, pcov_coords = curve_fit(gaussian, arr, im_transect, p0=[1.0, mean, sig, 1.0])
            gauss = [gaussian(x, *popt_coords) for x in arr]
            axd0[1].plot(arr, gauss, ls='-')
            index = arr[np.argmax(gauss)]
            print('we also tried: ', index)
            row_max_index_array_mid_start.append(index)
        except RuntimeError:
            index = arr[np.argmax(im_transect)]
            print('RuntimeError Exception: ', index)
            row_max_index_array_mid_start.append(index)
row_max_index_array_start_mid = np.flip(row_max_index_array_mid_start)
plt.show()
row_max_index_array_mid_end = []
for counter, element in enumerate(cols_mid_end):
    if counter == 0:
        a = rows[0]
        b = rows[1]
        arr = np.array(np.arange(a, b, 1).astype(int))
        n = len(arr)
        im_transect = np.array(Corrected_CO2[arr, element])
        axd0[1].plot(arr, im_transect, ls='-')
        try:
            print('counter: ', counter)
            mean = sum(arr * im_transect) / n  # note this correction
            sig = sum(im_transect * (arr - mean) ** 2) / n
            popt_coords, pcov_coords = curve_fit(gaussian, arr, im_transect, p0=[1.0, mean, sig, 1.0])
            gauss = [gaussian(x, *popt_coords) for x in arr]
            axd0[1].plot(arr, gauss, ls='-')
            index = arr[np.argmax(gauss)]
            print(index)
            row_max_index_array_mid_end.append(index)
        except RuntimeError:
            index = arr[np.argmax(im_transect)]
            print('RuntimeError Exception: ', index)
            row_max_index_array_mid_end.append(index)
    if counter > 0:
        dpix = 10
        c = rows[0]
        d = rows[1]
        arr = np.array(np.arange(c, d, 1).astype(int))
        n = len(arr)
        im_transect = np.array(Corrected_CO2[arr, element])
        axd0[1].plot(arr, im_transect, ls='-')
        try:
            print('counter: ', counter)
            mean = sum(arr * im_transect) / n  # note this correction
            sig = sum(im_transect * (arr - mean) ** 2) / n
            popt_coords, pcov_coords = curve_fit(gaussian, arr, im_transect,  p0=[1.0, mean, sig, 1.0])
            gauss = [gaussian(x, *popt_coords) for x in arr]
            axd0[1].plot(arr, gauss, ls='-')
            index = arr[np.argmax(gauss)]
            print(index)
            row_max_index_array_mid_end.append(index)
        except RuntimeError:
            index = arr[np.argmax(im_transect)]
            print('RuntimeError Exception: ', index)
            row_max_index_array_mid_end.append(index)
rmia = np.append(np.concatenate((row_max_index_array_start_mid, row_max_index_array_mid_end)).flatten(), row_max_index_array_mid_end[-1])
row_max_index_array = rmia
#plt.show()
'''
row_max_index_array = []
for counter, element in enumerate(cols_array):
    a = rows[0]
    b = rows[1]
    arr = np.arange(a, b, 1).astype(int)
    im_transect = Corrected_CO2[arr, element]
    index_nosub = np.argmax(im_transect)
    row_max_index_array.append(index_nosub + rows[0])


'''
# polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
sigma_pixels = 20
mid = np.array(row_max_index_array)
top = np.array(row_max_index_array) + sigma_pixels
bot = np.array(row_max_index_array) - sigma_pixels
'''

polynomial_fit = np.poly1d(np.polyfit(cols_array, row_max_index_array, deg=2))
sigma_pixels = 20
mid = polynomial_fit(cols_array)
top = polynomial_fit(cols_array) - sigma_pixels
bot = polynomial_fit(cols_array) + sigma_pixels


# pretty picture plots for background signal corrections
# plots of all averaged images and profile coordinates
# plot of averaged background images
f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
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
f0.savefig(Path_Save + '/BKG.png', format='png')
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
f1.savefig(Path_Save + '/He.png', format='png')
plt.show()

# pretty picture plots for background signal corrections
# plot of the N2 corrected (helium and bkg) subtracted data
f2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
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
f2.savefig(Path_Save + '/N2.png', format='png')
plt.show()


f2x, ax2x = plt.subplots(1, 2, figsize=(12, 6))
im_f2a = ax2x[0].pcolormesh(Raw_CO2, cmap='gray')
im_f2b = ax2x[1].pcolormesh(Corrected_CO2, cmap='gray')
ax2x[1].plot(cols_array, top, ls='-', color='lawngreen')
ax2x[1].plot(cols_array, mid, ls='-', color='red')
ax2x[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
ax2x[1].plot(cols_array, bot, ls='-', color='lawngreen')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider_a = make_axes_locatable(ax2x[0])
divider_b = make_axes_locatable(ax2x[1])
cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
f2x.colorbar(im_f2a, cax=cax_a)
f2x.colorbar(im_f2a, cax=cax_b)
ax2x[0].set_title('$CO_2$ Image')
ax2x[1].set_title('Helium Corrected $CO_2$ Image')
f2x.savefig(Path_Save + '/CO2.png', format='png')
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
f3, ax3 = plt.subplots(2, 2, figsize=(12,6))
for counter, element in enumerate(bound_transect_ndarray_CO2_imsub):
    ax3[0, 0].plot(np.arange(0, len(element), 1), element, linestyle='-')
for counter, element in enumerate(bound_transect_ndarray_gfit_CO2_imsub):
    ax3[0, 1].plot(np.arange(0, len(element), 1), element, linestyle='-')
for counter, element in enumerate(bound_transect_ndarray_N2_imsub):
    ax3[1, 0].plot(np.arange(0, len(element), 1), element, linestyle='-')
for counter, element in enumerate(bound_transect_ndarray_gfit_N2_imsub):
    ax3[1, 1].plot(np.arange(0, len(element), 1), element, linestyle='-')
ax3[0, 0].set_xlabel('Profile')
ax3[0, 0].set_ylabel('Intensity (DN)')
ax3[0, 0].set_title('Summed Profiles $CO_2$')
ax3[0, 0].grid(True)
ax3[0, 1].set_xlabel('Profile')
ax3[0, 1].set_ylabel('Intensity (DN)')
ax3[0, 1].set_title('Gaussian Fit Profiles $CO_2$')
ax3[0, 1].grid(True)
ax3[1, 0].set_xlabel('Profile')
ax3[1, 0].set_ylabel('Intensity (DN)')
ax3[1, 0].set_title('Summed Profiles $N_2$')
ax3[1, 0].grid(True)
ax3[1, 1].set_xlabel('Profile')
ax3[1, 1].set_ylabel('Intensity (DN)')
ax3[1, 1].set_title('Gaussian Fit Profiles $N_2$')
ax3[1, 1].grid(True)
plt.tight_layout()
f3.savefig(Path_Save + '/F3_Profiles.png', format='png')
plt.show()

# savitzky-golay smooth gaussian data
SD_CO2_gfit_SG = savgol_filter(SD_CO2_gfit, window_length=151, polyorder=2, deriv=0)
SD_N2_gfit_SG = savgol_filter(SD_N2_gfit, window_length=151, polyorder=2, deriv=0)

# savitzky-golay phase functions from sum method
SD_CO2_imsub_SG = savgol_filter(SD_CO2_imsub, window_length=151, polyorder=2, deriv=0)
SD_N2_imsub_SG = savgol_filter(SD_N2_imsub, window_length=151, polyorder=2, deriv=0)
# plot of the Sample nitrogen subtracted data with bounds
f4, ax4 = plt.subplots(2, 2, figsize=(12, 6))
im_f4 = ax4[0, 0].pcolormesh(Raw_CO2, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f4.colorbar(im_f4, cax=cax)
ax4[0, 0].set_title('Averaged Sample Image')
ax4[0, 0].set_xlabel('Columns')
ax4[0, 0].set_ylabel('Rows')
ax4[0, 1].pcolormesh(Corrected_CO2, cmap='gray')
ax4[0, 1].plot(cols_array, top, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].plot(cols_array, mid, marker='.', ms=0.1, color='red')
ax4[0, 1].plot(cols_array, bot, marker='.', ms=0.1, color='lawngreen')
ax4[0, 1].set_xlabel('Columns')
ax4[0, 1].set_ylabel('Rows')
ax4[0, 1].set_title('Averaged Sample Image \n Nitrogen Subtracted')
for counter, element in enumerate(arr_ndarray_CO2):
    ax4[1, 0].plot(element, bound_transect_ndarray_CO2[counter], linestyle='-')
ax4[1, 0].set_xlabel('Rows')
ax4[1, 0].set_ylabel('Intensity (DN)')
ax4[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax4[1, 0].grid(True)
ax4[1, 1].plot(CO2_PN, SD_CO2, linestyle='-', color='red', label='SD: CO2 Raw')
ax4[1, 1].plot(CO2_PN_imsub, SD_CO2_imsub, linestyle='-', color='blue', label='SD: CO2 - He - BKG')
ax4[1, 1].plot(CO2_PN_imsub, SD_CO2_imsub_SG, linestyle='-', color='cyan', label='SD: CO2 - He - BKG SG')
ax4[1, 1].plot(CO2_PN_imsub, SD_CO2_gfit_SG, linestyle='-', color='green', label='SD: CO2 Raw Gaussian Fit SG')
ax4[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax4[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax4[1, 1].set_title('Scattering Diagram')
ax4[1, 1].grid(True)
ax4[1, 1].set_yscale('log')
ax4[1, 1].legend(loc=1)
plt.tight_layout()
f4.savefig(Path_Save + '/F4_CO2.png', format='png')
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
ax5[0, 0].set_xlabel('Columns')
ax5[0, 0].set_ylabel('Rows')
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
ax5[1, 1].plot(N2_PN, SD_N2, linestyle='-', color='red', label='SD: N2 Raw')
ax5[1, 1].plot(N2_PN_imsub, SD_N2_imsub, linestyle='-', color='blue', label='SD: N2 - He')
ax5[1, 1].plot(N2_PN_imsub, SD_N2_imsub_SG, linestyle='-', color='cyan', label='SD: N2 - He SG')
ax5[1, 1].plot(CO2_PN_imsub, SD_N2_gfit_SG, linestyle='-', color='green', label='SD: N2 Raw Gaussian Fit SG')
ax5[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax5[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax5[1, 1].set_title('Scattering Diagram')
ax5[1, 1].grid(True)
ax5[1, 1].set_yscale('log')
ax5[1, 1].legend(loc=1)
plt.tight_layout()
f5.savefig(Path_Save + '/F5_N2.png', format='png')
plt.show()

# columns to theta
slope = 0.2112
intercept = -47.972
# columns to theta
theta_N2 = (np.array(N2_PN) * slope) + intercept
print('N2 angular range:', [theta_N2[0], theta_N2[-1]])
rads_N2 = theta_N2 * pi/180.0
theta_CO2 = (np.array(CO2_PN) * slope) + intercept
print('CO2 angular range:', [theta_CO2[0], theta_CO2[-1]])
rads_CO2 = theta_CO2 * pi/180.0
print('ROI range', [(slope * cols[0]) + intercept, (slope * cols[1]) + intercept])


# Save Phase Function, the data saved here has no subtractions/corrections applied to them, each is raw signal
# note the CCD Noise cannot be backed out, as we would have to cover the lens to do it, if at some point we take
# covered images we could do it...
DF_Headers = ['CO2 Columns', 'N2 Columns', 'He Columns', 'BKG Columns', 'Theta', 'CO2 Intensity', 'N2 Intensity', 'He Intensity', 'BKG Intensity']
DF_CO2_C = pd.DataFrame(CO2_PN)
DF_N2_C = pd.DataFrame(N2_PN)
DF_He_C = pd.DataFrame(He_PN)
DF_BKG_C= pd.DataFrame(BKG_PN)
DF_Theta = pd.DataFrame(theta_CO2)
DF_SD_CO2 = pd.DataFrame(SD_CO2)
DF_SD_N2 = pd.DataFrame(SD_N2)
DF_SD_He = pd.DataFrame(SD_He)
DF_SD_BKG = pd.DataFrame(SD_BKG)
PhaseFunctionDF = pd.concat([DF_CO2_C, DF_N2_C, DF_He_C, DF_BKG_C, DF_Theta, DF_SD_CO2, DF_SD_N2, DF_SD_He, DF_SD_BKG], ignore_index=False, axis=1)
PhaseFunctionDF.columns = DF_Headers
PhaseFunctionDF.to_csv(Path_Save + '/SD_Rayleigh.txt')



f6, ax6 = plt.subplots(2, 2, figsize=(12, 6))
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
ax6[1, 1].plot(arr_ndarray_N2[350], bound_transect_ndarray_CO2[350], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2[350][:10])*len(bound_transect_ndarray_CO2[350])))
ax6[1, 1].plot(arr_ndarray_N2[350], bound_transect_ndarray_CO2_imsub[350], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_CO2_imsub[350])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_CO2_imsub[350][:10])*len(bound_transect_ndarray_CO2_imsub[350])))
ax6[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 1].set_title('Profiles Compared')
ax6[1, 1].grid(True)
ax6[1, 1].legend(loc=1)
plt.tight_layout()
f6.savefig(Path_Save + '/F6_Profiles.png', format='png')
plt.show()



# 1 + cos^2(theta) fits
popt_CO2, pcov_CO2 = curve_fit(rayleigh_scattering, theta_CO2, SD_CO2_imsub)
popt_N2, pcov_N2 = curve_fit(rayleigh_scattering, theta_N2, SD_N2_imsub)

# plug into function for each angle
rayleigh_cos_N2 = np.array([rayleigh_scattering(rad, *popt_N2) for rad in rads_N2])
rayleigh_cos_CO2 = np.array([rayleigh_scattering(rad, *popt_CO2) for rad in rads_CO2])
rayleigh_ideal_cos = np.array([1 + (np.cos(rad) ** 2) for rad in rads_CO2])


# take the ratio of the data to the fit
ral_CO2 = np.array([rayleigh_scattering(rad, *popt_CO2) for rad in rads_CO2])
ratio_CO2 = np.array(SD_CO2_imsub) / ral_CO2
ral_N2 = np.array([rayleigh_scattering(rad, *popt_N2) for rad in rads_N2])
ratio_N2 = np.array(SD_N2_imsub) / ral_N2
ratio_CO2_N2 = np.array(SD_CO2_gfit_SG) / np.array(SD_N2_gfit_SG)
ratio_CO2_N2_normed = (np.array(SD_CO2_gfit_SG)/np.linalg.norm(np.array(SD_CO2_gfit_SG))) / (np.array(SD_N2_gfit_SG)/np.linalg.norm(np.array(SD_N2_gfit_SG)))
#print(ratio_CO2_N2)
ratio_CO2_min = ratio_CO2 / np.amin(ratio_CO2)
ratio_N2_min = ratio_N2 / np.amin(ratio_N2)
ratio_ideal_CO2 = np.array(SD_CO2_imsub) / rayleigh_ideal_cos
ratio_ideal_N2 = np.array(SD_N2_imsub) / rayleigh_ideal_cos
ideal_CO2_N2_ratio = ral_CO2 / ral_N2


# filters and pchips
ratio_CO2_min_savgol = savgol_filter(ratio_CO2_min, window_length=151, polyorder=2, deriv=0)
ratio_N2_min_savgol = savgol_filter(ratio_N2_min, window_length=151, polyorder=2, deriv=0)
ratio_CO2_min_pchip = pchip_interpolate(theta_CO2, ratio_CO2_min_savgol, cols_array, der=0, axis=0)
ratio_N2_min_pchip = pchip_interpolate(theta_N2, ratio_N2_min_savgol, cols_array, der=0, axis=0)


f7, ax7 = plt.subplots(figsize=(12, 6))
ax7.plot(theta_CO2, SD_CO2, linestyle='-', color='black', label='CO2 Scattering')
ax7.plot(theta_N2, SD_N2, linestyle='-', color='green', label='N2 Scattering')
ax7.plot(theta_N2, SD_He, linestyle='-', color='cyan', label='He Scattering')
ax7.plot(theta_N2, SD_BKG, linestyle='-', color='red', label='Background Scattering')
ax7.set_title('Scattering Contributions to Raw Sample Scattering Diagram')
ax7.set_ylabel('Intensity (DN)')
ax7.set_xlabel('Profile Number (Column Number)')
ax7.grid(True)
ax7.legend(loc=1)
plt.tight_layout()
f7.savefig(Path_Save + '/F7_Contributions.png', format='png')
plt.show()

# we did not do a semilogy plot here, y data cannot have zeros in it! log of zero is not defined!!!
f8, ax8 = plt.subplots(figsize=(12, 6))
ax8.plot(theta_CO2, SD_CO2_imsub, linestyle='-', color='black', label='$CO_2$ Scattering - He Scattering')
ax8.plot(theta_CO2, SD_CO2_imsub_SG, linestyle='--', color='purple', label='$CO_2$ Scattering - He Scattering')
ax8.plot(theta_CO2, rayleigh_cos_CO2, linestyle='--', color='yellow', label='CO2 Scattering fit')
ax8.plot(theta_N2, SD_N2_imsub, linestyle='-', color='green', label='$N_2$ Scattering - He Scattering')
ax8.plot(theta_N2, SD_N2_imsub_SG, linestyle='--', color='lawngreen', label='$N_2$ Scattering - He Scattering')
ax8.plot(theta_N2, rayleigh_cos_N2, linestyle='--', color='orange', label='N2 Scattering fit')
ax8.plot(theta_N2, SD_He_imsub, linestyle='-', color='cyan', label='He Scattering - Background')
ax8.plot(theta_N2, SD_BKG_imsub, linestyle='-', color='red', label='Background')
ax8.set_title('Corrected Scattering Diagram')
ax8.set_ylabel('Intensity (DN)')
ax8.set_xlabel('Profile Number (Column Number)')
ax8.grid(True)
ax8.legend(loc=1)
plt.tight_layout()
f8.savefig(Path_Save + '/F8_Contributions_Corr.png', format='png')
plt.show()


f9, ax9 = plt.subplots(2, 2, figsize=(20, 6))
ax9[0, 0].plot(theta_CO2, rayleigh_ideal_cos, linestyle='-', color='red', label='$(1 + cos^2(\u0398)$')
ax9[0, 0].grid(True)
ax9[0, 0].set_xlabel('\u0398')
ax9[0, 0].set_ylabel('Ratio')
ax9[0, 0].set_title('1 + $cos^2(\u0398)$')
ax9[0, 0].legend(loc=1)
ax9[0, 1].plot(theta_CO2, ratio_CO2, linestyle='-', color='red', label='$CO_2$/$(ab(1 + cos^2(\u0398)$ \n a = ' + str(popt_CO2[0]) + ' b = ' + str(popt_CO2[1]))
ax9[0, 1].plot(theta_N2, ratio_N2, linestyle='-', color='blue', label='$N_2$/$(ab(1 + cos^2(\u0398)$ \n a = ' + str(popt_N2[0]) + ' b = ' + str(popt_N2[1]))
ax9[0, 1].plot(theta_CO2, ratio_CO2_N2, linestyle='-', color='green', label='$CO_2$/$N_2$')
ax9[0, 1].plot(theta_CO2, ratio_CO2_N2_normed, linestyle='-', color='purple', label='$CO_2$/$N_2$ normed')
ax9[0, 1].plot(theta_CO2, ideal_CO2_N2_ratio, linestyle='-', color='orange', label='$CO_2 fit$/$N_2 fit$')
ax9[0, 1].grid(True)
ax9[0, 1].set_xlabel('\u0398')
ax9[0, 1].set_ylabel('Ratio')
ax9[0, 1].set_title('Ratio Plot \n Rayleigh Scattering : ab(2 + $cos^2(\u0398)$')
ax9[0, 1].legend(loc=1)
ax9[1, 0].plot(theta_CO2, ratio_ideal_CO2, linestyle='-', color='red', label='$CO_2$/$(1 + cos^2(\u0398)$')
ax9[1, 0].plot(theta_N2, ratio_ideal_N2, linestyle='-', color='blue', label='$N_2$/$(1 + cos^2(\u0398)$')
ax9[1, 0].plot(theta_CO2, ratio_CO2_N2, linestyle='-', color='green', label='$CO_2$/$N_2$')
ax9[1, 0].plot(theta_CO2, ideal_CO2_N2_ratio, linestyle='-', color='orange', label='$CO_2 fit$/$N_2 fit$')
ax9[1, 0].grid(True)
ax9[1, 0].set_xlabel('\u0398')
ax9[1, 0].set_ylabel('Ratio')
ax9[1, 0].set_title('Ratio Plot \n Rayleigh Scattering : 1 + $cos^2(\u0398)$')
ax9[1, 0].legend(loc=1)
ax9[1, 1].plot(SD_CO2_imsub, rayleigh_ideal_cos, ls='-', color='red', label='$CO_2$ Correlation')
ax9[1, 1].plot(SD_N2_imsub, rayleigh_ideal_cos, ls='-', color='blue', label='$N_2$ Correlation')
ax9[1, 1].grid(True)
ax9[1, 1].set_xlabel('Intensity Experiment')
ax9[1, 1].set_ylabel('Intensity (1 + $cos^2(\u0398)$)')
ax9[1, 1].set_title('Correlation Plots to Rayleigh Theory: 1 + $cos^2(\u0398)$')
ax9[1, 1].legend(loc=1)
plt.tight_layout()
f9.savefig(Path_Save + '/F9_Ratio.png', format='png')
plt.show()

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

'''
chi_array = np.abs(np.nan_to_num(SD_CO2_imsub) - ral_CO2)
chi_array_SG = savgol_filter(chi_array, window_length=151, polyorder=2, deriv=0)

idx_relmin = argrelmin(chi_array_SG, order=50)

idx_0 = find_nearest_idx(chi_array_SG, 0)
idx_1 = find_nearest_idx(chi_array_SG[idx_0+10:-1], 0)
minima = [idx_0, idx_1]


f10, ax10 = plt.subplots(figsize=(12, 6))
ax10.plot(theta_CO2, chi_array, ls='-', color='blue', label='Chi Array')
ax10.plot(theta_CO2, chi_array_SG, ls='-', color='yellow', label='Chi Array')
for element in idx_relmin:
    print('angular range: ', theta_CO2[element])
    ax10.plot(theta_CO2[element], chi_array_SG[element], ls=' ', marker='x', color='red', label='Limits')
ax10.set_xlabel('\u0398')
ax10.set_ylabel('Chi Values')
ax10.set_title('Chi Test for Minima')
ax10.grid(True)
ax10.legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F10_Chi.png', format='png')
plt.show()
'''
'''
dy = []
for counter, element in enumerate(row_max_index_array):
    if counter == 0:
        del_y = row_max_index_array[counter + 1] - row_max_index_array[counter]
        dy.append(del_y)
        dy.append(del_y)
    if counter > 0:
        del_y = row_max_index_array[counter - 1] - row_max_index_array[counter]
        dy.append(del_y)
    if counter == len(row_max_index_array)-1:
        del_y = row_max_index_array[counter - 1] - row_max_index_array[counter]
        dy.append(del_y)


# derivative of the polynomial fit, this should give us nonlinear binning
dz_dy = [derivative(polynomial_fit, x, dx=1)**-1 for x in cols_array]
dz = []
for counter, element in dz_dy:
    dz.append(element * dy[counter])

nonlin_dz = []
for counter, element in dz:
    nonlin_dz.append(cols_array[counter] + element)


nonlin_d_angle_bins = [(x * slope) + intercept for x in nonlin_dz]
print(nonlin_d_angle_bins)


f11, ax11 = plt.subplots(figsize=(12, 6))
ax11.bar(theta_CO2, SD_CO2_imsub, width=nonlin_d_angle_bins, align='center', ls='-', color='red', label='CO2')
ax11.bar(theta_N2, SD_N2_imsub, width=nonlin_d_angle_bins, align='center', ls='-', color='blue', label='N2')
ax11.plot(theta_CO2, ral_CO2, color='black', ls='-', label='$CO_2$ a + a * $cos^2(\u0398)$')
ax11.plot(theta_N2, ral_N2, color='black', ls='--', label='$N_2$ a + a * $cos^2(\u0398)$')
ax11.set_title('Rayleigh Phase Functions for $N_2$ and $CO_2$ Gases')
ax11.set_ylabel('Intensity')
ax11.set_xlabel('\u0398')
ax11.grid(True)
ax11.legend(loc=1)
f11.savefig(Path_Save + '/Nonlinear_Binning.png', format='png')
plt.show()
'''
