'''
Austen K. Scruggs
06-24-2019
Description: This function evaluates images and produces phase functions, it is designed to be used inside the greater
labview code
'''


import numpy as np
from scipy.optimize import curve_fit

def gaussian(x, a, b, c, d):
    return d + (abs(a) * np.exp((-1 * (x - b) ** 2) / (2 * c ** 2)))
# Sun has changed this function slightly
def Image_Analysis(im, row_i, row_f, col_i, col_f, sp):
    rows = [row_i, row_f]
    cols = [col_i, col_f]
    cols_array = np.arange(cols[0], cols[1], 1).astype(int)
    row_max_index_array = []
    for element in cols_array:
        arr = np.arange(rows[0], rows[1], 1).astype(int)
        arr0 = arr[0]
        arrf = arr[-1]
        im_transect = im[arr0:arrf, element]
        max_index = np.argmax(im_transect)
        row_max_index_array.append(max_index + rows[0])
    mid = []
    top = []
    bot = []
    sigma_pixels = sp
    polynomial_fit = np.poly1d(np.polyfit(cols_array, row_max_index_array, deg=6))
    [mid.append(polynomial_fit(element)) for element in cols_array]
    [top.append(polynomial_fit(element) - sigma_pixels) for element in cols_array]
    [bot.append(polynomial_fit(element) + sigma_pixels) for element in cols_array]

    # loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
    PN = []
    SD = []
    arr_ndarray = []
    bound_transect_ndarray = []
    bound_transect_ndarray_gfit = []
    bound_transect_ndarray_gfit_bc = []
    SD_gfit = []
    SD_gfit_bkg_corr = []
    for counter, element in enumerate(cols_array):
        arr = np.arange(top[counter], bot[counter], 1).astype(int)
        bound_transect = np.array(im[arr, element]).astype('int')
        if np.amax(bound_transect) < 4095:
            idx_max = np.argmax(bound_transect)
            PN.append(element)
            # raw data wrangling
            arr_ndarray.append(arr)
            bound_transect_ndarray.append(bound_transect)
            transect_summed = np.sum(bound_transect)
            SD.append(transect_summed)
            # gaussian fitting of raw data
            try:
                popt, pcov = curve_fit(gaussian, arr, bound_transect, p0=[bound_transect[idx_max], arr[idx_max], 5.0, 5.0])
                gfit = [gaussian(x, *popt) for x in arr]
                # print(popt)
                bound_transect_ndarray_gfit.append(gfit)
                gfit_sum = np.sum(gfit)
                SD_gfit.append(gfit_sum)
                # gaussian fitting of raw data with background correction
                bound_transect_ndarray_gfit_bc.append(gfit - popt[3])
                gfit_sum_bc = np.sum(gfit - popt[3])
                SD_gfit_bkg_corr.append(gfit_sum_bc)
            except RuntimeError:
                gfit = np.empty(len(arr))
                gfit[:] = np.nan
                bound_transect_ndarray_gfit.append(gfit)
                gfit_sum = np.nan
                SD_gfit.append(gfit_sum)
                # gaussian fitting of raw data with background correction
                bound_transect_ndarray_gfit_bc.append(gfit)
                gfit_sum_bc = np.nan
                SD_gfit_bkg_corr.append(gfit_sum_bc)
    return(SD, SD_gfit, SD_gfit_bkg_corr)

