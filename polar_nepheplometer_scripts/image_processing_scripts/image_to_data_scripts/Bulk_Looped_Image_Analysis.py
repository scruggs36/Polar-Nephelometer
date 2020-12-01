'''
Austen K. Scruggs
08-26-2020
Desctription: This code averages and 12bit 2darrays (containing 12bit image data) , then subtracts the averaged 2darrays
from their corresponding background 2darrays. This is the update that was necessary to apply from the Mono12 update. This is for PSL samples
not an analysis of Rayleigh scattering images. This is an improvement on the looped analysis particles 6, it will find the proper background
to subtract from the image data analyzed between the boundaries I have set (30 to 859), this eliminates hours of manual background subtraction
so that I can focus more on analyzing and presenting my newest findings! Any new exposure time, laser power, and polarization combinations unfortunately
have to be manually subtracted and added to the library
'''

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from PIL import Image
from scipy.interpolate import pchip_interpolate
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import shutil
import cv2
import os
import math


def gaussian(x, a, b, c, d):
    return d + (abs(a) * np.exp((-1 * (x - b) ** 2) / (2 * c ** 2)))


def gaussian_integral(x, a, b, c, d):
    total_signal = np.sum(np.array([gaussian(i, a, b, c, d) for i in x]))
    corrected_signal = np.sum(np.array([gaussian(i, a, b, c, d) - d for i in x]))
    background = d * (x[-1] - x[0])
    return [total_signal, corrected_signal, background]


# Create DataFrame/Library of Backgrounds, this will eventually just be commented out because we will have created our
# dataframe of N2 measurements to subtract from any data we take under the same conditions

# CANNOT PCHIP N2 BACKGROUNDS
def N2_Library_From_Measurements(n2_directory_unevaluated, n2_save_directory, n2_directory_evaluated):
    n2_riemann_2dlist = []
    n2_gfit_2dlist = []
    file_list = os.listdir(n2_directory_unevaluated)
    for counter, file in enumerate(file_list):
        print(file)
        # break down the file name into conditions for the n2 library
        conditions = file.split('_')
        slope = .2095
        intercept = -3.1433
        sample_str = 'N2'
        size_str = 'Molecule'
        exposure_str = conditions[2]
        polarization_str = conditions[3]
        power_str = conditions[4]
        averages_str = conditions[5]
        # use spaces to collect the date and time all in once, then use datetime package to format it in python right
        date_str = conditions[6]
        time_str = conditions[7]
        conc_str = 'None'
        # datetime wrangling...
        date_list = date_str.split(' ')
        meas_date = datetime.date(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2].split('.')[0]))
        time_list = time_str.split(' ')
        meas_time = datetime.time(hour=int(time_list[0]), minute=int(time_list[1]), second=int(time_list[2].split('.')[0]))
        header = ['Sample', 'Size (nm)', 'Exposure Time (s)', 'Polarization', 'Laser Power (mW)', 'Number of Averages', 'Concentration (p/cc)', 'Date', 'Time', 'Calibration Slope', 'Calibration Intercept']
        conditions_df = pd.DataFrame([[sample_str, size_str, exposure_str, polarization_str, power_str, averages_str, conc_str, meas_date, meas_time, slope, intercept]], columns=header)
        #print(conditions_df)
        # read in measurement
        measurement = pd.read_csv(n2_directory_unevaluated + '/' + file, sep=',', header=0)
        n2_columns = np.array(measurement['N2 Columns'])
        n2_riemann = np.array(measurement['N2 Intensity'])
        n2_gfit = np.array(measurement['N2 Intensity gfit'])

        # pchip meas, can't use pchip! It prdouces negative values due to extremely rapid change (noise)
        #n2_riemann_pchip = pchip_interpolate(xi=n2_theta, yi=n2_riemann, x=theta_pchip)
        #n2_gfit_pchip = pchip_interpolate(xi=n2_theta, yi=n2_gfit, x=theta_pchip)

        # make pchips dataframes
        n2_riemann_df = pd.DataFrame([n2_riemann], columns=n2_columns)
        n2_gfit_df = pd.DataFrame([n2_gfit], columns=n2_columns)

        # concatenate dataframes
        n2_df_riemann_row = pd.concat([conditions_df, n2_riemann_df], axis=1)
        n2_df_gfit_row = pd.concat([conditions_df, n2_gfit_df], axis=1)


        # run the if statements if your creating the file, for the first time
        # if you have all the files you wanna look at, this compiles them all at once
        if counter == 0:
            n2_df_riemann_row.to_csv(n2_save_directory + '/n2_riemann_library.txt', sep=',', header=True, index=False)
            n2_df_gfit_row.to_csv(n2_save_directory + '/n2_gfit_library.txt', sep=',', header=True, index=False)
            # moves a file that has been evaluated into the sorted
            shutil.move(n2_directory_unevaluated + '/' + file, n2_directory_evaluated + '/' + file)


        if counter > 0:
            n2_df_riemann_row.to_csv(n2_save_directory + '/n2_riemann_library.txt', sep=',', mode='a', header=False, index=False)
            n2_df_gfit_row.to_csv(n2_save_directory + '/n2_gfit_library.txt', sep=',', mode='a', header=False, index=False)
            # moves a file that has been evaluated into the sorted
            shutil.move(n2_directory_unevaluated + '/' + file, n2_directory_evaluated + '/' + file)


# n2 images --> n2 pf backgrounds --> add to n2 background df
def Add_2_N2_Library(cal_im_directory, add_n2_directory, n2_save_directory, n2_directory_evaluated, create_file):
    Calibration_Image = np.loadtxt(cal_im_directory, delimiter='\t')
    add_files_list = os.listdir(add_n2_directory)
    for it_number, file in enumerate(add_files_list):
        print(file)
        # break down the file name into conditions for the n2 library
        conditions = file.split('_')
        slope = 'NA'
        intercept = 'NA'
        sample_str = conditions[0]
        size_str = conditions[1]
        exposure_str = conditions[2]
        polarization_str = conditions[3]
        power_str = conditions[4]
        averages_str = conditions[5]
        # use spaces to collect the date and time all in once, then use datetime package to format it in python right
        date_str = conditions[6]
        time_str = conditions[7]
        # add _2darray_ to all file names at the end so the concentration is read properly
        conc_str = conditions[8]
        #conc_str_tosplit = conditions[8]
        #conc_str = conc_str_tosplit.split('.')[0]
        # datetime wrangling...
        date_list = date_str.split(' ')
        meas_date = datetime.date(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2].split('.')[0]))
        time_list = time_str.split(' ')
        meas_time = datetime.time(hour=int(time_list[0]), minute=int(time_list[1]), second=int(time_list[2].split('.')[0]))
        header = ['Sample', 'Size (nm)', 'Exposure Time (s)', 'Polarization', 'Laser Power (mW)', 'Number of Averages', 'Concentration (p/cc)', 'Date', 'Time', 'Calibration Slope', 'Calibration Intercept']
        conditions_df = pd.DataFrame([[sample_str, size_str, exposure_str, polarization_str, power_str, averages_str, conc_str, meas_date, meas_time, slope, intercept]], columns=header)
        #print(conditions_df)
        # read in the measurement
        Raw_N2 = np.loadtxt(add_n2_directory + '/' + file, delimiter='\t').astype(np.int64)
        # Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
        rows = [200, 300]
        cols = [30, 860]
        cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
        # ROI = im[rows[0]:rows[1], cols[0]:cols[1]]

        # find coordinates based on sample - N2 scattering averaged image (without corrections)
        row_max_index_array = []
        for element in cols_array:
            arr = np.arange(rows[0], rows[1], 1).astype(int)
            im_transect = Calibration_Image[arr, element]
            index_nosub = np.argmax(im_transect)
            row_max_index_array.append(index_nosub + rows[0])

        # polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
        tuner = len(cols_array)
        iterator = round(len(cols_array) / tuner)
        # based on the division in iterator, sometimes it needs an extra iteration to capture the rest of the coord points
        # iterator = round(len(cols_array)/tuner) + 1
        #print(iterator)
        mid = []
        top = []
        bot = []
        #sigma_pixels = 15
        sigma_pixels = 25
        degree = 2
        for counter, element in enumerate(range(iterator)):
            if counter < iterator:
                #print(counter)
                x = cols_array[(counter) * tuner: (counter + 1) * tuner]
                y = row_max_index_array[(counter) * tuner: (counter + 1) * tuner]
                #print(x)
                # print(y)
                polynomial_fit = np.poly1d(np.polyfit(x, y, deg=degree))
                # sigma_pixels = 20
                [mid.append(polynomial_fit(element)) for element in x]
                [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
                [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]
            if counter == iterator:
                #print(counter)
                x = cols_array[(counter) * tuner: len(cols_array)]
                y = row_max_index_array[(counter) * tuner: len(row_max_index_array)]
                #print(x)
                # print(y)
                polynomial_fit = np.poly1d(np.polyfit(x, y, deg=degree))
                # sigma_pixels = 20
                [mid.append(polynomial_fit(element)) for element in x]
                [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
                [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]

        '''
        coords_DF = pd.read_csv(coords_Dir + 'image_coordinates.txt', sep=',', header=0)
        top = coords_DF['Top']
        mid = coords_DF['Middle']
        bot = coords_DF['Bottom']
        '''

        # evaluate nitrogen background image
        N2_PN = []
        SD_N2 = []
        arr_ndarray_N2 = []
        bound_transect_ndarray_N2 = []
        bound_transect_ndarray_gfit_N2 = []
        bound_transect_ndarray_gfit_bc_N2 = []
        bkg_signal_ndarray = []
        bkg_signal = []
        SD_N2_gfit = []
        SD_N2_gfit_bkg_corr = []
        for counter, element in enumerate(cols_array):
            arr = np.arange(top[counter], bot[counter], 1).astype(int)
            bound_transect = np.array(Raw_N2[arr, element]).astype(int)
            #if np.amax(bound_transect) < 4095:
            if np.amax(bound_transect) > 1:
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
                    signal_params = gaussian_integral(x=arr, a=popt[0], b=popt[1], c=popt[2], d=popt[3])
                    gfit = np.array([gaussian(x, *popt) for x in arr])
                    # print(popt)
                    bound_transect_ndarray_gfit_N2.append(gfit)
                    #gfit_sum_N2 = np.sum(gfit)
                    gfit_sum_N2 = signal_params[0]
                    SD_N2_gfit.append(gfit_sum_N2)
                    # gaussian fitting of raw data with background correction
                    bound_transect_ndarray_gfit_bc_N2.append(gfit - popt[3])
                    #gfit_sum_N2_bc = np.sum(gfit - popt[3]*(arr[-1] - arr[0]))
                    gfit_sum_N2_bc = signal_params[1]
                    SD_N2_gfit_bkg_corr.append(gfit_sum_N2_bc)
                    bkg_signal_ndarray.append(np.repeat(popt[3],len(arr)))
                    bkg_signal_sum = signal_params[2]
                    bkg_signal.append(bkg_signal_sum)

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
                    # if it cannot distinguish signal from noise, everything is nan
                    bkg_signal_ndarray.append(np.repeat(popt[3], len(arr)))
                    bkg_signal_sum = np.nan
                    bkg_signal.append(bkg_signal_sum)
            # catching any other saturation as a Nan value, we've essentially turned this off, by bound_transect>1
            # if you turn it back on, the else statement needs to be udpated!!!
            else:
                idx_max = np.argmax(bound_transect)
                N2_PN.append(element)
                # data wrangling
                arr_ndarray_N2.append(arr)
                bound_transect_ndarray_N2.append(bound_transect)
                transect_summed = np.nan
                SD_N2.append(transect_summed)
                # gaussian fitting of raw data
                gfit_sum_N2 = np.nan
                SD_N2_gfit.append(gfit_sum_N2)
                # gaussian fitting of raw data with background correction
                gfit_sum_bc_N2 = np.nan
                SD_N2_gfit_bkg_corr.append(gfit_sum_bc_N2)


        #n2_riemann_pchip = pchip_interpolate(xi=np.array(N2_PN), yi=np.array(SD_N2), x=theta_pchip)
        #n2_gfit_pchip = pchip_interpolate(xi=np.array(N2_PN), yi=np.array(SD_N2_gfit), x=theta_pchip)
        meas_riemann_df = pd.DataFrame([np.array(SD_N2)], columns=np.array(N2_PN))
        meas_gfit_df = pd.DataFrame([np.array(SD_N2_gfit)], columns=np.array(N2_PN))
        meas_gfit_bc_df = pd.DataFrame([np.array(SD_N2_gfit_bkg_corr)], columns=np.array(N2_PN))
        meas_gfit_bkg_df = pd.DataFrame([np.array(bkg_signal)], columns=np.array(N2_PN))

        # create the dataframes side by side concatenation
        n2_riemann_df = pd.concat([conditions_df, meas_riemann_df], axis=1)
        n2_gfit_df = pd.concat([conditions_df, meas_gfit_df], axis=1)
        n2_gfit_bc_df = pd.concat([conditions_df, meas_gfit_bc_df], axis=1)
        n2_gfit_bkg_df = pd.concat([conditions_df, meas_gfit_bkg_df], axis=1)
        # save the dataframes
        # '''
        # run this loop if your creating the file
        if (it_number==0) and (create_file==True):
            # run this if its not the first time your creating a file, I am throwing everything into one csv
            n2_riemann_df.to_csv(n2_save_directory + '/n2_riemann_library.txt', sep=',', header=True, index=False)
            n2_gfit_df.to_csv(n2_save_directory + '/n2_gfit_library.txt', sep=',', header=True, index=False)
            n2_gfit_bc_df.to_csv(n2_save_directory + '/n2_gfit_bc_library.txt', sep=',', header=True, index=False)
            n2_gfit_bkg_df.to_csv(n2_save_directory + '/n2_gfit_bkg_library.txt', sep=',', header=True, index=False)
            # moves a file that has been evaluated into the sorted
            shutil.move(add_n2_directory + '/' + file, n2_directory_evaluated + '/' + file)
        else:
            # index as to be false or else it appends index column which is 0 for each row (ew)
            n2_riemann_df.to_csv(n2_save_directory + '/n2_riemann_library.txt', sep=',', mode='a', header=False, index=False)
            n2_gfit_df.to_csv(n2_save_directory + '/n2_gfit_library.txt', sep=',', mode='a', header=False, index=False)
            n2_gfit_bc_df.to_csv(n2_save_directory + '/n2_gfit_bc_library.txt', sep=',', mode='a', header=False, index=False)
            n2_gfit_bkg_df.to_csv(n2_save_directory + '/n2_gfit_bkg_library.txt', sep=',', mode='a', header=False, index=False)
            # moves a file that has been evaluated into the sorted
            shutil.move(add_n2_directory + '/' + file, n2_directory_evaluated + '/' + file)
        # '''


# Sample images directory --> Good data dataframes
def Add_Meas_2_MeasLibrary(sample_directory, sample_save_directory, evaluated_sample_directory, create_file):
    # import backgrounds and set index based upon parameters
    riemann_n2_directory = '/home/austen/Desktop/Recent/n2_riemann_library.txt'
    gfit_n2_directory = '/home/austen/Desktop/Recent/n2_gfit_library.txt'
    n2_library_riemann = pd.read_csv(riemann_n2_directory, sep=',', header=0)
    n2_library_riemann.set_index(['Exposure Time (s)', 'Polarization', 'Laser Power (mW)'], inplace=True)
    print('Background Library Shape: ', n2_library_riemann.shape)
    n2_library_gfit = pd.read_csv(gfit_n2_directory, sep=',', header=0)
    n2_library_gfit.set_index(['Exposure Time (s)', 'Polarization', 'Laser Power (mW)'], inplace=True)
    N2_PN = np.arange(30, 860, 1)

    # calibration?
    slope = 'NA'
    intercept = 'NA'

    # list of image files
    im_file_list = os.listdir(sample_directory)
    for it_number, file in enumerate(im_file_list):
        print(file)
        conditions = file.split('_')
        sample_str = conditions[0]
        size_str = conditions[1]
        exposure_str = conditions[2]
        polarization_str = conditions[3]
        power_str = conditions[4]
        averages_str = conditions[5]
        # use spaces to collect the date and time all in once, then use datetime package to format it in python right
        date_str = conditions[6]
        time_str = conditions[7]
        conc_str = conditions[8]
        # datetime wrangling...
        date_list = date_str.split(' ')
        meas_date = datetime.date(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2].split('.')[0]))
        time_list = time_str.split(' ')
        meas_time = datetime.time(hour=int(time_list[0]), minute=int(time_list[1]), second=int(time_list[2].split('.')[0]))
        header = ['Sample', 'Size (nm)', 'Exposure Time (s)', 'Polarization', 'Laser Power (mW)', 'Number of Averages', 'Concentration (p/cc)', 'Date', 'Time', 'Calibration Slope', 'Calibration Intercept']
        conditions_df = pd.DataFrame([[sample_str, size_str, exposure_str, polarization_str, power_str, averages_str, conc_str, meas_date, meas_time, slope, intercept]], columns=header)
        # print(conditions_df)

        # use the conditions to select a N2 background that is appropriate for the subtraction
        n2_library_riemann_subset = n2_library_riemann.xs((float(exposure_str), polarization_str, float(power_str)))#.reset_index()
        print('riemann length returned: ', len(n2_library_riemann_subset.shape))

        if len(n2_library_riemann_subset.shape) > 1:
            SD_N2 = np.array(n2_library_riemann_subset.loc[:, '30':'859'].mean(axis=0))



        else:
            SD_N2 = np.array(n2_library_riemann_subset.loc['30':'859'])


        #print(SD_N2)
        #print(SD_N2.shape)
        n2_library_gfit_subset = n2_library_gfit.xs((float(exposure_str), polarization_str, float(power_str)))#.reset_index()
        print('gfit length returned: ', len(n2_library_gfit_subset.shape))

        if len(n2_library_gfit_subset.shape) > 1:
            SD_N2_gfit = np.array(n2_library_gfit_subset.loc[:, '30':'859'].mean(axis=0))


        else:
            SD_N2_gfit = np.array(n2_library_gfit_subset.loc['30':'859'])

        # print(SD_N2_gfit)
        # print(SD_N2_gfit.shape)
        # actually importing in the sample
        Sample = np.loadtxt(sample_directory + '/' + file, delimiter='\t').astype(np.int64)
        # Initial boundaries on the image , cols can be: [250, 1040], [300, 1040], [405, 887]
        rows = [200, 300]
        cols = [30, 860]
        cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
        #ROI = im[rows[0]:rows[1], cols[0]:cols[1]]



        # find coordinates based on sample - N2 scattering averaged image (without corrections)
        row_max_index_array = []
        for element in cols_array:
            arr = np.arange(rows[0], rows[1], 1).astype(int)
            im_transect = Sample[arr, element]
            index_nosub = np.argmax(im_transect)
            row_max_index_array.append(index_nosub + rows[0])

        # polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
        tuner = len(cols_array)
        iterator = round(len(cols_array)/tuner)
        # based on the division in iterator, sometimes it needs an extra iteration to capture the rest of the coord points
        #iterator = round(len(cols_array)/tuner) + 1
        #print(iterator)
        mid = []
        top = []
        bot = []
        #sigma_pixels = 15
        sigma_pixels = 25
        degree = 2
        for counter, element in enumerate(range(iterator)):
            if counter < iterator:
                print(counter)
                x = cols_array[(counter) * tuner: (counter + 1) * tuner]
                y = row_max_index_array[(counter) * tuner: (counter + 1) * tuner]
                #print(x)
                #print(y)
                polynomial_fit = np.poly1d(np.polyfit(x, y, deg=degree))
                #sigma_pixels = 20
                [mid.append(polynomial_fit(element)) for element in x]
                [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
                [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]
            if counter == iterator:
                #print(counter)
                x = cols_array[(counter) * tuner: len(cols_array)]
                y = row_max_index_array[(counter) * tuner: len(row_max_index_array)]
                #print(x)
                # print(y)
                polynomial_fit = np.poly1d(np.polyfit(x, y, deg=degree))
                #sigma_pixels = 20
                [mid.append(polynomial_fit(element)) for element in x]
                [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
                [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]

        '''
        coords_DF = pd.read_csv(coords_Dir + 'image_coordinates.txt', sep=',', header=0)
        top = coords_DF['Top']
        mid = coords_DF['Middle']
        bot = coords_DF['Bottom']
        '''


        # this is important for evaluating profiles along transects between the bounds
        # loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
        Samp_PN = []
        SD_Samp = []
        arr_ndarray_Samp = []
        bound_transect_ndarray_Samp = []
        bound_transect_ndarray_gfit_Samp = []
        bound_transect_ndarray_gfit_bc_Samp = []
        bkg_signal_ndarray = []
        bkg_signal = []
        SD_Samp_gfit = []
        SD_Samp_gfit_bkg_corr = []
        for counter, element in enumerate(cols_array):
            arr = np.arange(top[counter], bot[counter], 1).astype(int)
            bound_transect = np.array(Sample[arr, element]).astype(int)
            #if np.amax(bound_transect) < 4095:
            if np.amax(bound_transect) > 1:
                #print(np.amax(bound_transect))
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
                    signal_params = gaussian_integral(x=arr, a=popt[0], b=popt[1], c=popt[2], d=popt[3])
                    gfit = np.array([gaussian(x, *popt) for x in arr])
                    #print(popt)
                    bound_transect_ndarray_gfit_Samp.append(gfit)
                    #gfit_sum_Samp = np.sum(gfit)
                    gfit_sum_Samp = signal_params[0]
                    SD_Samp_gfit.append(gfit_sum_Samp)
                    # gaussian fitting of raw data with background correction
                    bound_transect_ndarray_gfit_bc_Samp.append(gfit - popt[3])
                    #gfit_sum_bc_Samp = np.sum(gfit - popt[3])
                    gfit_sum_bc_Samp = signal_params[1]
                    SD_Samp_gfit_bkg_corr.append(gfit_sum_bc_Samp)
                    bkg_signal_ndarray.append(np.repeat(popt[3], len(arr)))
                    bkg_signal_sum = signal_params[2]
                    bkg_signal.append(bkg_signal_sum)
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
                    # if it cannot distinguish signal from noise, everything is nan
                    bkg_signal_ndarray.append(np.repeat(popt[3], len(arr)))
                    bkg_signal_sum = np.nan
                    bkg_signal.append(bkg_signal_sum)
            # catching any other saturation as a Nan value, we turned this off, we want it to show regardless 09/15/2020
            # if we turn saturation elimination on we would need to update the else condition!!!
            else:
                # this bit of code turns saturation into Nan value
                idx_max = np.argmax(bound_transect)
                Samp_PN.append(element)
                # data wrangling
                arr_ndarray_Samp.append(arr)
                bound_transect_ndarray_Samp.append(bound_transect)
                transect_summed = np.nan
                SD_Samp.append(transect_summed)
                # gaussian fitting of raw data
                gfit_sum_Samp = np.nan
                SD_Samp_gfit.append(gfit_sum_Samp)
                # gaussian fitting of raw data with background correction
                gfit_sum_bc_Samp = np.nan
                SD_Samp_gfit_bkg_corr.append(gfit_sum_bc_Samp)


        #samp_theta = np.array([slope * i + intercept for i in Samp_PN])
        #pf_riemann_pchip = pchip_interpolate(xi=samp_theta, yi=np.array(SD_Samp), x=theta_pchip)
        #pf_gfit_pchip = pchip_interpolate(xi=samp_theta, yi=np.array(SD_Samp_gfit), x=theta_pchip)

        pf_riemann_c = np.array(SD_Samp) - SD_N2
        # don't use SD_N2_gfit, transects do not have gaussian profile as beam often cannot be seen!
        pf_gfit_c = np.array(SD_Samp_gfit) - SD_N2_gfit

        # alright, this shit below needs to be udpated, its making a single file, lets do both, make the file and also just straight up add the files to the measurement library
        # append new measurements to n2 library! NEED TO WRITE CODE HERE!

        meas_riemann_df = pd.DataFrame([pf_riemann_c], columns=np.array(Samp_PN))
        meas_gfit_df = pd.DataFrame([pf_gfit_c], columns=np.array(Samp_PN))
        meas_gfit_bc_df = pd.DataFrame([np.array(SD_Samp_gfit_bkg_corr)], columns=np.array(Samp_PN))
        meas_gfit_bkg_df = pd.DataFrame([np.array(bkg_signal)], columns=np.array(Samp_PN))

        # create the dataframes side by side concatenation
        samp_pf_riemann_df = pd.concat([conditions_df, meas_riemann_df], axis=1)
        samp_pf_gfit_df = pd.concat([conditions_df, meas_gfit_df], axis=1)
        samp_pf_gfit_bc_df = pd.concat([conditions_df, meas_gfit_bc_df], axis=1)
        samp_pf_gfit_bkg_df = pd.concat([conditions_df, meas_gfit_bkg_df], axis=1)

        # save the dataframes
        # '''
        if (it_number==0) and (create_file==True):
            # run this if its not the first time your creating a file, I am throwing everything into one csv
            # index as to be false or else it appends index column which is 0 for each row (ew)
            samp_pf_riemann_df.to_csv(sample_save_directory + '/Good_Data_Riemann.txt', sep=',', header=True, index=False)
            samp_pf_gfit_df.to_csv(sample_save_directory + '/Good_Data_gfit.txt', sep=',', header=True, index=False)
            samp_pf_gfit_bc_df.to_csv(sample_save_directory + '/Good_Data_gfit_bc.txt', sep=',', header=True, index=False)
            samp_pf_gfit_bkg_df.to_csv(sample_save_directory + '/Good_Data_gfit_bkg.txt', sep=',', header=True, index=False)
            # moves a file that has been evaluated into the sorted
            shutil.move(sample_directory + '/' + file, evaluated_sample_directory + '/' + file)
        else:
            # run this if its not the first time your creating a file, I am throwing everything into one csv
            # index as to be false or else it appends index column which is 0 for each row (ew)
            samp_pf_riemann_df.to_csv(sample_save_directory + '/Good_Data_Riemann.txt', sep=',', mode='a', header=False, index=False)
            samp_pf_gfit_df.to_csv(sample_save_directory + '/Good_Data_gfit.txt', sep=',', mode='a', header=False, index=False)
            samp_pf_gfit_bc_df.to_csv(sample_save_directory + '/Good_Data_gfit_bc.txt', sep=',', mode='a', header=False, index=False)
            samp_pf_gfit_bkg_df.to_csv(sample_save_directory + '/Good_Data_gfit_bkg.txt', sep=',', mode='a', header=False, index=False)
            # moves a file that has been evaluated into the sorted
            shutil.move(sample_directory + '/' + file, evaluated_sample_directory + '/' + file)
            # '''

        # Save Phase Function, the data saved here has no subtractions/corrections applied to them, each is raw signal
        # note the CCD Noise cannot be backed out, as we would have to cover the lens to do it, if at some point we take
        # covered images we could do it...
        #DF_Headers = ['Sample Columns', 'N2 Columns', 'Sample Intensity', 'Sample Intensity gfit', 'N2 Intensity', 'N2 Intensity gfit']
        DF_Headers = ['Sample Columns', 'N2 Columns', 'Sample Intensity', 'Sample Intensity gfit', 'N2 Intensity']
        DF_S_C = pd.DataFrame(Samp_PN)
        DF_N2_C = pd.DataFrame(N2_PN)
        DF_PF_S = pd.DataFrame(pf_riemann_c)
        DF_PF_S_G = pd.DataFrame(pf_gfit_c)
        DF_PF_N2 = pd.DataFrame(SD_N2)
        #DF_PF_N2_G = pd.DataFrame(SD_N2_gfit)
        #PhaseFunctionDF = pd.concat([DF_S_C, DF_N2_C, DF_PF_S, DF_PF_S_G, DF_PF_N2, DF_PF_N2_G], ignore_index=False, axis=1)
        PhaseFunctionDF = pd.concat([DF_S_C, DF_N2_C, DF_PF_S, DF_PF_S_G, DF_PF_N2], ignore_index=False, axis=1)
        #PhaseFunctionDF.columns = DF_Headers
        PhaseFunctionDF.to_csv(sample_save_directory + '/' + sample_str + '_' + size_str + '_' + exposure_str + '_' + polarization_str + '_' + power_str + '_' + averages_str + '_' + date_str + '_' + time_str + '_' + conc_str + '.txt', sep=',', header=DF_Headers)
        # use spaces to collect the date and time all in once, then use datetime package to format it in python right


def Loop_Plot_Images(sample_directory, sample_save_directory, evaluated_sample_directory):
    # set up ROI on image
    rows = [200, 300]
    cols = [30, 860]
    cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)

    # loop read files
    file_list = os.listdir(sample_directory)
    for file in file_list:
        print(file)
        conditions = file.split('_')
        sample_str = conditions[0]
        size_str = conditions[1]
        exposure_str = conditions[2]
        polarization_str = conditions[3]
        power_str = conditions[4]
        averages_str = conditions[5]
        # use spaces to collect the date and time all in once, then use datetime package to format it in python right
        date_str = conditions[6]
        time_str = conditions[7]
        conc_str = conditions[8]
        # datetime wrangling...
        date_list = date_str.split(' ')
        meas_date = datetime.date(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2].split('.')[0]))
        time_list = time_str.split(' ')
        meas_time = datetime.time(hour=int(time_list[0]), minute=int(time_list[1]), second=int(time_list[2].split('.')[0]))

        # load the image
        image = np.loadtxt(sample_directory + '/' + file, delimiter='\t')

        # find the maximum of each transect and append in array
        row_max_index_array = []
        for element in cols_array:
            arr = np.arange(rows[0], rows[1], 1).astype(int)
            im_transect = image[arr, element]
            index_nosub = np.argmax(im_transect)
            row_max_index_array.append(index_nosub + rows[0])



        # polynomial fit to find the middle of the beam, the top bound, and bot bound, these give us our coordinates!
        polynomial_fit = np.poly1d(np.polyfit(cols_array, row_max_index_array, deg=2))
        sigma_pixels = 25
        mid = polynomial_fit(cols_array)
        top = polynomial_fit(cols_array) - sigma_pixels
        bot = polynomial_fit(cols_array) + sigma_pixels

        # grab the bound transects and shove them in a dataframe hopefully we never actually have to use them
        transect_list = []
        for counter, element in enumerate(cols_array):
            arr = np.arange(top[counter], bot[counter], 1).astype(int)
            im_transect_bound = image[arr, element]

            # creates transect dataframe for every image, literally we can replot everything if need be, hopefully this will just be extra stuff
            transect_list.append(im_transect_bound)
        # pandas does better when data is presented as a list of lists
        transect_df = pd.DataFrame(transect_list)
        transect_df_filename = sample_str + '_' + size_str + '_' + exposure_str + '_' + polarization_str + '_' + power_str + '_' + averages_str + '_' + date_str + '_' + time_str + '_' + conc_str + '_'
        transect_df.to_csv(sample_save_directory + '/' + transect_df_filename + 'transectdf.txt', sep=',')


        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
        im_fa = ax[0].pcolormesh(image, cmap='gray')
        im_fb = ax[1].pcolormesh(image, cmap='gray')
        ax[1].plot(cols_array, top, ls='-', color='lawngreen')
        ax[1].plot(cols_array, mid, ls='-', color='red')
        ax[1].plot(cols_array, row_max_index_array, ls='-', color='purple')
        ax[1].plot(cols_array, bot, ls='-', color='lawngreen')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider_a = make_axes_locatable(ax[0])
        divider_b = make_axes_locatable(ax[1])
        cax_a = divider_a.append_axes("right", size="5%", pad=0.05)
        cax_b = divider_b.append_axes("right", size="5%", pad=0.05)
        f.colorbar(im_fa, cax=cax_a)
        f.colorbar(im_fb, cax=cax_b)
        ax[0].set_title('Image')
        ax[1].set_title('Image & Boundaries')
        plt.savefig(sample_save_directory + '/' + transect_df_filename + 'image.png', format='png')
        plt.clf()
        shutil.move(sample_directory + '/' + file, evaluated_sample_directory + '/' + file)
# KEY POINT, IF CALIBRATION CHANGES NEW DATA HAS TO BE PUT IN NEW FILE!
'''
# just remember currently all steps should not be run simultaneously!
# Alright, step 1, create the df, this was successful! If you already have a file going, this is not needed
n2_dir_ev = '/home/austen/Desktop/Recent/Good Data/Evaluated'
n2_dir_unev = '/home/austen/Desktop/Recent/Good Data/Unevaluated'
n2_save_dir = '/home/austen/Desktop/Recent'
N2_Library_From_Measurements(n2_directory_unevaluated=n2_dir_unev, n2_save_directory=n2_save_dir, n2_directory_evaluated=n2_dir_ev)
n2_riemann_library = pd.read_csv(n2_save_dir + '/n2_riemann_library.txt', sep=',', header=0)
print(n2_riemann_library)
'''

# width of bounds changed from 15 to 25 9/15/2020, saturation to np.nan turned off
'''
# Alright, step 2, rename and add all the new N2 backgrounds to the N2 library, this was successful!
add_n2_dir = '/home/austen/Desktop/Recent/Background Processing/Unevaluated'
cal_im_dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-11-16/PSL/2darray/PSL_1000_18_SL_100_23_2020 11 16_22 19 01_260_2darray_.txt'
n2_save_dir = '/home/austen/Desktop/Recent'
eval_n2_dir = '/home/austen/Desktop/Recent/Background Processing/Evaluated'
Add_2_N2_Library(cal_im_directory=cal_im_dir, add_n2_directory=add_n2_dir, n2_save_directory=n2_save_dir, n2_directory_evaluated=eval_n2_dir, create_file=True)
n2_riemann_library = pd.read_csv(n2_save_dir + '/n2_riemann_library.txt', sep=',', header=0)
print(n2_riemann_library)
'''


#'''
# Alright, step 3, analyze all the new images
# Add meas to meas library
sample_dir = '/home/austen/Desktop/Recent/Good Data/Unevaluated'
sample_save_dir = '/home/austen/Desktop/Recent'
sample_evaluated_dir = '/home/austen/Desktop/Recent/Good Data/Evaluated/2darray'
Add_Meas_2_MeasLibrary(sample_directory=sample_dir, sample_save_directory=sample_save_dir, evaluated_sample_directory=sample_evaluated_dir, create_file=True)
meas_df = pd.read_csv('/home/austen/Desktop/Recent/Good_Data_Riemann.txt', sep=',', header=0)
print(meas_df)
#'''

'''
sample_dir = '/home/austen/Desktop/Recent/Good Data/Unevaluated'
sample_save_dir = '/home/austen/Desktop/Recent'
sample_evaluated_dir = '/home/austen/Desktop/Recent/Good Data/Evaluated/2darray'
Loop_Plot_Images(sample_directory=sample_dir, sample_save_directory=sample_save_dir, evaluated_sample_directory=sample_evaluated_dir)
'''

