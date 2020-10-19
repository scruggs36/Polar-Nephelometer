'''
Austen K. Scruggs
07-08-2020
Description: Compile all phase functions into a single DataFrame and file with extensively detailed notes
Sampling set up changed to mike set up August 6th 2020
'''

import numpy as np
import pandas as pd
import os
import shutil
import datetime
from scipy.interpolate import pchip_interpolate

# directories
save_file_riemann = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
save_file_gfit = '/home/austen/Desktop/Recent/Good_Data_gfit.txt'
# good data directory
unevaluated_directory = '/home/austen/Desktop/Recent/Good Data/Unevaluated'
evaluated_directory = '/home/austen/Desktop/Recent/Good Data/Evaluated'
file_list = os.listdir(unevaluated_directory)

for counter, file in enumerate(file_list):
    # conditions to transform columns to angles
    slope = .2095
    intercept = -3.1433
    # print file name
    print(file)
    # header created
    header = ['Sample', 'Size (nm)', 'Exposure Time (s)', 'Polarization', 'Laser Power (mW)', 'Number of Averages', 'Concentration (p/cc)', 'Date', 'Time', 'Calibration Slope', 'Calibration Intercept']
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
    # create df
    conditions_df = pd.DataFrame([[sample_str, size_str, exposure_str, polarization_str, power_str, averages_str, conc_str, meas_date, meas_time, slope, intercept]], columns=header)
    # read in the data


    if size_str == 'Molecule':
        measurement = pd.read_csv(unevaluated_directory + '/' + file, sep=',', header=0, engine='python')
        riemann = np.array(measurement['CO2 Intensity'])
        gfit = np.array(measurement['CO2 Intensity gfit'])
        columns = np.array(measurement['CO2 Columns'])
        # convert column number to angles through calibration
        #angles = [(slope * i) + intercept for i in columns]
        # set desired angles and pchip data, get values for these angles
        #angles_pchip = np.arange(0.0, 180.2, 0.2)
        #riemann_pchip = pchip_interpolate(xi=angles, yi=riemann, x=angles_pchip, der=0)
        #gfit_pchip = pchip_interpolate(xi=angles, yi=gfit, x=angles_pchip, der=0)
        meas_riemann_df = pd.DataFrame([riemann], columns=columns)
        meas_gfit_df = pd.DataFrame([gfit], columns=columns)
        # concatenate dataframes side by side
        row_riemann_df = pd.concat([conditions_df, meas_riemann_df], axis=1)
        row_gfit_df = pd.concat([conditions_df, meas_gfit_df], axis=1)


    else:
        measurement = pd.read_csv(unevaluated_directory + '/' + file, sep=',', header=0, engine='python')
        riemann = np.array(measurement['Sample Intensity'])
        gfit = np.array(measurement['Sample Intensity gfit'])
        columns = np.array(measurement['Sample Columns'])
        # convert column number to angles through calibration
        #angles = [(slope * i) + intercept for i in columns]
        # set desired angles and pchip data, get values for these angles
        #angles_pchip = np.arange(0.0, 180.2, 0.2)
        #riemann_pchip = pchip_interpolate(xi=angles, yi=riemann, x=angles_pchip, der=0)
        #gfit_pchip = pchip_interpolate(xi=angles, yi=gfit, x=angles_pchip, der=0)
        meas_riemann_df = pd.DataFrame([riemann], columns=columns)
        meas_gfit_df = pd.DataFrame([gfit], columns=columns)
        # concatenate dataframes side by side
        row_riemann_df = pd.concat([conditions_df, meas_riemann_df], axis=1)
        row_gfit_df = pd.concat([conditions_df, meas_gfit_df], axis=1)
    #'''
    #run this if its not the first time your creating a file, I am throwing everything into one csv
    row_riemann_df.to_csv(save_file_riemann, sep=',', mode='a', header=False, index=False)
    row_gfit_df.to_csv(save_file_gfit, sep=',', mode='a', header=False, index=False)
    # moves a file that has been evaluated into the sorted
    shutil.move(unevaluated_directory + '/' + file, evaluated_directory + '/' + file)
    #'''
    '''
    # run the if statements if your creating the file, for the first time
    # if you have all the files you wanna look at, this compiles them all at once
    if counter == 0:
        row_riemann_df.to_csv(save_file_riemann, sep=',', header=True, index=False)
        row_gfit_df.to_csv(save_file_gfit, sep=',', header=True, index=False)
        # moves a file that has been evaluated into the sorted
        shutil.move(unevaluated_directory + '/' + file, evaluated_directory + '/' + file)


    if counter > 0:
        row_riemann_df.to_csv(save_file_riemann, sep=',', mode='a', header=False, index=False)
        row_gfit_df.to_csv(save_file_gfit, sep=',', mode='a', header=False, index=False)
        # moves a file that has been evaluated into the sorted
        shutil.move(unevaluated_directory + '/' + file, evaluated_directory + '/' + file)
    '''

# read dataframe back in
print('-------Data compiling completed data frames are now ready-------')
data_read = pd.read_csv(save_file_riemann, sep=',', header=0)
print(data_read)
#print(data_read[['Sample', 'Date', 'Time', 'Size (nm)', 'Exposure Time (s)', 'Retardation (Degrees)', 'Integration']])