'''
Austen K. Scruggs
07-08-2020
Description: Compile all phase functions into a single DataFrame and file with extensively detailed nots
'''

import numpy as np
import pandas as pd
import datetime


# directories
save_file = '/home/austen/Desktop/Recent/All_Austens_Data.txt'
# add and save data to data frame
import_file = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-06-23/Analysis/1mW/SD_Particle_900nm_1mW.txt'
measurement_file = pd.read_csv(import_file, sep=',', header=0, engine='python')

# header created
header = ['Sample', 'Date', 'Time', 'Size (nm)', 'Exposure Time (s)', 'Laser Power (mW)', 'Retardation (Degrees)', 'Number of Averages', 'Integration', 'Calibration Slope', 'Calibration Intercept', 'Phase Function', 'Angles (Degrees)']

# append to data
sample = 'PSL'
# year, month, day --> integers
date = datetime.date(2020, 6, 23)
# hour, minute, second --> integers
time = datetime.time(12, 53, 11)
size = 'Rayleigh'
#size = 900
exposure_time = 100
laser_power = 100
retardation = 'Straight From Laser'
num_averages = 3
cal_theta_slope = .2095
cal_theta_int = -3.1433
#integration = 'Riemann'
integration = 'Gaussian'
#phase_function = np.array(measurement_file['CO2 Intensity'])
phase_function = np.array(measurement_file['CO2 Intensity gfit'])
columns = np.array(measurement_file['CO2 Columns'])
degrees = np.array([(cal_theta_slope * i) + cal_theta_int for i in columns])
df = pd.DataFrame([[sample, date, time, size, exposure_time, laser_power, retardation, num_averages, integration, cal_theta_slope, cal_theta_int, phase_function, degrees]], columns=header)
print(df)

# call and append to file on network (once per huge file containing types of data) mode a = append
# call the below the first time to make the file
#df.to_csv(save_file, mode='a', header=True)
# call the below for all other subsequent times
df.to_csv(save_file, mode='a', header=False)

# read dataframe back in
data_read = pd.read_csv(save_file, sep=',', header=0)
print(data_read[['Sample', 'Date', 'Time', 'Size (nm)', 'Exposure Time (s)', 'Retardation (Degrees)', 'Integration']])