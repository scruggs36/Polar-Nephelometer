'''
Austen K. Scruggs
12-18-2018
Description: For a single exposure time, all the SDs collected, it takes the average and standard deviation
 of the intensity at every angle in the scattering diagram and plots it
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#
theory_path = '/home/austen/Documents/PSL600nm_MieTheory.txt'
theory = pd.read_csv(theory_path, sep=',', header=0)
sd_theory_int = theory['SU']
sd_theory_theta = theory['Theta']
# get file names in directory as list
file_path = '/home/austen/Documents/Good_Data/600_SDs_1s'
files = os.listdir(file_path)

# create empty array to fill
sd_2darray_int = []
sd_2darray_theta = []

# for loop
for file in files:
    slope = .2056
    intercept = -45.2769
    data = pd.read_csv(file_path + '/' + str(file), sep=',', header=0)
    sample_sd_int = np.array(data['Sample Intensity']) - np.array(data['Nitrogen Intensity'])
    print(file)
    print(len(sample_sd_int))
    sample_sd_theta = (slope * np.array(data['Columns'])) + intercept
    sd_2darray_int.append(sample_sd_int)
    sd_2darray_theta.append(sample_sd_theta)

# we calculate mean and standard deviation for each angle, also, the standard error is how different the sample mean is
# likely to be from the population mean, the standard deviation is how different each sample measurement is different from the
# mean sample measurement
mean_sd_int = np.mean(sd_2darray_int, axis=0)
std_sd_int = np.std(sd_2darray_int, axis=0)
mean_sd_theta = np.mean(sd_2darray_theta, axis=0)
std_sd_theta = np.std(sd_2darray_theta, axis=0)
scalar = sd_theory_int[163]/mean_sd_int[360]
mean_sd_int_scaled = scalar * mean_sd_int
std_sd_int_scaled = scalar * std_sd_int
# plot
f0, ax0 = plt.subplots(figsize=(10, 6))
ax0.semilogy(mean_sd_theta, mean_sd_int, color='dodgerblue', ls='-', label='Meas. Avg')
ax0.fill_between(mean_sd_theta, mean_sd_int + std_sd_int, mean_sd_int - std_sd_int, color='deepskyblue', label='Meas. Stdev')
ax0.semilogy(mean_sd_theta, mean_sd_int_scaled, color='green', ls='-', label='Meas. Avg Scaled')
ax0.fill_between(mean_sd_theta, mean_sd_int_scaled + std_sd_int_scaled, mean_sd_int_scaled - std_sd_int_scaled, color='yellowgreen', label='Meas. Stdev Scaled')
ax0.semilogy(sd_theory_theta, sd_theory_int, color='red', ls='-', label='Theory')
ax0.semilogy(mean_sd_theta[360], mean_sd_int[360], color='orange', ls='', marker='X', ms='7', label='Meas. Scaling Point')
ax0.semilogy(sd_theory_theta[163], sd_theory_int[163], color='orange', ls='', marker='X', ms='7', label='Theory Scaling Point')
ax0.set_title('Scattering Diagram Mean and Standard Deviation')
ax0.set_ylabel('Intensity')
ax0.set_xlabel('\u0398')
ax0.legend(loc=1)
ax0.grid(True)
plt.tight_layout()
plt.savefig('/home/austen/Documents/Mean_STDEV_PSL600nm.pdf', format='pdf')
plt.show()


