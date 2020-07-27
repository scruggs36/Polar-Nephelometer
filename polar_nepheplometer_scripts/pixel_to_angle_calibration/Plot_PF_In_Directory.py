'''
Austen K. Scruggs
06-25-2020
Description: Plot the SDs for each exposure time
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import PyMieScatt as PMS
from math import pi
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelextrema

# Directories
data_directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-06-26/Analysis/PFs_10mW/180_deg'
save_directory = '/home/sm3/Desktop/Recent/'

# list of functions used
def Gaussian(x, mu, sigma, N):
   return N * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# list/print/number files in directory
file_list = os.listdir(data_directory)
#order = [1, 0, 4, 3, 5, 2]
#ordered_file_list = [file_list[i] for i in order]
num_files = len(file_list)


# create figure
f0, ax0 = plt.subplots(1, 2, figsize=(18, 6))
# initiate looped plotting
for file in file_list:
    data = pd.read_csv(data_directory + '/' + str(file), sep=',', header=0)
    columns = data['Sample Columns']
    pf_riemann = data['Sample Intensity']
    pf_gfit = data['Sample Intensity gfit']
    ax0[0].plot(columns, pf_riemann, ls='-', label=str(file))
    ax0[1].plot(columns, pf_gfit, ls='-', label=str(file))
ax0[0].set_xlabel('Column', fontsize=12)
ax0[0].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax0[0].set_title('Riemann Sum of Profile\nAlong Columnar Transect', fontsize=12)
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].set_xlabel('Column', fontsize=12)
ax0[1].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax0[1].set_title('Integrated Gaussian Fit of Profile\nAlong Columnar Transect', fontsize=12)
ax0[1].grid(True)
ax0[1].legend(loc=1)
# this must be done so that the figure title (suptitle) is spaced apart from the top x axis!
plt.suptitle('1000nm PSL Phase Function at Parallel Polarization and Five Second Exposure Time')
plt.savefig(save_directory + 'PF v mW.png', format='png')
plt.savefig(save_directory + 'PF v mW.pdf', format='pdf')
plt.show()

# Mie Theory
m_val = 1.59
wav_val = 663.0
d_val = 1000.0
med_val = 1.0
dp_vals = np.arange(d_val-100, d_val+100, 1.0)
ndp_vals = [Gaussian(i, d_val, 1.005, 100) for i in dp_vals]
Radians_Mie, SL_Mie, SR_Mie, SU_Mie = PMS.SF_SD(m=m_val, wavelength=wav_val, dp=dp_vals, ndp=ndp_vals, nMedium=med_val, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
# radians to degrees
Theta_Mie = np.array(Radians_Mie) * 180.0/pi
# normalize the Mie phase functions
SL_Mie_Norm = SL_Mie / np.sum(SL_Mie)
SR_Mie_Norm = SR_Mie / np.sum(SR_Mie)
SU_Mie_Norm = SU_Mie / np.sum(SU_Mie)
# create figure
f1, ax1 = plt.subplots(1, 2, figsize=(18, 6))
# intitiate looped plotting
for file in file_list:
    data = pd.read_csv(data_directory + '/' + str(file), sep=',', header=0)
    columns = np.array(data['Sample Columns'])
    pf_riemann = np.array(data['Sample Intensity'])
    pf_gfit = np.array(data['Sample Intensity gfit'])
    # normalize
    pf_riemann_norm = pf_riemann / np.nansum(pf_riemann)
    pf_gfit_norm = pf_gfit / np.nansum(pf_gfit)
    print(np.nansum(pf_gfit))
    # pixel to angle calibration data
    slope = 0.2120
    intercept = -5.1631
    Theta_Exp = np.array([(slope * i) + intercept for i in columns])
    # plot measurements
    ax1[0].semilogy(Theta_Exp, pf_riemann_norm, ls='-', label=str(file))
    ax1[1].semilogy(Theta_Exp, pf_gfit_norm, ls='-', label=str(file))
#plot theory and add details to figure
ax1[0].semilogy(Theta_Mie, SR_Mie_Norm, color='black', ls='-', lw=2, label='SL Mie Theory')
ax1[0].set_xlabel('Column', fontsize=12)
ax1[0].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax1[0].set_title('Riemann Sum of Profile\nAlong Columnar Transect', fontsize=12)
ax1[0].grid(True)
ax1[0].legend(loc=1)
#plot theory and add details to figure
ax1[1].semilogy(Theta_Mie, SR_Mie_Norm, color='black', ls='-', lw=2, label='SL Mie Theory')
ax1[1].set_xlabel('Column', fontsize=12)
ax1[1].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax1[1].set_title('Integrated Gaussian Fit of Profile\nAlong Columnar Transect', fontsize=12)
ax1[1].grid(True)
ax1[1].legend(loc=1)
# this must be done so that the figure title (suptitle) is spaced apart from the top x axis!
plt.suptitle('1000nm PSL Normalized Phase Functions at Parallel Polarization and Five Second Exposure Times')
plt.savefig(save_directory + 'PF v mW_Calibrated.png', format='png')
plt.savefig(save_directory + 'PF v mW_Calibrated.pdf', format='pdf')
plt.show()


