'''
Austen K. Scruggs
11-06-2018
Description: Plot the SDs for each exposure time
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter, argrelextrema

# Directories
Data_Dir = '/home/austen/Documents/PSL_600nm_v_Exposure/T2/SDs'
Save_Directory = '/home/austen/Documents'
# list/print/number files in directory
file_list = os.listdir(Data_Dir)
order = [3, 2, 0, 4, 5, 1]
ordered_file_list = [file_list[i] for i in order]
num_files = len(file_list)
#print(file_list)

# pixel to angle calibration data
slope = .2056
intercept = -45.2769
int_scalar = .015

local_max = []
lmax = [365, 506, 645]
local_min = []
lmin = [365, 564, 645]
int_v_exposure_lmax0 = []
int_v_exposure_lmax1 = []
int_v_exposure_lmax2 = []
int_v_exposure_lmax3 = []
f0, ax0 = plt.subplots(1, 3, figsize=(20, 5))
for counter, fn in enumerate(ordered_file_list):
    SD = pd.read_csv(Data_Dir + '/' + str(fn), header=0, sep=',')
    theta = (np.array(SD['Columns']) * slope) + intercept
    intensity = (np.array(SD['Sample Intensity']) - np.array(SD['Nitrogen Intensity']))
    intensity_filtered = savgol_filter(intensity, window_length=301, polyorder=2, deriv=0)
    ax0[0].semilogy(theta, intensity, label=str(fn).replace('.', '_').split('_')[2])
    ax0[1].semilogy(theta, intensity_filtered, label=str(fn).replace('.', '_').split('_')[2])
    local_max.append(argrelextrema(intensity_filtered, np.greater, order=10))
    local_min.append(argrelextrema(intensity_filtered, np.less, order=10))
    print(local_min)
    print([theta[element] for element in local_min])
    int_v_exposure_lmax0.append(intensity_filtered[lmax[0]])
    int_v_exposure_lmax1.append(intensity_filtered[lmax[1]])
    int_v_exposure_lmax2.append(intensity_filtered[lmax[2]])
    #int_v_exposure_lmax3.append(intensity_filtered[lmax[3]])
    ax0[0].plot([theta[element] for element in lmax], [intensity[element] for element in lmax], color='black', marker='X', ls='')
    ax0[1].plot([theta[element] for element in lmax], [intensity_filtered[element] for element in lmax], color='black', marker='X', ls='')
ax0[0].legend(loc=1)
ax0[0].set_xlabel("\u03b8", fontsize=12)
ax0[0].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax0[0].set_title('900nm PSL Raw \n Scattering Diagrams', fontsize=12)
ax0[0].grid(True)
ax0[1].legend(loc=1)
ax0[1].set_xlabel("\u03b8", fontsize=12)
ax0[1].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax0[1].set_title('900nm PSL Filtered \n Scattering Diagram', fontsize=12)
ax0[1].grid(True)
ax0[2].plot([1, 3, 6, 9, 12, 15], sorted(int_v_exposure_lmax0), color='red', marker='o', ls='', label='local max 0')
ax0[2].plot([1, 3, 6, 9, 12, 15], sorted(int_v_exposure_lmax1), color='green', marker='*', ls='', label='local max 1')
ax0[2].plot([1, 3, 6, 9, 12, 15], sorted(int_v_exposure_lmax2), color='blue', marker='x', ls='', label='local max 2')
m0, b0, r0, p0, stderr0 = linregress([1, 3, 6, 9, 12, 15], sorted(int_v_exposure_lmax0))
m1, b1, r1, p1, stderr1 = linregress([1, 3, 6, 9, 12, 15], sorted(int_v_exposure_lmax0))
m2, b2, r2, p2, stderr2 = linregress([1, 3, 6, 9, 12, 15], sorted(int_v_exposure_lmax0))
ax0[2].plot([1, 3, 6, 9, 12, 15], (m0 * np.array([1, 3, 6, 9, 12, 15])) + b0, marker='', color='red', ls='--', label='local max 0 \n y = ' + str('{:.3f}'.format(m0)) + 'x ' + str('{:.3f}'.format(b0)) + ' $R^2$: ' + str('{:.3f}'.format(r0)))
ax0[2].plot([1, 3, 6, 9, 12, 15], (m1 * np.array([1, 3, 6, 9, 12, 15])) + b1, marker='', color='green', ls='--', label='local max 1 \n y = ' + str('{:.3f}'.format(m1)) + 'x ' + str('{:.3f}'.format(b1)) + ' $R^2$: ' + str('{:.3f}'.format(r1)))
ax0[2].plot([1, 3, 6, 9, 12, 15], (m2 * np.array([1, 3, 6, 9, 12, 15])) + b2, marker='', color='blue', ls='--', label='local max 2 \n y = ' + str('{:.3f}'.format(m2)) + 'x ' + str('{:.3f}'.format(b2)) + ' $R^2$: ' + str('{:.3f}'.format(r2)))
ax0[2].legend(loc=1)
ax0[2].set_xlabel("Exposure Time (s)", fontsize=12)
ax0[2].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax0[2].set_title('Intensity vs. Exposure', fontsize=12)
ax0[2].grid(True)
# this must be done so that the figure title (suptitle) is spaced apart from the top x axis!
plt.tight_layout()
plt.savefig(Save_Directory + '/Exp_V_MieTheory.pdf', format='pdf')
plt.show()
