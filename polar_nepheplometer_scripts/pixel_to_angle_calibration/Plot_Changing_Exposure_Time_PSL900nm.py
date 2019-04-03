'''
Austen K. Scruggs
11-06-2018
Description: Plot the SDs for each exposure time
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelextrema

# Directories
Data_Dir = '/home/austen/Documents/Good_Data/PSL_900nm_T6/SDs'
Save_Directory = '/home/austen/Documents'
# list/print/number files in directory
file_list = os.listdir(Data_Dir)
#order = [1, 0, 4, 3, 5, 2]
#ordered_file_list = [file_list[i] for i in order]
num_files = len(file_list)
#print(file_list)

# pixel to angle calibration data
slope = 0.21116737541023334
intercept = -47.97208663718679
int_scalar = .015

def funct(t, a, b, c, d):
    return a*np.exp(-b*t) + c*t + d
# set up subplots

local_max = []
lmax = [34, 236, 440, 660]
local_min = []
lmin = [179, 384, 531, 694]
int_v_exposure_lmax0 = []
int_v_exposure_lmax1 = []
int_v_exposure_lmax2 = []
int_v_exposure_lmax3 = []
f0, ax0 = plt.subplots(1, 3, figsize=(20, 5))
for counter, fn in enumerate(file_list):
    SD = pd.read_csv(Data_Dir + '/' + str(fn), header=0, sep=',')
    theta = (np.array(SD['Columns']) * slope) + intercept
    intensity = (np.array(SD['Sample Intensity']) - np.array(SD['Nitrogen Intensity']))
    intensity_filtered = savgol_filter(intensity, window_length=151, polyorder=2, deriv=0)
    ax0[0].semilogy(theta, intensity, label=str(fn).replace('.', '_').split('_')[2])
    ax0[1].semilogy(theta, intensity_filtered, label=str(fn).replace('.', '_').split('_')[2])
    local_max.append(argrelextrema(intensity_filtered, np.greater, order=10))
    local_min.append(argrelextrema(intensity_filtered, np.less, order=10))
    #print(local_min)
    #print([theta[element] for element in local_min])
    int_v_exposure_lmax0.append(intensity_filtered[lmax[0]])
    int_v_exposure_lmax1.append(intensity_filtered[lmax[1]])
    int_v_exposure_lmax2.append(intensity_filtered[lmax[2]])
    int_v_exposure_lmax3.append(intensity_filtered[lmax[3]])
    ax0[0].plot([theta[element] for element in lmax], [intensity[element] for element in lmax], color='black', marker='X', ls='')
    #ax0[1].plot([theta[element] for element in lmax], [intensity_filtered[element] for element in lmax], color='black', marker='X', ls='')
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
ax0[2].plot([3, 6, 9, 12, 15], sorted(int_v_exposure_lmax0), marker='o', ls='', label='local max 0')
ax0[2].plot([3, 6, 9, 12, 15], sorted(int_v_exposure_lmax1), marker='*', ls='', label='local max 1')
ax0[2].plot([3, 6, 9, 12, 15], sorted(int_v_exposure_lmax2), marker='x', ls='', label='local max 2')
ax0[2].plot([3, 6, 9, 12, 15], sorted(int_v_exposure_lmax3), marker='^', ls='', label='local max 3')
ax0[2].legend(loc=1)
ax0[2].set_xlabel("Exposure Time (s)", fontsize=12)
ax0[2].set_ylabel('Intensity', fontsize=12, labelpad=10)
ax0[2].set_title('Intensity vs. Exposure', fontsize=12)
ax0[2].grid(True)
# this must be done so that the figure title (suptitle) is spaced apart from the top x axis!
plt.tight_layout()
plt.savefig(Save_Directory + '/Exp_V_MieTheory.pdf', format='pdf')
plt.show()
