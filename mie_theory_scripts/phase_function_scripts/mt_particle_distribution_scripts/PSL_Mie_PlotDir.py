'''
Austen K. Scruggs
12-04-2018
This script plots many mie theory text files
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

Mie_Dir = '/home/austen/Documents/Mie/txt_files/612 v 630'
Exp_Dir = '/home/austen/Documents/Good_Data/PSL_600nm_T4/SD_Offline.txt'
Save_Dir = '/home/austen/Documents'
# list files in directory
file_list = os.listdir(Mie_Dir)
#print(file_list)
# number of files in directory
num_files = len(file_list)
Exp_Data = pd.read_csv(Exp_Dir, sep=',')
slope = .2056
intercept = -45.2769
int_scalar = .015
f0, ax0 = plt.subplots(2, 2, figsize=(20, 7))
for counter, fn in enumerate(file_list):
    data = pd.read_csv(Mie_Dir + '/' + str(fn), sep=',')
    ax0[0, 0].plot(data['Theta'], data['SL'], label=str(fn.split('_')[0]))
    ax0[0, 1].plot(data['Theta'], data['SR'], label=str(fn.split('_')[0]))
    ax0[1, 0].plot(data['Theta'], data['SU'], label=str(fn.split('_')[0]))
    ax0[1, 1].plot(data['Theta'], np.asarray(data['SR'])/np.asarray(data['SL']), label=str(fn.split('_')[0]))
ax0[0, 0].plot((np.array(Exp_Data['Columns']) * slope) + intercept, (np.asarray(Exp_Data['Sample Intensity']) - np.asarray(Exp_Data['Nitrogen Intensity'])) * int_scalar, color='black', ls='--', label='Exp Data')
ax0[0, 0].set_title('600nm PSL Scattering Diagram of 663nm Light Polarized\n Perpendicular to the Scattering Plane')
ax0[0, 0].set_yscale('log')
ax0[0, 0].set_xlabel('Theta')
ax0[0, 0].set_ylabel('Intensity')
ax0[0, 0].legend(loc=1, bbox_to_anchor=[1.155, 1.0])
ax0[0, 0].grid(True)
ax0[0, 1].set_title('600nm PSL Scattering Diagram of 663nm Light Polarized\n Parallel to the Scattering Plane')
ax0[0, 1].set_yscale('log')
ax0[0, 1].set_xlabel('Theta')
ax0[0, 1].set_ylabel('Intensity')
ax0[0, 1].legend(loc=1, bbox_to_anchor=[1.155, 1.0])
ax0[0, 1].grid(True)
ax0[1, 0].set_title('600nm PSL Scattering Diagram of 663nm Light Polarized\n Unpolarized to the Scattering Plane')
ax0[1, 0].set_yscale('log')
ax0[1, 0].set_xlabel('Theta')
ax0[1, 0].set_ylabel('Intensity')
ax0[1, 0].legend(loc=1, bbox_to_anchor=[1.155, 1.0])
ax0[1, 0].grid(True)
ax0[1, 1].set_title('600nm PSL DLP')
ax0[1, 1].set_yscale('log')
ax0[1, 1].set_xlabel('Theta')
ax0[1, 1].set_ylabel('Intensity')
ax0[1, 1].legend(loc=1, bbox_to_anchor=[1.155, 1.0])
ax0[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Dir + '/PSL_600nm.pdf', format='pdf')
plt.show()

