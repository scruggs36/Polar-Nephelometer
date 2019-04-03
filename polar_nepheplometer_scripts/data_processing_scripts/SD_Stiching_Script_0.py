'''
Austen K. Scruggs
12-10-2018
Description: Stitching scattering diagrams together of two different exposure times!
Ultimately the stitching script doesn't work due to leveling off at saturation issues!
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


# import data
SD_Path_1 = '/home/austen/Documents/Good_Data/PSL_600nm_T4/SD_Offline.txt'
SD_Path_2 = '/home/austen/Documents/Good_Data/PSL_600nm_T5/SD_Offline.txt'

Data_1 = pd.read_csv(SD_Path_1, sep=',')
Data_2 = pd.read_csv(SD_Path_2, sep=',')

Exp_Col_1 = Data_1['Columns']
Exp_Col_2 = Data_2['Columns']

Exp_Int_N2_1 = Data_1['Nitrogen Intensity']
Exp_Int_N2_2 = Data_2['Nitrogen Intensity']

Exp_Int_Sample_1 = Data_1['Sample Intensity']
Exp_Int_Sample_2 = Data_2['Sample Intensity']

SD_1 = np.array(Exp_Int_Sample_1) - np.array(Exp_Int_N2_1)
SD_2 = np.array(Exp_Int_Sample_2) - np.array(Exp_Int_N2_2)
SD_1_Norm = SD_1 / norm(SD_1)
SD_2_Norm = SD_2 / norm(SD_2)

# this is how the automated stiching works
SD_Diff = SD_2_Norm - SD_1_Norm
zero_val = min(SD_Diff, key=lambda x: abs(x-0))
print(zero_val)
idx = np.asscalar(np.where(SD_Diff == zero_val)[0])
print(idx)

# profiles to angles
slope = 0.2056
intercept = -45.2769
print(((idx + Exp_Col_1[0]) * slope) + intercept)

Exp_Theta_1 = (slope * np.array(Exp_Col_1)) + intercept
Exp_Theta_2 = (slope * np.array(Exp_Col_2)) + intercept

f0, ax0 = plt.subplots(1, 3, figsize=(10, 6))
ax0[0].semilogy(Exp_Theta_1, SD_1, color='green', ls='-', label='1s')
ax0[0].semilogy(Exp_Theta_2, SD_2, color='blue', ls='-', label='15s')
ax0[0].set_ylabel('Intensity')
ax0[0].set_xlabel('\u0398')
ax0[0].set_title('Scattering Diagrams')
ax0[0].legend(loc=1)
ax0[0].grid(True)
ax0[1].semilogy(Exp_Theta_1, SD_1 / norm(SD_1), color='green', ls='-', label='1s')
ax0[1].semilogy(Exp_Theta_2, SD_2 / norm(SD_2), color='blue', ls='-', label='15s')
ax0[1].semilogy((((idx + Exp_Col_1[0]) * slope) + intercept), SD_2_Norm[idx], color='red', ls='', marker='X', ms='7')
ax0[1].set_ylabel('Intensity')
ax0[1].set_xlabel('\u0398')
ax0[1].set_title('Scattering Diagrams')
ax0[1].legend(loc=1)
ax0[1].grid(True)
ax0[2].semilogy(slope * np.concatenate((Exp_Col_1[0:idx], Exp_Col_2[idx:-1]), axis=0) + intercept, np.concatenate((SD_1_Norm[0:idx], SD_2_Norm[idx:-1]), axis=0), color='purple', ls='-', label='Stiched SD')
ax0[2].set_ylabel('Intensity')
ax0[2].set_xlabel('\u0398')
ax0[2].set_title('Scattering Diagrams')
ax0[2].legend(loc=1)
ax0[2].grid(True)
plt.tight_layout()
plt.show()
