'''
Austen K. Scruggs
01-27-2019
Description: Plots measurement and mie theory data together just so I can see how good the matches are.
'''

import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt


SL_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/0R/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])[50:-50]
SL_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/0R/SD_Particle.txt', header=0, sep=',')['Sample Columns'])[50:-50]

SR_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/0.5R/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])[50:-50]
SR_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/0.5R/SD_Particle.txt', header=0, sep=',')['Sample Columns'])[50:-50]

SU_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/0.25R/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])[50:-50]
SU_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/0.25R/SD_Particle.txt', header=0, sep=',')['Sample Columns'])[50:-50]

SU_QWP_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/qw/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])[50:-50]
SU_QWP_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/Measurements/2s/qw/SD_Particle.txt', header=0, sep=',')['Sample Columns'])[50:-50]

SL_MT = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/MT800/Mie_Theory_DF.txt', header=0, sep=',')['Gienger SL'])
SR_MT = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/MT800/Mie_Theory_DF.txt', header=0, sep=',')['Gienger SR'])
SU_MT = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/MT800/Mie_Theory_DF.txt', header=0, sep=',')['Gienger SU'])
MT_theta = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-04/2020-02-04_Analysis/MT800/Mie_Theory_DF.txt', header=0, sep=',')['Theta'])

slope = .2095
intercept = -3.1433
scalar = 0.005

SL_theta = [slope * element + intercept for element in SL_X]
SR_theta = [slope * element + intercept for element in SR_X]
SU_theta = [slope * element + intercept for element in SU_X]
SU_QWP_theta = [slope * element + intercept for element in SU_QWP_X]

SL_MT = pchip_interpolate(MT_theta, SL_MT, SL_theta)
SR_MT = pchip_interpolate(MT_theta, SR_MT, SR_theta)
SU_MT = pchip_interpolate(MT_theta, SU_MT, SU_theta)

SL_Residuals = (((SL_EXP * scalar) - SL_MT) / SL_MT) * 100
SR_Residuals = (((SR_EXP * scalar) - SR_MT) / SR_MT) * 100
SU_Residuals = (((SU_EXP * scalar) - SU_MT) / SU_MT) * 100
SU_QWP_Residuals = (((SU_QWP_EXP * scalar) - SU_MT) / SU_MT) * 100

f, ax = plt.subplots(2, 3, figsize=(12, 6))
ax[0, 0].semilogy(SL_theta, SL_EXP * scalar, color='red', ls='-', label='SL EXP')
ax[0, 0].semilogy(SL_theta, SL_MT, color='red', ls='--', label='SL MT')
ax[0, 0].set_xlabel('\u0398')
ax[0, 0].set_ylabel('Scaled Intensity')
ax[0, 0].set_title('Measured and Calculated Phase Functions of\n 800nm Polystyrene Latex Spheres')
ax[0, 0].grid(True)
ax[0, 0].legend(loc=1)
ax[0, 1].semilogy(SR_theta, SR_EXP * scalar, color='green', ls='-', label='SR EXP')
ax[0, 1].semilogy(SR_theta, SR_MT, color='green', ls='--', label='SR MT')
ax[0, 1].set_xlabel('\u0398')
ax[0, 1].set_ylabel('Scaled Intensity')
ax[0, 1].set_title('Measured and Calculated Phase Functions of\n 800nm Polystyrene Latex Spheres')
ax[0, 1].grid(True)
ax[0, 1].legend(loc=1)
ax[0, 2].semilogy(SU_theta, SU_EXP * scalar, color='blue', ls='-', label='SU EXP')
ax[0, 2].semilogy(SU_theta, SU_QWP_EXP * scalar, color='purple', ls='-', label='SU QWP EXP')
ax[0, 2].semilogy(SU_theta, SU_MT, color='blue', ls='--', label='SU MT')
ax[0, 2].set_xlabel('\u0398')
ax[0, 2].set_ylabel('Scaled Intensity')
ax[0, 2].set_title('Measured and Calculated Phase Functions of\n 800nm Polystyrene Latex Spheres')
ax[0, 2].grid(True)
ax[0, 2].legend(loc=1)
ax[1, 0].plot(SL_theta, SL_Residuals, color='red', ls='-', label='SL Error')
ax[1, 0].set_xlabel('\u0398')
ax[1, 0].set_ylabel('%')
ax[1, 0].set_title('Percent Error')
ax[1, 0].grid(True)
ax[1, 0].legend(loc=1)
ax[1, 1].plot(SR_theta, SR_Residuals, color='green', ls='-', label='SR Error')
ax[1, 1].set_xlabel('\u0398')
ax[1, 1].set_ylabel('%')
ax[1, 1].set_title('Percent Error')
ax[1, 1].grid(True)
ax[1, 1].legend(loc=1)
ax[1, 2].plot(SU_theta, SU_Residuals, color='blue', ls='-', label='SU Error')
ax[1, 2].plot(SU_theta, SU_QWP_Residuals, color='purple', ls='-', label='SU QWP Error')
ax[1, 2].set_xlabel('\u0398')
ax[1, 2].set_ylabel('%')
ax[1, 2].set_title('Percent Error')
ax[1, 2].grid(True)
ax[1, 2].legend(loc=1)
plt.tight_layout()
plt.show()
