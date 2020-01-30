'''
Austen K. Scruggs
01-27-2019
Description: Plots measurement and mie theory data together just so I can see how good the matches are.
'''

import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt


SL_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-27/2020-01-27_Analysis/0R/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])
SL_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-27/2020-01-27_Analysis/0R/SD_Particle.txt', header=0, sep=',')['Sample Columns'])

SR_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-27/2020-01-27_Analysis/0.5R/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])
SR_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-27/2020-01-27_Analysis/0.5R/SD_Particle.txt', header=0, sep=',')['Sample Columns'])

SU_EXP = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-27/2020-01-27_Analysis/0.25R/SD_Particle.txt', header=0, sep=',')['Sample Intensity'])
SU_X = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-27/2020-01-27_Analysis/0.25R/SD_Particle.txt', header=0, sep=',')['Sample Columns'])

SL_MT = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-20/2020-01-20_Analysis/MT/Mie_Theory_DF.txt', header=0, sep=',')['Gienger SL'])
SR_MT = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-20/2020-01-20_Analysis/MT/Mie_Theory_DF.txt', header=0, sep=',')['Gienger SR'])
SU_MT = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-20/2020-01-20_Analysis/MT/Mie_Theory_DF.txt', header=0, sep=',')['Gienger SU'])
MT_theta = np.array(pd.read_csv('/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-20/2020-01-20_Analysis/MT/Mie_Theory_DF.txt', header=0, sep=',')['Theta'])

slope = .2052
intercept = -0.7795
scalar = 0.005

SL_theta = [slope * element + intercept for element in SL_X]
SR_theta = [slope * element + intercept for element in SR_X]
SU_theta = [slope * element + intercept for element in SU_X]

SL_MT = pchip_interpolate(MT_theta, SL_MT, SL_theta)
SR_MT = pchip_interpolate(MT_theta, SR_MT, SR_theta)
SU_MT = pchip_interpolate(MT_theta, SU_MT, SU_theta)

SL_Residuals = (((SL_EXP * scalar) - SL_MT) / SL_MT) * 100
SR_Residuals = (((SR_EXP * scalar) - SR_MT) / SR_MT) * 100
SU_Residuals = (((SU_EXP * scalar) - SU_MT) / SU_MT) * 100

f, ax = plt.subplots(2, 1, figsize=(12, 6))
ax[0].semilogy(SL_theta, SL_EXP * scalar, color='red', ls='-', label='SL EXP')
ax[0].semilogy(SR_theta, SR_EXP * scalar, color='green', ls='-', label='SR EXP')
ax[0].semilogy(SU_theta, SU_EXP * scalar, color='blue', ls='-', label='SU EXP')
ax[0].semilogy(SL_theta, SL_MT, color='red', ls='--', label='SL MT')
ax[0].semilogy(SR_theta, SR_MT, color='green', ls='--', label='SR MT')
ax[0].semilogy(SU_theta, SU_MT, color='blue', ls='--', label='SU MT')
ax[0].set_xlabel('\u0398')
ax[0].set_ylabel('Scaled Intensity')
ax[0].set_title('Measured and Calculated Phase Functions of 900nm Polystyrene Latex Spheres')
ax[0].grid(True)
ax[0].legend(loc=1, ncol=2)
ax[1].plot(SL_theta, SL_Residuals, color='red', ls='-', label='SL Error')
ax[1].plot(SR_theta, SR_Residuals, color='green', ls='-', label='SR Error')
ax[1].plot(SU_theta, SU_Residuals, color='blue', ls='-', label='SU Error')
ax[1].set_xlabel('\u0398')
ax[1].set_ylabel('%')
ax[1].set_title('Percent Error')
ax[1].grid(True)
ax[1].legend(loc=1)
plt.tight_layout()
plt.show()
