'''
Austen K. Scruggs
10-02-2019
Description: After finding the linear calibratoin using the Profile_Number_To_Scattering_Angle_Calibration_6.py
script, use this to plot up the data for 600nm, 800nm, and 903nm PSL
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# calibration slope and y intercept
def calibration(pixels, m, b):
    theta = [(element * m) + b for element in np.array(pixels)]
    return theta


# slope and intercept
slope = .2046
intercept = -43.8992

# save directory
save = '/home/austen/Desktop/2019-09-26_Analysis/'

# path for 900nm PSL experimental data
exp_903_path_SL = '/home/austen/Desktop/2019-09-26_Analysis/PSL900nm/0.2s/0lamda/SD_Particle.txt'
exp_903_path_SR = '/home/austen/Desktop/2019-09-26_Analysis/PSL900nm/0.2s/0.5lamda/SD_Particle.txt'
exp_903_path_SU = '/home/austen/Desktop/2019-09-26_Analysis/PSL900nm/0.2s/0.25lamda/SD_Particle.txt'

# path for 900nm PSL Mie theory data
mt_903_path = '/home/austen/Desktop/2019-09-26_Analysis/PSL900nm/Mie_Theory_DF.txt'

# read experimental data
exp_903_SL = pd.read_csv(exp_903_path_SL)
exp_903_SL_theta = calibration(exp_903_SL['Sample Columns'], slope, intercept)
exp_903_SL_intensity = np.array(exp_903_SL['Sample Intensity'])
exp_903_SR = pd.read_csv(exp_903_path_SR)
exp_903_SR_theta = calibration(exp_903_SR['Sample Columns'], slope, intercept)
exp_903_SR_intensity = np.array(exp_903_SR['Sample Intensity'])
exp_903_SU = pd.read_csv(exp_903_path_SU)
exp_903_SU_theta = calibration(exp_903_SU['Sample Columns'], slope, intercept)
exp_903_SU_intensity = np.array(exp_903_SU['Sample Intensity'])

# read Mie theory data
mt_903_data = pd.read_csv(mt_903_path)
mt_903_theta = mt_903_data['Theta']
mt_903_SL = mt_903_data['Gienger SL']
mt_903_SR = mt_903_data['Gienger SR']
mt_903_SU = mt_903_data['Gienger SU']

scalar_900 = .01
f903, ax903 = plt.subplots(1, 3, figsize=(12, 6))
ax903[0].semilogy(exp_903_SL_theta, scalar_900 * exp_903_SL_intensity, label='scaled measurement', color='k', ls='-')
ax903[0].semilogy(mt_903_theta, mt_903_SL, label='theory (Gienger)', color='r', ls='--')
ax903[0].set_xlabel('\u0398')
ax903[0].set_ylabel('Intensity')
ax903[0].set_title('900nm PSL SL Phase Function')
ax903[0].grid(True)
ax903[0].legend(loc=1)

ax903[1].semilogy(exp_903_SR_theta, scalar_900 * exp_903_SR_intensity, label='scaled measurement', color='k', ls='-')
ax903[1].semilogy(mt_903_theta, mt_903_SR, label='theory (Gienger)', color='r', ls='--')
ax903[1].set_xlabel('\u0398')
ax903[1].set_ylabel('Intensity')
ax903[1].set_title('900nm PSL SR Phase Function')
ax903[1].grid(True)
ax903[1].legend(loc=1)

ax903[2].semilogy(exp_903_SU_theta, scalar_900 * exp_903_SU_intensity, label='scaled measurement', color='k', ls='-')
ax903[2].semilogy(mt_903_theta, mt_903_SU, label='theory (Gienger)', color='r', ls='--')
ax903[2].set_xlabel('\u0398')
ax903[2].set_ylabel('Intensity')
ax903[2].set_title('900nm PSL SU Phase Function')
ax903[2].grid(True)
ax903[2].legend(loc=1)

plt.tight_layout()
plt.savefig(save + 'mt_v_meas_PSL_903.png', format='png')
plt.savefig(save + 'mt_v_meas_PSL_903.pdf', format='pdf')
plt.show()

# path for 800nm PSL experimental data
exp_800_path_SL = '/home/austen/Desktop/2019-09-26_Analysis/PSL800nm/3s/0lamda/SD_Particle.txt'
exp_800_path_SR = '/home/austen/Desktop/2019-09-26_Analysis/PSL800nm/3s/0.5lamda/SD_Particle.txt'
exp_800_path_SU = '/home/austen/Desktop/2019-09-26_Analysis/PSL800nm/3s/0.25lamda/SD_Particle.txt'

# path for 800nm PSL Mie theory data
mt_800_path = '/home/austen/Desktop/2019-09-26_Analysis/PSL800nm/Mie_Theory_DF.txt'

# read 800nm PSL experimental data
exp_800_SL = pd.read_csv(exp_800_path_SL)
exp_800_SL_theta = calibration(exp_800_SL['Sample Columns'], slope, intercept)
exp_800_SL_intensity = np.array(exp_800_SL['Sample Intensity'])
exp_800_SR = pd.read_csv(exp_800_path_SR)
exp_800_SR_theta = calibration(exp_800_SR['Sample Columns'], slope, intercept)
exp_800_SR_intensity = np.array(exp_800_SR['Sample Intensity'])
exp_800_SU = pd.read_csv(exp_800_path_SU)
exp_800_SU_theta = calibration(exp_800_SU['Sample Columns'], slope, intercept)
exp_800_SU_intensity = np.array(exp_800_SU['Sample Intensity'])

# read Mie theory data
mt_800_data = pd.read_csv(mt_800_path)
mt_800_theta = mt_800_data['Theta']
mt_800_SL = mt_800_data['Gienger SL']
mt_800_SR = mt_800_data['Gienger SR']
mt_800_SU = mt_800_data['Gienger SU']

scalar_800 = .004
f800, ax800 = plt.subplots(1, 3, figsize=(12, 6))
ax800[0].semilogy(exp_800_SL_theta, scalar_800 * exp_800_SL_intensity, label='scaled measurement', color='k', ls='-')
ax800[0].semilogy(mt_800_theta, mt_800_SL, label='theory (Gienger)', color='r', ls='--')
ax800[0].set_xlabel('\u0398')
ax800[0].set_ylabel('Intensity')
ax800[0].set_title('800nm PSL SL Phase Function')
ax800[0].grid(True)
ax800[0].legend(loc=1)

ax800[1].semilogy(exp_800_SR_theta, scalar_800 * exp_800_SR_intensity, label='scaled measurement', color='k', ls='-')
ax800[1].semilogy(mt_800_theta, mt_800_SR, label='theory (Gienger)', color='r', ls='--')
ax800[1].set_xlabel('\u0398')
ax800[1].set_ylabel('Intensity')
ax800[1].set_title('800nm PSL SR Phase Function')
ax800[1].grid(True)
ax800[1].legend(loc=1)

ax800[2].semilogy(exp_800_SU_theta, scalar_800 * exp_800_SU_intensity, label='scaled measurement', color='k', ls='-')
ax800[2].semilogy(mt_800_theta, mt_800_SU, label='theory (Gienger)', color='r', ls='--')
ax800[2].set_xlabel('\u0398')
ax800[2].set_ylabel('Intensity')
ax800[2].set_title('800nm PSL SU Phase Function')
ax800[2].grid(True)
ax800[2].legend(loc=1)

plt.tight_layout()
plt.savefig(save + 'mt_v_meas_PSL_800.png', format='png')
plt.savefig(save + 'mt_v_meas_PSL_800.pdf', format='pdf')
plt.show()

# path 600nm PSL experimental data
exp_600_path_SL = '/home/austen/Desktop/2019-09-26_Analysis/PSL600nm/3s/0lamda/SD_Particle.txt'
exp_600_path_SR = '/home/austen/Desktop/2019-09-26_Analysis/PSL600nm/3s/0.5lamda/SD_Particle.txt'
exp_600_path_SU = '/home/austen/Desktop/2019-09-26_Analysis/PSL600nm/3s/0.25lamda/SD_Particle.txt'

# path 600nm PSL Mie theory data
mt_600_path = '/home/austen/Desktop/2019-09-26_Analysis/PSL600nm/Mie_Theory_DF.txt'

# read 600nm PSL experimental data
exp_600_SL = pd.read_csv(exp_600_path_SL)
exp_600_SL_theta = calibration(exp_600_SL['Sample Columns'], slope, intercept)
exp_600_SL_intensity = np.array(exp_600_SL['Sample Intensity'])
exp_600_SR = pd.read_csv(exp_600_path_SR)
exp_600_SR_theta = calibration(exp_600_SR['Sample Columns'], slope, intercept)
exp_600_SR_intensity = np.array(exp_600_SR['Sample Intensity'])
exp_600_SU = pd.read_csv(exp_600_path_SU)
exp_600_SU_theta = calibration(exp_600_SU['Sample Columns'], slope, intercept)
exp_600_SU_intensity = np.array(exp_600_SU['Sample Intensity'])

# read Mie theory data
mt_600_data = pd.read_csv(mt_600_path)
mt_600_theta = mt_600_data['Theta']
mt_600_SL = mt_600_data['Gienger SL']
mt_600_SR = mt_600_data['Gienger SR']
mt_600_SU = mt_600_data['Gienger SU']

scalar_600 = .002
f600, ax600 = plt.subplots(1, 3, figsize=(12, 6))
ax600[0].semilogy(exp_600_SL_theta, scalar_600 * exp_600_SL_intensity, label='scaled measurement', color='k', ls='-')
ax600[0].semilogy(mt_600_theta, mt_600_SL, label='theory (Gienger)', color='r', ls='--')
ax600[0].set_xlabel('\u0398')
ax600[0].set_ylabel('Intensity')
ax600[0].set_title('600nm PSL SL Phase Function')
ax600[0].grid(True)
ax600[0].legend(loc=1)

ax600[1].semilogy(exp_600_SR_theta, scalar_600 * exp_600_SR_intensity, label='scaled measurement', color='k', ls='-')
ax600[1].semilogy(mt_600_theta, mt_600_SR, label='theory (Gienger)', color='r', ls='--')
ax600[1].set_xlabel('\u0398')
ax600[1].set_ylabel('Intensity')
ax600[1].set_title('600nm PSL SR Phase Function')
ax600[1].grid(True)
ax600[1].legend(loc=1)

ax600[2].semilogy(exp_600_SU_theta, scalar_600 * exp_600_SU_intensity, label='scaled measurement', color='k', ls='-')
ax600[2].semilogy(mt_600_theta, mt_600_SU, label='theory (Gienger)', color='r', ls='--')
ax600[2].set_xlabel('\u0398')
ax600[2].set_ylabel('Intensity')
ax600[2].set_title('600nm PSL SU Phase Function')
ax600[2].grid(True)
ax600[2].legend(loc=1)

plt.tight_layout()
plt.savefig(save + 'mt_v_meas_PSL_600.png', format='png')
plt.savefig(save + 'mt_v_meas_PSL_600.pdf', format='pdf')
plt.show()

