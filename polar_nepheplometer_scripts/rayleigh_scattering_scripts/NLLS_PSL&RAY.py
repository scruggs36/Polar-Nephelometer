'''
Austen K. Scruggs
02-17-2020
Description: Heatmap
'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

# this tool is for finding non-idealities in the images!
'''
directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-05/PSL600/3s/2darray/PSL600_3s_0.25R_200avg_Average_Thu Mar 5 2020 8_04_09 PM.txt'
im = np.array(pd.read_csv(directory, sep='\t'))
print(im)

plt.pcolormesh(im, cmap='gray', vmax=1000, vmin=0)
plt.colorbar()
plt.show()
'''
'''
directory2 = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/CO2_100s_0R_5AVG_Average_Mon Feb 17 2020 12_27_45 PM.txt'
im2 = np.array(pd.read_csv(directory, sep='\t'))
print(im)

plt.pcolormesh(im2, cmap='gray', vmax=4095, vmin=0)
plt.colorbar()
plt.show()
'''
directory3 = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0.5R'
file_list = os.listdir(directory3)
#print(file_list)


rayleigh = []
theta = []
f, ax = plt.subplots(2, 1, figsize=(12, 6))
for file in file_list:
    print(file)
    data = pd.read_csv(directory3 + '/' + file, sep=',', header=0)
    #data = data.dropna()
    rayleigh = np.array(data['CO2 Intensity'])
    theta = np.array(data['CO2 Theta'])
    print(theta)
    ax[0].semilogy(theta, rayleigh, label=file)
    ax[1].plot(theta, rayleigh, label=file)



ax[0].set_xlabel('\u03b8')
ax[0].set_ylabel('Intensity(\u03a3(DN))')
ax[0].set_title('Rayleigh Scattering Measurements')
ax[0].grid(True)
ax[0].legend(bbox_to_anchor=[1.0, 0.75])

ax[1].set_xlabel('\u03b8')
ax[1].set_ylabel('Intensity(\u03a3(DN))')
ax[1].set_title('Rayleigh Scattering Measurements')
ax[1].grid(True)
ax[1].legend(bbox_to_anchor=[1.0, 0.75])
plt.tight_layout()
plt.savefig('/home/sm3/Desktop/Recent/All_CO2_Rayleigh_Cases_0R.png', format='png')
plt.show()
'''
def SL_SR_NLLS(x, SL_theory, SR_theory, measurement):
    residuals = measurement - (x[0] * SL_theory + x[1] * SR_theory)
    return residuals

sl_array = []
sr_array = []
ssr_array = []
retardance = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.125, 0.15, 0.20, 0.25]
for element in retardance:
    meas_dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-19/2020-02-19_Analysis/Polarization Test/' + str(element) + 'R/SD_Particle.txt'
    measured_sample = pd.read_csv(meas_dir, sep=',', header=0)['Sample Intensity'] / np.sum(pd.read_csv(meas_dir, sep=',', header=0)['Sample Intensity'])
    measured_columns = pd.read_csv(meas_dir, sep=',', header=0)['Sample Columns']
    measured_sample = savgol_filter(measured_sample, window_length=51, polyorder=3, deriv=0)
    measured_sample = measured_sample[100:-1]
    measured_columns = measured_columns[100:-1]

    slope = .2095
    intercept = -3.1433
    measured_theta = [(slope * element) + intercept for element in measured_columns]


    mt_dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/PSL_Analysis/MT800/Mie_Theory_DF.txt'
    mt_SL = pd.read_csv(mt_dir, sep=',', header=0)['Gienger SL'] / np.sum(pd.read_csv(mt_dir, sep=',', header=0)['Gienger SL'])
    mt_SR = pd.read_csv(mt_dir, sep=',', header=0)['Gienger SR']/ np.sum(pd.read_csv(mt_dir, sep=',', header=0)['Gienger SR'])
    mt_theta = pd.read_csv(mt_dir, sep=',', header=0)['Theta']
    mt_SL_pchip = pchip_interpolate(mt_theta, mt_SL, measured_theta)
    mt_SR_pchip = pchip_interpolate(mt_theta, mt_SR, measured_theta)

    result = least_squares(SL_SR_NLLS, x0=[1, 1], args=(mt_SL_pchip, mt_SR_pchip, measured_sample))
    best_fit = (result.x[0] * mt_SL_pchip + np.abs(result.x[1]) * mt_SR_pchip)

    save_directory = '/home/sm3/Desktop/Recent/'

    f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
    ax0[0].semilogy(measured_theta, best_fit, color='red', ls='-', label='LVMQ NLLS: ' + 'measurement = ' + str('{:.3f}'.format(result.x[0])) + ' * SL + ' + str('{:.3f}'.format(np.abs(result.x[1]))) + ' * SR \n Key Points: %SL: ' + str('{:.3f}'.format((result.x[0]/(result.x[0] + np.abs(result.x[1]))) * 100)) + ' , %SR: ' + str('{:.3f}'.format((np.abs(result.x[1])/(result.x[0] + np.abs(result.x[1]))) * 100)) + ' , SL/SR: ' + str('{:.3f}'.format(result.x[0]/np.abs(result.x[1]))))
    ax0[0].semilogy(measured_theta,  measured_sample, color='black', ls='-', label='measurement')
    ax0[0].set_title('Non-Linear Least Squares Fit')
    ax0[0].set_xlabel('\u03b8')
    ax0[0].set_ylabel('Intensity')
    ax0[0].grid(True)
    ax0[0].legend(loc=1)
    ax0[1].plot(measured_theta,  (result.fun/best_fit) * 100, color='black', ls='-', label='Measurement')
    ax0[1].set_title('Non-Linear Least Squares Percent Error')
    ax0[1].set_xlabel('\u03b8')
    ax0[1].set_ylabel('%')
    ax0[1].grid(True)
    ax0[1].legend(loc=1)
    plt.tight_layout()
    plt.savefig(save_directory + 'NLLS_PSL_' + str(element) + '.png', format='png')
    plt.savefig(save_directory + 'NLLS_PSL_' + str(element) + '.pdf', format='pdf')
    #plt.show()

    ssr = np.sum(np.square(result.fun))
    print('sum of squared residuals', ssr)
    ssr_array.append(ssr)
    sl_array.append(result.x[0])
    sr_array.append(result.x[1])
# Do the polarization combination with the Rayleigh Scattering Theory and see if it matches our measurement!!!

f1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(retardance, ssr_array, color='black', marker='o', ls='', label='Sum of Squared Residuals vs. Retardance')
ax1.set_title('Sum of Squared Residuals vs. Retardance')
ax1.set_xlabel('Retardance')
ax1.set_ylabel('Sum of Squared Residuals')
ax1.grid(True)
ax1.legend(loc=1)
plt.tight_layout()
plt.savefig(save_directory + 'SSR_v_Retardance.png', format='png')
plt.savefig(save_directory + 'SSR_v_Retardance.pdf', format='pdf')
plt.show()

f1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(retardance, sl_array, color='black', marker='o', ls='', label='SL vs. Retardance')
ax1.set_title('SL vs. Retardance')
ax1.set_xlabel('Retardance')
ax1.set_ylabel('Sum of Squared Residuals')
ax1.grid(True)
ax1.legend(loc=1)
plt.tight_layout()
plt.savefig(save_directory + 'SL_v_Retardance.png', format='png')
plt.savefig(save_directory + 'SL_v_Retardance.pdf', format='pdf')
plt.show()


f1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(retardance, sr_array, color='black', marker='o', ls='', label='SR vs. Retardance')
ax1.set_title('SR vs. Retardance')
ax1.set_xlabel('Retardance')
ax1.set_ylabel('SR')
ax1.grid(True)
ax1.legend(loc=1)
plt.tight_layout()
plt.savefig(save_directory + 'SR_v_Retardance.png', format='png')
plt.savefig(save_directory + 'SR_v_Retardance.pdf', format='pdf')
plt.show()
'''
