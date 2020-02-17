'''
Austen K. Scruggs
02-17-2020
Description: Heatmap
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.optimize import least_squares

# this tool is for finding non-idealities in the images!
'''
directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/CO2/300s/2darray/CO2_300s_0R_Average_Sat Feb 8 2020 3_57_03 PM.txt'
im = np.array(pd.read_csv(directory, sep='\t'))
print(im)

plt.pcolormesh(im, cmap='gray', vmax=4095, vmin=0)
plt.colorbar()
plt.show()


directory2 = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/CO2_100s_0R_5AVG_Average_Mon Feb 17 2020 12_27_45 PM.txt'
im2 = np.array(pd.read_csv(directory, sep='\t'))
print(im)

plt.pcolormesh(im2, cmap='gray', vmax=4095, vmin=0)
plt.colorbar()
plt.show()
'''

def SL_SR_NLLS(x, SL_theory, SR_theory, measurement):
    residuals = measurement - (x[0] * SL_theory + x[1] * SR_theory)
    return residuals


meas_dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/PSL_Analysis/0R/SD_Particle.txt'
measured_sample = pd.read_csv(meas_dir, sep=',', header=0)['Sample Intensity'] / np.sum(pd.read_csv(meas_dir, sep=',', header=0)['Sample Intensity'])
measured_columns = pd.read_csv(meas_dir, sep=',', header=0)['Sample Columns']
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
best_fit = (result.x[0] * mt_SL_pchip + result.x[1] * mt_SR_pchip)

save_directory = '/home/sm3/Desktop/Recent/'

f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].semilogy(measured_theta, best_fit, color='red', ls='-', label='LVMQ NLLS: ' + 'measurement = ' + str('{:.3f}'.format(result.x[0])) + ' * SL + ' + str('{:.3f}'.format(result.x[1])) + ' * SR \n Key Points: %SL: ' + str('{:.3f}'.format((result.x[0]/(result.x[0] + result.x[1])) * 100)) + ' , %SR: ' + str('{:.3f}'.format((result.x[1]/(result.x[0] + result.x[1])) * 100)) + ' , SL/SR: ' + str('{:.3f}'.format(result.x[0]/result.x[1])))
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
plt.savefig(save_directory + 'NLLS_PSL.png', format='png')
plt.savefig(save_directory + 'NLLS_PSL.pdf', format='pdf')
plt.show()