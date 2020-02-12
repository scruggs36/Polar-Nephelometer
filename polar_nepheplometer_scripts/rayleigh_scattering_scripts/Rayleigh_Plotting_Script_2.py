'''
Austen K. Scruggs
10-23-2019
Description: Looped plotting rayleigh scattering data at different polarizations, updated
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from cycler import cycler
from matplotlib.pyplot import cm
from math import pi
from scipy.optimize import curve_fit
from scipy.interpolate import pchip_interpolate

# Rayleigh theory
#dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/Rayleigh_Analysis/'
file_directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/Rayleigh_Analysis/sigma_pixels_5/'
H_path = file_directory + '/0.5R/SD_Rayleigh.txt'
V_path = file_directory + '/0R/SD_Rayleigh.txt'
Theory_path = dir + 'T3/Rayleigh_PF.txt'
save_directory = '/home/sm3/Desktop/Recent/'


def Cosine_Squared_Fit(theta, a, b):
    rads = (theta * pi) / 180.0
    f = a * (1 + b * (np.cos(rads)**2))
    return f


Rayleigh_Theory_DF = pd.read_csv(Theory_path, sep=',', header=0)
CO2_PF = Rayleigh_Theory_DF['CO2 anisotropic']


q = 20
r = 14
s = 14
f0, ax0 = plt.subplots(2, 1, figsize=(20, 10))
#color_array = ['red', 'green', 'blue']
df = pd.read_csv(H_path, sep=',', header=0)
df = df.dropna(axis=1, how='any')
#print(df)
#pf_CO2_lamda05 = np.array(df['CO2 Intensity']) / np.linalg.norm(np.array(df['CO2 Intensity']))
#pf2_CO2_lamda05 = np.array(df['CO2 Intensity gfit corr']) / np.linalg.norm(np.array(df['CO2 Intensity gfit corr']))

pf_CO2_lamda05 = np.array(df['CO2 Intensity'])
pf2_CO2_lamda05 = np.array(df['CO2 Intensity gfit corr'])

columns_CO2_lamda05 = np.array(df['CO2 Columns'])
#print(columns_CO2)
#pf_N2 = df['N2 Intensity gfit corr']
#pf_N2 = pf_N2
#pf2_N2 = df['N2 Intensity']
slope = 0.2049
intercept = -2.7594
theta = [(slope * x) + intercept for x in columns_CO2_lamda05]
print(len(theta))
print(len(pf_CO2_lamda05))
#theta = theta
theta_theory = np.arange(0.0, 180.0, 1.0)
popt_CO2_lamda05, pcov_CO2_lamda05 = curve_fit(Cosine_Squared_Fit, theta, pf_CO2_lamda05)
#popt_N2, pcov_N2 = curve_fit(Rayleigh_Fit_Perpendicular, theta, pf_N2)
ax0[0].plot(theta, pf_CO2_lamda05, label='$CO_2$ \u03bb = 0.5\n', ls='-', color='black')
ax0[0].plot(theta, pf2_CO2_lamda05, label='$CO_2$ \u03bb = 0.5\n', ls='-', color='green')
ax0[0].plot(theta_theory, Cosine_Squared_Fit(theta_theory, *popt_CO2_lamda05), ls='--', color='red', label='Fit: ' + str('{:.3f}'.format(popt_CO2_lamda05[0])) + '(1 + ' + str('{:.3f}'.format(popt_CO2_lamda05[1])) + '$cos^2(\u03b8)$)\n')
#ax1[1].plot(theta, pf_N2, label='$N_2$ \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
#ax1[1].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_N2), ls='--', color=color_array[counter], label='Fit: ' + str('{:.3f}'.format(popt_N2[0])) + '(1 + ' + str('{:.3f}'.format(popt_N2[1])) + '$cos^2(\u03b8)$)\n' + str(file))
ax0[0].set_title('$CO_2$ Angular Scattering (\u03bb = 0.5)', fontsize=q)
ax0[0].set_ylabel('Intensity', fontsize=r)
ax0[0].set_xlabel('\u03b8', fontsize=r)
ax0[0].grid(True)
ax0[0].legend(loc=1, fontsize=s)
pf_fit_lamda05 = Cosine_Squared_Fit(theta_theory, *popt_CO2_lamda05)
pf_fit_lamda05_pchip = pchip_interpolate(theta_theory, pf_fit_lamda05, theta)
residual_lamda05 = (pf_CO2_lamda05 - pf_fit_lamda05_pchip)**2
ax0[1].plot(theta, residual_lamda05, color='black', ls='-', label='residuals')
ax0[1].set_title('$CO_2$ Residual as a Function of \u03b8 (\u03bb = 0.5)', fontsize=q)
ax0[1].set_ylabel('Intensity', fontsize=r)
ax0[1].set_xlabel('\u03b8', fontsize=r)
ax0[1].grid(True)
ax0[1].legend(loc=1, fontsize=s)
#ax0[1].set_title('$N_2$ Angular Scattering (\u03bb = 0.5)', fontsize=q)
#ax0[1].set_ylabel('Intensity', fontsize=r)
#ax0[1].set_xlabel('\u03b8', fontsize=r)
#ax0[1].grid(True)
#ax0[1].legend(loc=1, fontsize=s)
f0.tight_layout()
f0.savefig(save_directory + 'CO2_lamda0.5_RAYFIG.png', format='png')
f0.show()




f1, ax1 = plt.subplots(2, 1, figsize=(20, 10))
df = pd.read_csv(V_path, sep=',', header=0)
df = df.dropna(axis=1, how='any')
#pf_CO2_lamda0 = np.array(df['CO2 Intensity']) / np.linalg.norm(np.array(df['CO2 Intensity']))
#pf2_CO2_lamda0 = np.array(df['CO2 Intensity gfit corr']) / np.linalg.norm(np.array(df['CO2 Intensity gfit corr']))

pf_CO2_lamda0 = np.array(df['CO2 Intensity'])
pf2_CO2_lamda0 = np.array(df['CO2 Intensity gfit corr'])

#pf_N2 = df['N2 Intensity gfit corr']
#pf_N2 = pf_N2
#pf2_N2 = df['N2 Intensity'] - df['He Intensity']
slope = 0.2049
intercept = -2.7594
theta = [(slope * x) + intercept for x in np.array(df['CO2 Columns'])]
print(len(theta))
print(len(pf_CO2_lamda0))
theta_theory = np.arange(0.0, 180.0, 1.0)
popt_CO2_lamda0, pcov_CO2_lamda0 = curve_fit(Cosine_Squared_Fit, theta, pf_CO2_lamda0)
#popt_N2, pcov_N2 = curve_fit(Rayleigh_Fit_Perpendicular, theta, pf_N2)
ax1[0].plot(theta, pf_CO2_lamda0, label='$CO_2$ \u03bb = 0.0\n', ls='-', color='black')
ax1[0].plot(theta, pf2_CO2_lamda0, label='$CO_2$ \u03bb = 0.0\n', ls='-', color='green')
ax1[0].plot(theta_theory, Cosine_Squared_Fit(theta_theory, *popt_CO2_lamda0), ls='--', color='red', label='Fit: ' + str('{:.3f}'.format(popt_CO2_lamda0[0])) + '(1 + ' + str('{:.3f}'.format(popt_CO2_lamda0[1])) + '$cos^2(\u03b8)$)\n')
ax1[0].set_title('$CO_2$ Angular Scattering (\u03bb = 0.0)', fontsize=q)
ax1[0].set_ylabel('Intensity', fontsize=r)
ax1[0].set_xlabel('\u03b8', fontsize=r)
ax1[0].grid(True)
ax1[0].legend(loc=1, fontsize=s)
pf_fit_lamda0 = Cosine_Squared_Fit(theta_theory, *popt_CO2_lamda0)
pf_fit_lamda0_pchip = pchip_interpolate(theta_theory, pf_fit_lamda0, theta)
residual_lamda0 = (pf_CO2_lamda0 - pf_fit_lamda0_pchip)**2
ax1[1].plot(theta, residual_lamda0, color='black', ls='-', label='residuals')
ax1[1].set_title('$CO_2$ Residual as a Function of \u03b8 (\u03bb = 0.0)', fontsize=q)
ax1[1].set_ylabel('Intensity', fontsize=r)
ax1[1].set_xlabel('\u03b8', fontsize=r)
ax1[1].grid(True)
ax1[1].legend(loc=1, fontsize=s)
#ax1[1].set_title('$N_2$ Angular Scattering (\u03bb = 0.0)', fontsize=q)
#ax1[1].set_ylabel('Intensity', fontsize=r)
#ax1[1].set_xlabel('\u03b8', fontsize=r)
#ax1[1].grid(True)
#ax1[1].legend(loc=1, fontsize=s)
f1.tight_layout()
f1.savefig(save_directory + 'CO2_lamda0_RAYFIG.png', format='png')
f1.show()

f2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(theta, pf2_CO2_lamda0, color='red', ls='-', label='Meas. \u03bb = 0.0')
ax2.plot(theta, pf2_CO2_lamda05, color='blue', ls='-', label='Meas. \u03bb = 0.5')
ax2.set_title('$CO_2$ Phase Functions (\u03bb = 0.0 & 0.5)', fontsize=q)
ax2.set_ylabel('Intensity', fontsize=r)
ax2.set_xlabel('\u03b8', fontsize=r)
ax2.grid(True)
ax2.legend(loc=1, fontsize=s)
f2.tight_layout()
f2.savefig(save_directory + 'CO2_lamda0&05.png', format='png')
f2.show()

delta = [(x - np.amin(pf_CO2_lamda0)) * (1/40) for x in pf_CO2_lamda0]
pf2_CO2_lamda0_fixed = [x - (x - np.amin(pf_CO2_lamda0)) for x in pf_CO2_lamda0]

# import 900nm PSL data

PSL900_lamda0_dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/11.21.19/2019-11-21_Analysis/PSL/900/1s/0lamda/SD_Particle.txt'
PSL900_lamda05_dir = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/11.21.19/2019-11-21_Analysis/PSL/900/1s/0.5lamda/SD_Particle.txt'

PSL900_lamda0_data = pd.read_csv(PSL900_lamda0_dir, sep=',', header=0)
PSL900_lamda05_data = pd.read_csv(PSL900_lamda05_dir, sep=',', header=0)

PSL_theta = [(slope * x) + intercept for x in PSL900_lamda0_data['Sample Columns']]
PSL900_lamda0_pf = PSL900_lamda0_data['Sample Intensity']
PSL900_lamda05_pf = PSL900_lamda05_data['Sample Intensity']

PSL900_lamda0_pf_pchip = pchip_interpolate(PSL_theta, PSL900_lamda0_pf, theta)
PSL900_lamda05_pf_pchip = pchip_interpolate(PSL_theta, PSL900_lamda05_pf, theta)

f3, ax3 = plt.subplots(2, 2, figsize=(12, 6))
ax3[0, 0].plot(theta, pf2_CO2_lamda0, color='black', label='$CO_2$ fixed \u03bb = 0.0')
ax3[0, 0].plot(theta, pf2_CO2_lamda0_fixed, color='red', label='$CO_2$ fixed \u03bb = 0.0')
ax3[0, 0].set_title('$CO_2$ Phase Function Fixed (\u03bb = 0.0)')
ax3[0, 0].set_ylabel('Intensity')
ax3[0, 0].set_xlabel('\u03b8')
ax3[0, 0].grid(True)
ax3[0, 0].legend(loc=1)
ax3[0, 1].semilogy(theta, PSL900_lamda0_pf_pchip, color='black', ls='-', label='PSL 900nm OG')
ax3[0, 1].semilogy(theta, PSL900_lamda0_pf_pchip - delta, color='red', ls='-', label='PSL 900nm "fixed"')
ax3[0, 1].set_title('900nm PSL Phase Function (\u03bb = 0.0)')
ax3[0, 1].set_ylabel('Intensity')
ax3[0, 1].set_xlabel('\u03b8')
ax3[0, 1].grid(True)
ax3[0, 1].legend(loc=1)
ax3[1, 0].semilogy(theta, PSL900_lamda05_pf_pchip, color='black', ls='-', label='PSL 900nm OG')
ax3[1, 0].semilogy(theta, PSL900_lamda05_pf_pchip - delta, color='red', ls='-', label='PSL 900nm "fixed"')
ax3[1, 0].set_title('900nm PSL Phase Function (\u03bb = 0.5)')
ax3[1, 0].set_ylabel('Intensity')
ax3[1, 0].set_xlabel('\u03b8')
ax3[1, 0].grid(True)
ax3[1, 0].legend(loc=1)


f3.tight_layout()
f3.savefig(save_directory + 'CO2_lamda0_fixed.png', format='png')
f3.show()

