'''
Austen K. Scruggs
10-23-2019
Description: Looped plotting rayleigh scattering data at different polarizations
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from cycler import cycler
from matplotlib.pyplot import cm
from math import pi
from scipy.optimize import curve_fit

# Rayleigh theory
H_path = '/home/austen/Desktop/Rayleigh_Analysis/T2B/lamda_0.5/Example/PFs'
V_path = '/home/austen/Desktop/Rayleigh_Analysis/T2B/lamda_0/Example/PFs'
Theory_path = '/home/austen/Desktop/Rayleigh_Analysis/T3/Rayleigh_PF.txt'
save_directory = '/home/austen/Desktop/Rayleigh_Analysis/T2B/'
def Rayleigh_Fit_Perpendicular(theta, a, b):
    rads = (theta * pi) / 180.0
    f = a * (1 + b * (np.cos(rads)**2))
    return f

def Rayleigh_Fit_Parallel(theta, m, b):
    y = (m * theta) + b
    return y


Rayleigh_Theory_DF = pd.read_csv(Theory_path, sep=',', header=0)
CO2_PF = Rayleigh_Theory_DF['CO2 anisotropic']

#plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))
f0, ax0 = plt.subplots(1, 3, figsize=(20, 10))
f1, ax1 = plt.subplots(1, 3, figsize=(20, 10))


H_files = os.listdir(H_path)
for counter, file in enumerate(H_files):
    #color_array = ['red', 'green', 'blue']
    color_array = ['orange', 'blue', 'pink']
    df = pd.read_csv(H_path + '/' + str(file), sep=',', header=0)
    df = df.dropna(axis=0, how='any')
    pf_CO2 = df['CO2 Intensity gfit corr']
    pf_CO2 = pf_CO2[5:len(pf_CO2)-5]
    pf2_CO2 = df['CO2 Intensity'] - df['He Intensity']
    pf_N2 = df['N2 Intensity gfit corr']
    pf_N2 = pf_N2[5:len(pf_N2)-5]
    pf2_N2 = df['N2 Intensity'] - df['He Intensity']
    theta = df['CO2 Theta']
    theta = theta[5:len(theta)-5]
    theta_theory = np.arange(0.0, 180.0, 1.0)
    popt_CO2, pcov_CO2 = curve_fit(Rayleigh_Fit_Perpendicular, theta, pf_CO2)
    popt_N2, pcov_N2 = curve_fit(Rayleigh_Fit_Perpendicular, theta, pf_N2)
    ax0[0].plot(theta, pf_CO2, label='$CO_2$ \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    #ax0[0, 0].plot(theta, pf2_CO2, label='$CO_2$ Riemann \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    ax0[0].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_CO2), ls='--', color=color_array[counter], label='Fit: ' + str('{:.3f}'.format(popt_CO2[0])) + '(1 + ' + str('{:.3f}'.format(popt_CO2[1])) + '$cos^2(\u03b8)$)\n' + str(file))

    ax0[2].plot(theta, pf_CO2, label='$CO_2$ \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    #ax0[0, 2].plot(theta, pf2_CO2, label='$CO_2$ Riemann \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    ax0[2].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_CO2), ls='--', color=color_array[counter], label='Fit: ' + str('{:.3f}'.format(popt_CO2[0])) + '(1 + ' + str('{:.3f}'.format(popt_CO2[1])) + '$cos^2(\u03b8)$)\n' + str(file))

    ax1[0].plot(theta[15:-1], pf_N2[15:-1], label='$N_2$ \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    #ax1[1, 0].plot(theta, pf2_N2, label='$N_2$ Riemann \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    ax1[0].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_N2), ls='--', color=color_array[counter], label='Fit: ' + str('{:.3f}'.format(popt_N2[0])) + '(1 + ' + str('{:.3f}'.format(popt_N2[1])) + '$cos^2(\u03b8)$)\n' + str(file))

    ax1[2].plot(theta[15:-1], pf_N2[15:-1], label='$N_2$ \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    #ax1[1, 2].plot(theta, pf2_N2, label='$N_2$ Riemann \u03bb = 0.5\n' + str(file), ls='-', color=color_array[counter])
    ax1[2].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_N2), ls='--', color=color_array[counter], label='Fit: ' + str('{:.3f}'.format(popt_N2[0])) + '(1 + ' + str('{:.3f}'.format(popt_N2[1])) + '$cos^2(\u03b8)$)\n' + str(file))

V_files = os.listdir(V_path)
for counter, file in enumerate(V_files):
    color_array = ['red', 'green', 'blue']
    df = pd.read_csv(V_path + '/' + str(file), sep=',', header=0)
    df = df.dropna(axis=0, how='any')
    pf_CO2 = df['CO2 Intensity gfit corr']
    pf_CO2 = pf_CO2[5:len(pf_CO2)-5]
    pf2_CO2 = df['CO2 Intensity'] - df['He Intensity']
    pf_N2 = df['N2 Intensity gfit corr']
    pf_N2 = pf_N2[5:len(pf_N2)-5]
    pf2_N2 = df['N2 Intensity'] - df['He Intensity']
    theta = df['CO2 Theta']
    theta = theta[5:len(theta)-5]
    theta_theory = np.arange(0.0, 180.0, 1.0)
    popt_CO2, pcov_CO2 = curve_fit(Rayleigh_Fit_Perpendicular, theta, pf_CO2)
    popt_N2, pcov_N2 = curve_fit(Rayleigh_Fit_Perpendicular, theta, pf_N2)
    ax0[1].plot(theta, pf_CO2, label='$CO_2$ \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[0, 1].plot(theta, pf2_CO2, label='$CO_2$ Riemann \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[0, 1].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_CO2), ls='--', label='$cos^2(\u03b8)$ fit: ' + str('{:.3f}'.format(popt_CO2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_CO2[1] + str(file))))
    #ax0[0, 1].plot(theta_theory, Rayleigh_Fit_Parallel(theta_theory, *popt_CO2), ls='--', label='$cos^2(\u03b8)$ fit: y = ' + str('{:.3f}'.format(popt_CO2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_CO2[1] + str(file))))

    ax0[2].plot(theta, pf_CO2, label='$CO_2$ \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[0, 2].plot(theta, pf2_CO2, label='$CO_2$ Riemann \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[0, 2].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_CO2), ls='--', label='linear fit: y = ' + str('{:.3f}'.format(popt_CO2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_CO2[1] + str(file))))
    #ax0[0, 2].plot(theta_theory, Rayleigh_Fit_Parallel(theta_theory, *popt_CO2), ls='--', label='$cos^2(\u03b8)$ fit: y = ' + str('{:.3f}'.format(popt_CO2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_CO2[1] + str(file))))

    ax1[1].plot(theta[15:-1], pf_N2[15:-1], label='$N_2$ \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[1, 1].plot(theta, pf2_N2, label='$N_2$ Riemann \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[1, 1].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_N2), ls='--', label='linear fit: y = ' + str('{:.3f}'.format(popt_N2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_N2[1] + str(file))))
    #ax0[1, 1].plot(theta_theory, Rayleigh_Fit_Parallel(theta_theory, *popt_N2), ls='--', label='$cos^2(\u03b8)$ fit: y = ' + str('{:.3f}'.format(popt_N2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_N2[1]) + str(file)))

    ax1[2].plot(theta[15:-1], pf_N2[15:-1], label='$N_2$ \u03bb = 0.0\n' + str(file), ls='-', color=color_array[counter])
    #ax0[1, 2].plot(theta, pf2_N2, label='$N_2$ Riemann \u03bb = 0.0\n' + str(file), ls='-')
    #ax0[1, 2].plot(theta_theory, Rayleigh_Fit_Perpendicular(theta_theory, *popt_N2), ls='--', label='linear fit: y = ' + str('{:.3f}'.format(popt_N2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_N2[1] + str(file))))
    #ax0[1, 2].plot(theta_theory, Rayleigh_Fit_Parallel(theta_theory, *popt_N2), ls='--', label='$cos^2(\u03b8)$ fit: y = ' + str('{:.3f}'.format(popt_N2[0])) + '\u03b8 + ' + str('{:.3f}'.format(popt_N2[1] + str(file))))

a = 20
b = 14
c = 14
ax0[0].set_title('$CO_2$ Angular Scattering Perpendicular Polarization \n(\u03bb = 0.5)', fontsize=a)
ax0[0].set_ylabel('Intensity', fontsize=b)
ax0[0].set_xlabel('\u03b8', fontsize=b)
ax0[0].grid(True)
ax0[0].legend(loc=1, fontsize=c)
ax0[1].set_title('$CO_2$ Angular Scattering Parallel Polarization \n(\u03bb = 0.0)', fontsize=a)
ax0[1].set_ylabel('Intensity', fontsize=b)
ax0[1].set_xlabel('\u03b8', fontsize=b)
ax0[1].grid(True)
ax0[1].legend(loc=1, fontsize=c)
ax0[2].set_title('$CO_2$ Scattering Parallel & \nPerpendicular Polarizations', fontsize=a)
ax0[2].set_ylabel('Intensity', fontsize=b)
ax0[2].set_xlabel('\u03b8', fontsize=b)
ax0[2].grid(True)
ax0[2].legend(loc=1, fontsize=c)
ax0[2].set_ylim(0, 40000)
f0.tight_layout()
f0.savefig(save_directory + 'CO2_RAYFIG.png', format='png')
f0.show()

ax1[0].set_title('$N_2$ Angular Scattering Perpendicular Polarization \n(\u03bb = 0.5)', fontsize=a)
ax1[0].set_ylabel('Intensity')
ax1[0].set_xlabel('\u03b8')
ax1[0].grid(True)
ax1[0].legend(loc=1, fontsize=c)
ax1[1].set_title('$N_2$ Angular Scattering Parallel Polarization \n(\u03bb = 0.0)', fontsize=a)
ax1[1].set_ylabel('Intensity', fontsize=b)
ax1[1].set_xlabel('\u03b8', fontsize=b)
ax1[1].grid(True)
ax1[1].legend(loc=1, fontsize=c)
ax1[2].set_title('$N_2$ Scattering Parallel & \nPerpendicular Polarizations', fontsize=a)
ax1[2].set_ylabel('Intensity', fontsize=b)
ax1[2].set_xlabel('\u03b8', fontsize=b)
ax1[2].grid(True)
ax1[2].legend(loc=1, fontsize=c)
f1.tight_layout()
f1.savefig(save_directory + 'N2_RAYFIG.png', format='png')
f1.show()

