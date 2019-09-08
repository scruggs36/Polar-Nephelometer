'''
Austen K. Sruggs
date created:  09-04-2019
Description: script solves for local maxima and minima in
Mie theory and measured scattering diagrams and finds their
indicesThen plots the Mie theory angle of the local maxima
and minima as a function of the profile number, does a pixel to angle calibration
and then attempts to apply a lens correction to the scattering anlge axis by using
the rayleigh scattering data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import PyMieScatt as ps
from matplotlib import gridspec
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.signal import savgol_filter, argrelmax, argrelmin
from scipy.optimize import curve_fit
from math import sqrt, log, pi
from matplotlib.ticker import MultipleLocator

# import N2 Rayleigh scattering data
Save_Directory = '/home/sm3/Desktop/'



# refractive index for PSL calculated for each group (literally until line 240 skip!)
# wavelength in centimeters
w_c_array = [350E-7, 405E-7, 532E-7, 663E-7]

# wavelength in microns
w_u_array = [.350, .405, .532, .663]

# wavelength in nanometers
w_n_array = [350, 405, 532, 663]

# wavelength in angstroms
w_a_array = [3050, 4050, 5320, 6630]

# UV-Visible spectrum in angstroms
uv_visible_spectrum_centimeters = np.arange(300E-7, 1050E-7, 10E-7)
#print(len(uv_visible_spectrum_centimeters))

# UV-Visible spectrum in microns
uv_visible_spectrum_microns = np.arange(.300, 1.060, .010)

# UV-Visible spectrum in nanometers
uv_visible_spectrum_nanometers = np.arange(300, 1060, 10)
#print(len(uv_visible_spectrum_nanometers))

# UV-Visible spectrum in angstroms
uv_visible_spectrum_angstroms = np.arange(3000, 10600, 100)



# complex refractive index, Cauchy parameters for PSL Matheson & Sanderson 1952, wavelength in microns from Greenslade
A0 = 1.5663
B0 = 0.00785
C0 = 0.000334

# complex refractive index, Cauchy parameters for PSL Bateman 1959 wavelength in centimeters
A1 = 1.5683
B1 = 10.087E-11

# defining cauchy equation functions
def cauchy_2term(wav, A_2term, B_2term):
    return A_2term + (B_2term / wav ** 2)


def cauchy_4term(wav, A_4term, B_4term, C_4term):
    return A_4term + (B_4term / wav ** 2) + (C_4term / (wav ** 4))


def cauchy_6term(wav, A_1_6term, A_2_6term, A_3_6term, A_4_6term, A_5_6term, A_6_6term):
    return np.sqrt(A_1_6term + (A_2_6term * wav ** 2) + (A_3_6term / wav ** 2) + (A_4_6term / (wav ** 4)) + (A_5_6term / (wav ** 6)) + (A_6_6term / (wav ** 8)))


# fit Nikalov data for cauchy coefficients, wavelength in microns
w_Nikalov_microns = np.array([0.436, 0.486, 0.546, 0.588, 0.633, 0.656, 0.703, 0.752, 0.804, 0.833, 0.879, 1.052])
w_Nikalov_nanometers = w_Nikalov_microns * 1000
n_RI_Nikalov = np.array([1.617, 1.606, 1.596, 1.592, 1.587, 1.586, 1.582, 1.579, 1.578, 1.577, 1.576, 1.572])
#print(w_Nikalov_microns)
popt_Nikalov, pcov_Nikalov = curve_fit(cauchy_6term, w_Nikalov_microns, n_RI_Nikalov, p0=[2.44675093, -1.011623E-3, 2.840749E-2, -3.761631E-4, 8.193491E-5, 9.055861E-4])
#print(popt_Nikalov)
# complex refractive index, Cauchy parameters for PSL Nikalov et al 2000, wavelength in microns
A2 = popt_Nikalov[0]
B2 = popt_Nikalov[1]
C2 = popt_Nikalov[2]
D2 = popt_Nikalov[3]
E2 = popt_Nikalov[4]
F2 = popt_Nikalov[5]
# complex refractive index, Cauchy parameters for PSL Ma et al 2003, wavelength in microns
A3 = 1.5725
B3 = 0.003108
C3 = 0.00034779

# complex refractive index, Cauchy parameters for PSL Sultanova et al 2003 wavelength in microns
A4 = 2.44675093
B4 = -1.011623E-3
C4 = 2.840749E-2
D4 = -3.761631E-4
E4 = 8.193491E-5
F4 = 2.186304E-5

# complex refractive index, Cauchy parameters for PSL Kasarova et al 2006 wavelength in microns
A5 = 2.610025
B5 = -6.143673E-2
C5 = -1.312267E-1
D5 = 6.865432E-2
E5 = -1.295968E-2
F5 = 9.055861E-4

# complex refractive index, Cauchy parameters for PSL Miles et al 2010 wavelength in microns
A6 = 1.5663
B6 = 0.00785
C6 = 0.000334

# complex refractive index, Cauchy parameters for PSL Jones et al 2013 wavelength in nanometers
A7 = 1.5718
B7 = 8412
C7 = 2.35E8

# complex refractive index, Cauchy Greenslade 2017, wavelength in microns
A8 = 1.53811
B8 = 0.004316
C8 = 0.000945

# complex refractive index, Cauchy Gienger 2017, wavelength in microns, uses Sellmeier equation, wavelength in nanometers
B9 = 1.4432
wav9 = 142.1

# n of refractive index for PSL as cauchy equation at a specific wavelength

n_cauchy0 = np.array([A0 + (B0 / w_u ** 2) + (C0 / (w_u ** 4)) for w_u in w_u_array])
n_cauchy1 = np.array([A1 + (B1 / w_c ** 2) for w_c in w_c_array])
n_cauchy2 = np.array([np.sqrt(A2 + (B2 * w_u ** 2) + (C2 / w_u ** 2) + (D2 / (w_u ** 4)) + (E2 / (w_u ** 6)) + (F2 / (w_u ** 8))) for w_u in w_u_array])
n_cauchy3 = np.array([A3 + (B3 / w_u ** 2) + (C3 / (w_u ** 4)) for w_u in w_u_array])
n_cauchy4 = np.array([np.sqrt(A4 + (B4 * w_u ** 2) + (C4 / w_u ** 2) + (D4 / (w_u ** 4)) + (E4 / (w_u ** 6)) + (F4 / (w_u ** 8))) for w_u in w_u_array])
n_cauchy5 = np.array([np.sqrt(A5 + (B5 * w_u ** 2) + (C5 / w_u ** 2) + (D5 / (w_u ** 4)) + (E5 / (w_u ** 6)) + (F5 / (w_u ** 8))) for w_u in w_u_array])
n_cauchy6 = np.array([A6 + (B6 / w_u ** 2) + (C6 / (w_u ** 4)) for w_u in w_u_array])
n_cauchy7 = np.array([A7 + (B7 / w_n ** 2) + (C7 / (w_n ** 4)) for w_n in w_n_array])
n_cauchy8 = np.array([A8 + (B8 / w_u ** 2) + (C8 / (w_u ** 4)) for w_u in w_u_array])
n_sellmeier9 = np.array([np.sqrt(1 + ((B9 * w_n ** 2)/(w_n ** 2 - wav9 ** 2))) for w_n in w_n_array])

# basic statistics
n_all_groups = [n_cauchy0, n_cauchy1, n_cauchy2, n_cauchy3, n_cauchy4, n_cauchy5, n_cauchy6, n_cauchy7, n_cauchy8, n_sellmeier9]
n_groups_red = [n_all_groups[x][3] for x in range(3)]
n_groups_green = [n_all_groups[x][2] for x in range(3)]
n_groups_blue = [n_all_groups[x][1] for x in range(3)]
n_groups_uv = [n_all_groups[x][0] for x in range(3)]
n_mean_red = np.mean(n_groups_red)
n_mean_green = np.mean(n_groups_green)
n_mean_blue = np.mean(n_groups_blue)
n_mean_uv = np.mean(n_groups_uv)
n_percentiles_red = np.percentile(n_groups_red, [0, 25, 50, 75, 100])
n_percentiles_green = np.percentile(n_groups_green, [0, 25, 50, 75, 100])
n_percentiles_blue = np.percentile(n_groups_blue, [0, 25, 50, 75, 100])
n_percentiles_uv = np.percentile(n_groups_uv, [0, 25, 50, 75, 100])


# n (real refractive index) over many wavelengths
n_cauchy_v_wav0 = [A0 + (B0 / element ** 2) + (C0 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav1 = [A1 + (B1 / element ** 2) for element in uv_visible_spectrum_centimeters]
n_cauchy_v_wav2 = [np.sqrt(A2 + (B2 * element ** 2) + (C2 / element ** 2) + (D2 / (element ** 4)) + (E2 / (element ** 6)) + (F2 / (element ** 8))) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav3 = [A3 + (B3 / element ** 2) + (C3 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav4 = [np.sqrt(A4 + (B4 * element ** 2) + (C4 / element ** 2) + (D4 / (element ** 4)) + (E4 / (element ** 6)) + (F4 / (element ** 8))) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav5 = [np.sqrt(A5 + (B5 * element ** 2) + (C5 / element ** 2) + (D5 / (element ** 4)) + (E5 / (element ** 6)) + (F5 / (element ** 8))) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav6 = [A6 + (B6 / element ** 2) + (C6 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav7 = [A7 + (B7 / element ** 2) + (C7 / (element ** 4)) for element in uv_visible_spectrum_nanometers]
n_cauchy_v_wav8 = [A8 + (B8 / element ** 2) + (C8 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_sellmeier_v_wav9 = [np.sqrt(1 + ((B9 * element ** 2)/(element ** 2 - wav9 ** 2))) for element in uv_visible_spectrum_nanometers]

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 7))
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav0, label='Matheson et. al. 1952')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav1, label='Bateman et. al. 1959')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav2, label='Nikalov et. al. 2000 Fit')
ax2[0].plot(w_Nikalov_nanometers, n_RI_Nikalov, marker='o', ms=3, ls='', label='Nikalov et. al. 2000 Meas.')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav3, label='Ma et. al. 2003')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav4, label='Sultanova et. al. 2003')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav5, label='Kasarova et. al. 2006')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav6, label='Miles et. al. 2010')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav7, label='Jones et. al. 2013')
ax2[0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav8, label='Greenslade et. al. 2017')
ax2[0].plot(uv_visible_spectrum_nanometers, n_sellmeier_v_wav9, label='Gienger et. al. 2017')
ax2[0].set_title('Polystyrene Latex Sphere Real Refractive Index (n) \n as a Function of Wavelength')
ax2[0].set_xlabel('Wavelength (nm)')
ax2[0].set_ylabel('n')
ax2[0].grid(True)
ax2[0].legend(loc=1)
ax2[1].plot([350, 405, 532, 663], n_cauchy0, marker='o', ms=3, ls='', label='Matheson et. al. 1952')
ax2[1].plot([350, 405, 532, 663], n_cauchy1, marker='o', ms=3, ls='', label='Bateman et. al. 1959')
ax2[1].plot([350, 405, 532, 663], n_cauchy2, marker='o', ms=3, ls='', label='Nikalov et. al. 2000')
ax2[1].plot([350, 405, 532, 663], n_cauchy3, marker='o', ms=3, ls='', label='Ma et. al. 2003')
ax2[1].plot([350, 405, 532, 663], n_cauchy4, marker='o', ms=3, ls='', label='Sultanova et. al. 2003')
ax2[1].plot([350, 405, 532, 663], n_cauchy5, marker='o', ms=3, ls='', label='Kasarova et. al. 2006')
ax2[1].plot([350, 405, 532, 663], n_cauchy6, marker='o', ms=3, ls='', label='Miles et. al. 2010')
ax2[1].plot([350, 405, 532, 663], n_cauchy7, marker='o', ms=3, ls='', label='Jones et. al. 2013')
ax2[1].plot([350, 405, 532, 663], n_cauchy8, marker='o', ms=3, ls='', label='Greenslade et. al. 2017')
ax2[1].plot([350, 405, 532, 663], n_sellmeier9, marker='o', ms=3, ls='', label='Gienger et. al. 2017')
ax2[1].set_title('All the Groups Values at 663nm')
ax2[1].set_ylabel('n')
ax2[1].grid(True)
ax2[1].legend(loc=1)
ax2[1].boxplot([n_groups_uv, n_groups_blue, n_groups_green, n_groups_red], positions=[350, 405, 532, 663], widths=(50, 50, 50, 50))
ax2[1].set_xlim(300, 700)
ax2[1].set_xlabel('Wavelength (nm)')
ax2[1].set_title('Box Plot Statistics of All the Groups Values\n at Wavelengths 350nm, 405nm, 532nm, and 663nm')
plt.tight_layout()
plt.savefig(Save_Directory + 'Cauchy.pdf', format='pdf')
plt.savefig(Save_Directory + 'Cauchy.png', format='png')
plt.show()
#plt.close()

# k, imaginary part of RI
k0 = .000j
k1 = .000j
k2 = .000j
k3 = .000j
k4 = .000j
k5 = .000j
k6 = .000j
k7 = .000j
k8 = .000j
k9 = .000j

# create RI
print('Wavelengths (nanometers): ' + str(w_n_array))
m0 = n_cauchy0 + k0
print('Matheson 1952 RI: ', m0)
m1 = n_cauchy1 + k1
print('Bateman 1959 RI: ', m1)
m2 = n_cauchy2 + k2
print('Nikalov 2000 RI: ',m2)
m3 = n_cauchy3 + k3
print('Ma 2003 RI: ', m3)
m4 = n_cauchy4 + k4
print('Sultanova 2003 RI: ', m4)
m5 = n_cauchy5 + k5
print('Kasarova 2006 RI: ', m5)
m6 = n_cauchy6 + k6
print('Miles 2010 RI: ', m6)
m7 = n_cauchy7 + k7
print('Jones 2013 RI: ', m7)
m8 = n_cauchy8 + k8
print('Greenslade 2017 RI: ', m8)
m9 = n_sellmeier9 + k9
print('Gienger 2017 RI: ', m9)

# total particle concentration
concentration = 1000
# geometric standard deviation
sigma_g = 1.05
# Particle diameter, geometric mean of the particle diameter
d= 903
# wavelength
w_n = 663
# CRI
m_array = [m0[2], m1[2], m2[2], m3[2], m4[2], m5[2], m6[2], m7[2], m8[2], m9[2]]
groups = ['Matheson 1952', 'Bateman 1959', 'Nikalov 2000', 'Ma 2003', 'Sultanova 2003', 'Kasarova 2006', 'Miles 2010', 'Jones 2013', 'Greenslade 2017', 'Gienger 2017']
# size array
size_array = np.arange(700, 1110, 10)

# log normal distribution function
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
# create log normal distribution values in array
log_dist = np.array([LogNormal(element, d, sigma_g, concentration) for element in size_array])

pf_average = []
pf_2darray = []
mie_data = []

# import mie data and format into intensities, angles, and profile numbers into separate arrays
for counter_0, m_i in enumerate(m_array):
    group = groups[counter_0]
    for counter_1, element_1 in enumerate(size_array):
        theta_mie, SL, SR, SU = ps.ScatteringFunction(m_i, w_n, element_1, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        pf_2darray.append(SL)
    pf_average = np.average(pf_2darray, axis=0, weights=log_dist)
    mie_data.append([group, m_i, pf_average])
    pf_2darray = []
# calculate weighted average of phase function data based on weights determined from the lognormal distribution
mie_df = np.array(mie_data)
max_min_array = []
for counter, element in enumerate(m_array):
    maximum = np.argmax(mie_data[counter][2], axis=0)
    minimum = np.argmin(mie_data[counter][2], axis=0)
    local_max = argrelmax(mie_data[counter][2], axis=0)
    local_min = argrelmin(mie_data[counter][2], axis=0)
    max_min_idx = np.sort(np.concatenate((local_max, local_min), axis=None))
    max_min_array.append(max_min_idx)
    max_min_idx = []




Mie_PF_Matheson = mie_data[0][2]
Mie_PF_Bateman = mie_data[1][2]
Mie_PF_Nikalov = mie_data[2][2]
Mie_PF_Ma = mie_data[3][2]
Mie_PF_Sultanova = mie_data[4][2]
Mie_PF_Kasarova = mie_data[5][2]
Mie_PF_Miles = mie_data[6][2]
Mie_PF_Jones = mie_data[7][2]
Mie_PF_Greenslade = mie_data[8][2]
Mie_PF_Gienger = mie_data[9][2]




theta_max_min_avg = np.mean([[theta_mie[x] for x in max_min_array[0]], [theta_mie[x] for x in max_min_array[1]], [theta_mie[x] for x in max_min_array[2]], [theta_mie[x] for x in max_min_array[3]], [theta_mie[x] for x in max_min_array[4]], [theta_mie[x] for x in max_min_array[5]], [theta_mie[x] for x in max_min_array[6]], [theta_mie[x] for x in max_min_array[7]], [theta_mie[x] for x in max_min_array[8]], [theta_mie[x] for x in max_min_array[9]]], axis=0)
pf_max_min_avg = np.mean([[Mie_PF_Matheson[x] for x in max_min_array[0]], [Mie_PF_Bateman[x] for x in max_min_array[1]], [Mie_PF_Nikalov[x] for x in max_min_array[2]], [Mie_PF_Ma[x] for x in max_min_array[3]], [Mie_PF_Sultanova[x] for x in max_min_array[4]], [Mie_PF_Kasarova[x] for x in max_min_array[5]], [Mie_PF_Miles[x] for x in max_min_array[6]], [Mie_PF_Jones[x] for x in max_min_array[7]], [Mie_PF_Greenslade[x] for x in max_min_array[8]], [Mie_PF_Gienger[x] for x in max_min_array[9]]], axis=0)
theta_max_min_std = np.std([[theta_mie[x] for x in max_min_array[0]], [theta_mie[x] for x in max_min_array[1]], [theta_mie[x] for x in max_min_array[2]], [theta_mie[x] for x in max_min_array[3]], [theta_mie[x] for x in max_min_array[4]], [theta_mie[x] for x in max_min_array[5]], [theta_mie[x] for x in max_min_array[6]], [theta_mie[x] for x in max_min_array[7]], [theta_mie[x] for x in max_min_array[8]], [theta_mie[x] for x in max_min_array[9]]], axis=0)
pf_max_min_std = np.std([[Mie_PF_Matheson[x] for x in max_min_array[0]], [Mie_PF_Bateman[x] for x in max_min_array[1]], [Mie_PF_Nikalov[x] for x in max_min_array[2]], [Mie_PF_Ma[x] for x in max_min_array[3]], [Mie_PF_Sultanova[x] for x in max_min_array[4]], [Mie_PF_Kasarova[x] for x in max_min_array[5]], [Mie_PF_Miles[x] for x in max_min_array[6]], [Mie_PF_Jones[x] for x in max_min_array[7]], [Mie_PF_Greenslade[x] for x in max_min_array[8]], [Mie_PF_Gienger[x] for x in max_min_array[9]]], axis=0)
theta_uncertainty = [theta_max_min_std[x] / theta_max_min_avg[x] * 100 for x in range(len(theta_max_min_avg))]

# figure with local max and minima and associated errors when averaging all theory theta
f0, ax0 = plt.subplots(figsize=(12, 7))
ax0.semilogy(theta_mie, Mie_PF_Matheson, label=groups[0])
ax0.semilogy([theta_mie[x] for x in max_min_array[0]], [Mie_PF_Matheson[x] for x in max_min_array[0]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[0]], [Mie_PF_Matheson[x] for x in max_min_array[0]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Bateman, label=groups[1])
ax0.semilogy([theta_mie[x] for x in max_min_array[1]], [Mie_PF_Bateman[x] for x in max_min_array[1]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[1]], [Mie_PF_Bateman[x] for x in max_min_array[1]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Nikalov, label=groups[2])
ax0.semilogy([theta_mie[x] for x in max_min_array[2]], [Mie_PF_Nikalov[x] for x in max_min_array[2]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[0]], [Mie_PF_Nikalov[x] for x in max_min_array[2]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Ma, label=groups[3])
ax0.semilogy([theta_mie[x] for x in max_min_array[3]], [Mie_PF_Ma[x] for x in max_min_array[3]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[3]], [Mie_PF_Ma[x] for x in max_min_array[3]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Sultanova, label=groups[4])
ax0.semilogy([theta_mie[x] for x in max_min_array[4]], [Mie_PF_Sultanova[x] for x in max_min_array[4]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[4]], [Mie_PF_Sultanova[x] for x in max_min_array[4]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Kasarova, label=groups[5])
ax0.semilogy([theta_mie[x] for x in max_min_array[5]], [Mie_PF_Kasarova[x] for x in max_min_array[5]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[5]], [Mie_PF_Kasarova[x] for x in max_min_array[5]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Miles, label=groups[6])
ax0.semilogy([theta_mie[x] for x in max_min_array[6]], [Mie_PF_Miles[x] for x in max_min_array[6]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[6]], [Mie_PF_Miles[x] for x in max_min_array[6]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Jones, label=groups[7])
ax0.semilogy([theta_mie[x] for x in max_min_array[7]], [Mie_PF_Jones[x] for x in max_min_array[7]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[7]], [Mie_PF_Jones[x] for x in max_min_array[7]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Greenslade, label=groups[8])
ax0.semilogy([theta_mie[x] for x in max_min_array[8]], [Mie_PF_Greenslade[x] for x in max_min_array[8]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[8]], [Mie_PF_Greenslade[x] for x in max_min_array[8]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10)
ax0.semilogy(theta_mie, Mie_PF_Gienger, label=groups[9])
ax0.semilogy([theta_mie[x] for x in max_min_array[9]], [Mie_PF_Gienger[x] for x in max_min_array[9]], color='black', marker='*', ms=3, ls=' ')
#ax0.errorbar([theta_mie[x] for x in max_min_array[9]], [Mie_PF_Gienger[x] for x in max_min_array[9]], xerr=2*np.array(theta_max_min_std), color='black', ls=' ', capsize=10, label='Maxima & Minima\n with Error in \u03b8')
ax0.errorbar(theta_max_min_avg, pf_max_min_avg, xerr=2*np.array(theta_max_min_std), yerr=2*np.array(pf_max_min_std), color='black', ls=' ', capsize=2.5, label='Maxima & Minima\n with Error in \u03b8')
ax0.set_title('PSL Phase Functions Calculated from Various Refractive Index\n Values Given in the Literature')
ax0.set_xlabel('\u03b8')
ax0.set_ylabel('Intensity')
ax0.grid(True)
ax0.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Mie_PFs.png', format='png')
plt.savefig(Save_Directory + 'Mie_PFs.pdf', format='pdf')
plt.show()


'''
# import experiment data for 900nm PSL
Exp_Directory = '/home/austen/Desktop/'
Exp_Data = pd.read_csv(Exp_Directory, delimiter=',', header=0)
Exp_Ray_PF = Exp_Data['N2 Intensity']
Exp_Ray_PN = np.asarray(Exp_Data['N2 Columns'])
Exp_Particle_PF = np.asarray(Exp_Data['Sample Intensity'] - Exp_Ray_PF)
Exp_Particle_PN = Exp_Data['Sample Columns'] # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!
'''

'''
# this makes the mie data the same array length as the experimental data
Mie_PF_Matheson = pchip_interpolate(theta_mie, mie_data[0, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Bateman = pchip_interpolate(theta_mie, mie_data[1, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Nikalov = pchip_interpolate(theta_mie, mie_data[2, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Ma = pchip_interpolate(theta_mie, mie_data[3, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Sultanova = pchip_interpolate(theta_mie, mie_data[4, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Kasarova = pchip_interpolate(theta_mie, mie_data[5, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Miles = pchip_interpolate(theta_mie, mie_data[6, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Jones = pchip_interpolate(theta_mie, mie_data[7, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Greenslade = pchip_interpolate(theta_mie, mie_data[8, 3], Exp_Particle_PF, der=0, axis=0)
Mie_PF_Gienger = pchip_interpolate(theta_mie, mie_data[9, 3], Exp_Particle_PF, der=0, axis=0)
'''

# smooth experimental scattering diagrams by savitzky golay to eliminate noise spikes!
'''
Exp_Particle_PF_Savgol = savgol_filter(Exp_Particle_PF, window_length=151, polyorder=2, deriv=0)


#PSL_900_Savgol_Pchip = pchip_interpolate(PSL_900_PN, PSL_900_Savgol, PSL_900_PN, der=0, axis=0)



# find all local maxima and minima in the 900nm PSL Mie scattering diagram

print('900nm PSL Features(Index) and PN:')
Mie_900_Max = np.argmax(Mie_900_Intensity_Var)
print('Mie maximum: ', Mie_900_Max)
Mie_900_Local_Max = np.asarray(argrelmax(Mie_900_Intensity_Var, axis=0)).flatten()
print('Mie local max indices: ', Mie_900_Local_Max)
Mie_900_Local_Min = np.asarray(argrelmin(Mie_900_Intensity_Var, axis=0)).flatten()
print('Mie local min indices: ', Mie_900_Local_Min)
Mie_900_Local_Features = sorted(list(set(np.concatenate((Mie_900_Max, Mie_900_Local_Max, Mie_900_Local_Min), axis=None).ravel().tolist())))
del Mie_900_Local_Features[0]
#print('Mie local features: ', Mie_Local_Features)
Mie_900_Featured_Angles = [Mie_900_Angles[element] for element in Mie_900_Local_Features]
print('Mie featured angles: ', Mie_900_Featured_Angles)

# find all local maxima and minima for 900nm PSL in measured scattering diagram
Exp_900_Max = np.argmax(PSL_900_Savgol_Pchip)
print('Exp maximum: ', Exp_900_Max)
Exp_900_Local_Max = np.asarray(argrelmax(PSL_900_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local max indices: ', Exp_900_Local_Max)
Exp_900_Local_Min = np.asarray(argrelmin(PSL_900_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local min indices: ', Exp_900_Local_Min)
# note that Exp_Local_Features parses over an index that corresponds to the length of the SD array (0 ~ 790)
Exp_900_Local_Features = sorted(list(set(np.concatenate((Exp_900_Max, Exp_900_Local_Max, Exp_900_Local_Min), axis=None).ravel().tolist())))
print('All exp local features indexes: \n', Exp_900_Local_Features)
print('All exp local features length: ', len(Exp_900_Local_Features))
drop = [0]
for index in sorted(drop, reverse=True):
    del Exp_900_Local_Features[index]
Exp_900_Local_Features = sorted(Exp_900_Local_Features)
# had to add these in when the last local minima is too shallow
Exp_900_Local_Features.append(636)
Exp_900_Local_Features.append(696)
Exp_900_Local_PN = [PSL_900_PN[x] for x in Exp_900_Local_Features]
print('All exp local features pn: \n', Exp_900_Local_PN)
print('Kept local features: ', Exp_900_Local_Features)
print('Length del local features: ', len(Exp_900_Local_Features))
Features_900_PN = []
for element in Exp_900_Local_Features:
    Features_900_PN.append(PSL_900_PN[element])
# note Features PN corresponds to the actual index of the CCD, (200 ~ 1000)
print('Profile Numbers @ exp local features: ', Features_900_PN)
# pull Mie 900nm PSL intensities from local max and minima, create arrays
Mie_900_Intensities_at_Features = [Mie_900_Intensity_Var[element] for element in Mie_900_Local_Features]
# element in Exp_Local_Features or Features_PN
Exp_900_Intensities_at_Features = [PSL_900_Savgol[element] for element in Exp_900_Local_Features]
Mie_900_to_Exp_Intenisty_Ratios_at_Features = np.divide(np.array(Mie_900_Intensities_at_Features), np.array(Exp_900_Intensities_at_Features))
Ratio_900_Avg = np.average(Mie_900_to_Exp_Intenisty_Ratios_at_Features)


# plot 900 imported data
f2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
ax2[0].plot(Mie_900_Angles, Mie_900_Intensity_Matheson, color='orange', ls='-', label='Matheson 1957 900nm PSL')
ax2[0].plot(Mie_900_Angles, Mie_900_Intensity_Ma, color='purple', ls='-', label='Ma 2003 900nm PSL')
ax2[0].plot(Mie_900_Angles, Mie_900_Intensity_Greenslade, color='black', ls='-', label='Greenslade 2017 900nm PSL')
ax2[0].set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax2[0].set_xlabel('Angles (\u00B0)')
ax2[0].set_ylabel('Intensity')
ax2[0].legend(loc=1)
ax2[0].set_yscale('log')
ax2[0].grid(True)
ax2[1].plot(PSL_900_PN, PSL_900_Intensity, color='blue', ls='-', label='Raw 900nm PSL')
ax2[1].plot(PSL_900_PN, PSL_900_Savgol, color='lawngreen', ls='-', label='Savgol 900nm PSL')
ax2[1].plot(PSL_900_PN, PSL_900_Savgol_Pchip, color='red', ls='-', label='Savgol + Pchip 900nm PSL')
ax2[1].plot(Exp_900_Local_PN, Exp_900_Intensities_at_Features, marker='*', ls='', ms=6, color='black', label='Local Max & Min')
ax2[1].set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax2[1].set_xlabel('Profile Number')
ax2[1].set_ylabel('Intensity')
ax2[1].set_yscale('log')
ax2[1].legend(loc=1)
ax2[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + '900nm_PSL.pdf', format='pdf')
plt.show()


# create a 2d array of Mie theta and Exp PN at local features, combining all the 600, 800, and 900 psl data
All_PN = np.concatenate((Features_600_PN, Features_800_PN, Features_900_PN), axis=None).ravel().tolist()
All_Theta = np.concatenate((Mie_600_Featured_Angles, Mie_800_Featured_Angles, Mie_900_Featured_Angles), axis=None).ravel().tolist()
print('All_PN: ', All_PN)
print('All_Theta: ', All_Theta)

# OLS on all PSL data
All_PN_W_Const = sm.add_constant(All_PN) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model0 = sm.OLS(All_Theta, All_PN_W_Const)
results0 = model0.fit()
print(results0.summary())

# plotting all the local max & min on one plot and conducting an OLS on all features
f3, ax3 = plt.subplots(figsize=(12, 7))
ax3.plot(Features_600_PN, Mie_600_Featured_Angles, marker='o', ms='4', ls='', color='red', label='600nm PSL Local Max & Min')
ax3.plot(Features_800_PN, Mie_800_Featured_Angles, marker='^', ms='4', ls='', color='blue', label='800nm PSL Local Max & Min')
ax3.plot(Features_900_PN, Mie_900_Featured_Angles, marker='x', ms='4', ls='', color='green', label='900nm PSL Local Max & Min')
ax3.plot(All_PN, results0.fittedvalues, color='black', ls='-', label='OLS: y = ' + str('{:.4f}'.format(results0.params[1])) + 'x + ' + str('{:.4f}'.format(results0.params[0])))
ax3.grid(True)
ax3.set_title('Linear Regression to All Local Features in the PSL Data')
ax3.set_xlabel('PN')
ax3.set_ylabel('\u03b8 (\u00b0)')
ax3.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'All_PSL_Calibration.pdf', format='pdf')
plt.show()

# do a linear fit to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
Exp_X_Vals_OLS = sm.add_constant(Features_900_PN) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model1 = sm.OLS(Mie_900_Featured_Angles, Exp_X_Vals_OLS)
results1 = model1.fit()
print(results1.summary())



# plot imported data
f4 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax4a = f4.add_subplot(gs[0, 0])
ax4a.plot(Mie_900_Angles, Mie_900_Intensity_Var, 'r-', label='900nm PSL')
ax4a.plot(Mie_900_Featured_Angles, Mie_900_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min')
ax4a.set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax4a.set_xlabel('Angles (\u00B0)')
ax4a.set_ylabel('Intensity')
ax4a.legend(loc=1)
ax4a.set_yscale('log')
ax4a.grid(True)
ax4b = f4.add_subplot(gs[0, 1])
ax4b.plot(PSL_900_PN, PSL_900_Intensity, 'b-', label='900nm PSL')
ax4b.plot(PSL_900_PN, PSL_900_Savgol, 'y-', label='900nm PSL Smoothed')
ax4b.plot(Features_900_PN, Exp_900_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax4b.set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax4b.set_xlabel('Profile Number')
ax4b.set_ylabel('Intensity')
ax4b.set_yscale('log')
ax4b.legend(loc=1)
ax4b.grid(True)
ax4c = f4.add_subplot(gs[1, 0])
ax4c.plot(Features_900_PN, Mie_900_Featured_Angles, marker='o', ls='', color='green', label='Angles vs Profile Numbers')
ax4c.plot(Features_900_PN, results1.fittedvalues, color='black', linestyle='-', label='OLS: y = ' + str('{:.4f}'.format(results1.params[1])) + 'x + ' + str('{:.4f}'.format(results1.params[0])))
ax4c.set_title('Scattering Angle as a Function of Profile Number')
ax4c.set_xlabel('Profile Number')
ax4c.set_ylabel('Scattering Angle (\u00B0)')
ax4c.legend(loc=2)
ax4c.grid(True)
ax4d = f4.add_subplot(gs[1, 1])
ax4d.plot(Mie_900_Featured_Angles, Mie_900_to_Exp_Intenisty_Ratios_at_Features, marker='^', color='yellow', label='Mie:Exp Intensity Ratio vs. Angle')
ax4d.set_title('Mie:Measured Intensity Ratio as a \n Function of Local Max/Min Angle')
ax4d.set_xlabel('Scattering Angle (\u00B0)')
ax4d.set_ylabel('Intensity Ratio')
ax4d.legend(loc=1)
ax4d.grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + '900nm_Calibration.pdf', format='pdf')
plt.show()


# Apply correction by adding the delta angle correction to the angle axis of the PSL scattering data

# Increase the resolution of the Mie data such that it is the same number of data points
# as the experimental data

Mie_600_Spline_Func = interp1d(Mie_600_Angles, Mie_600_Intensity_Var, kind='cubic')
Mie_600_Spline_Angles = np.linspace(0, 180, len(PSL_600_PN), endpoint=False)
Mie_600_Spline_Intensity = Mie_600_Spline_Func(Mie_600_Spline_Angles)

Mie_800_Spline_Func = interp1d(Mie_800_Angles, Mie_800_Intensity_Var, kind='cubic')
Mie_800_Spline_Angles = np.linspace(0, 180, len(PSL_800_PN), endpoint=False)
Mie_800_Spline_Intensity = Mie_800_Spline_Func(Mie_800_Spline_Angles)

Mie_900_Spline_Func = interp1d(Mie_900_Angles, Mie_900_Intensity_Var, kind='cubic')
Mie_900_Spline_Angles = np.linspace(0, 180, len(PSL_900_PN), endpoint=False)
Mie_900_Spline_Intensity = Mie_900_Spline_Func(Mie_900_Spline_Angles)

# First normalize the intensities and plot them one ontop of the other
Mie_600_Spline_Int_Norm = Mie_600_Spline_Intensity / np.linalg.norm(Mie_600_Spline_Intensity)
Mie_800_Spline_Int_Norm = Mie_800_Spline_Intensity / np.linalg.norm(Mie_800_Spline_Intensity)
Mie_900_Spline_Int_Norm = Mie_900_Spline_Intensity / np.linalg.norm(Mie_900_Spline_Intensity)

PSL_600_Savgol_Int_Norm = PSL_600_Savgol / np.linalg.norm(PSL_600_Savgol)
PSL_800_Savgol_Int_Norm = PSL_800_Savgol / np.linalg.norm(PSL_800_Savgol)
PSL_900_Savgol_Int_Norm = PSL_900_Savgol / np.linalg.norm(PSL_900_Savgol)

# parameters from linear calibration conducted like manfred et al.
slope = results0.params[1]
intercept = results0.params[0]
#slope = 0.2112
#intercept = -47.972

# convert profile numbers to angles with the OLS from the Manfred appraoach
PSL_600_Profiles_to_Angles = [(slope * x) + intercept for x in PSL_600_PN]
PSL_600_Profiles_to_Angles = np.array(PSL_600_Profiles_to_Angles)
print(type(PSL_600_Profiles_to_Angles), PSL_600_Profiles_to_Angles.shape)

PSL_800_Profiles_to_Angles = [(slope * x) + intercept for x in PSL_800_PN]
PSL_800_Profiles_to_Angles = np.array(PSL_800_Profiles_to_Angles)
print(type(PSL_800_Profiles_to_Angles), PSL_800_Profiles_to_Angles.shape)

PSL_900_Profiles_to_Angles = [(slope * x) + intercept for x in PSL_900_PN]
PSL_900_Profiles_to_Angles = np.array(PSL_900_Profiles_to_Angles)
print(type(PSL_900_Profiles_to_Angles), PSL_900_Profiles_to_Angles.shape)


# make a plot of the 900 spline data and the experimental data overlayed
f5, ax5 = plt.subplots(figsize=(12, 6))
#pt0, = ax5.plot(Mie_900_Spline_Angles, Mie_900_Spline_Intensity, color='red', linestyle='-', label='Mie 900nm PSL Spline vs. Theta')
pt = ax5.plot(PSL_900_Profiles_to_Angles, PSL_900_Savgol * Ratio_900_Avg, color='purple', linestyle='-', lw=4, label='Meas. 900nm PSL Savgol vs. Theta')
#pt2, = ax5.plot(PSL_900_Profiles_to_Angles, PSL_900_Savgol, color='green', linestyle='-', label='Exp 900nm PSL Savgol vs. Theta')
#ax5.set_xlabel('Angles (\u00B0)', color='red')
ax5.set_xlabel('Angles (\u00B0)', fontsize=20)
ax5.set_ylabel('Intensity', fontsize=20)
#ax5.set_title('900nm Polystyrene Latex Sphere Raw Phase Function to Calibrated Phase Function')
#ax5.tick_params(axis='x', labelcolor='red')
ax5.tick_params(axis='x')
ax5.minorticks_on()
ax5.grid(True, which='both')
ax5.set_yscale('log')
#ax5.legend(loc=1)
#ax5a = ax5.twiny()
#pt3, = ax5a.plot(PSL_900_PN, PSL_900_Savgol, color='blue', linestyle='-', label='Exp 900nm PSL Savgol vs. PN')
#ax5a.set_xlabel('Profile Numbers', color='blue')
#ax5a.set_ylabel('Intensity')
#ax5a.tick_params(axis='x', labelcolor='blue')
#pt = [pt0, pt1, pt2, pt3]
ax5.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
#ax5.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + '900nm_PSL_Calibrated.pdf', format='pdf')
plt.show()

C0 = pd.Series(Mie_900_Spline_Angles, name='Spline Mie Theta')
C1 = pd.Series(Mie_900_Spline_Intensity, name='Spline Mie Intensity')
C2 = pd.Series(PSL_900_PN, name='PN')
C3 = pd.Series(PSL_900_Profiles_to_Angles, name='PN to Angle')
C4 = pd.Series(PSL_900_Savgol, name='Exp Smoothed Intensity')
C5 = pd.Series(Features_900_PN, name='Cal Exp Profiles Local Max & Min')
C6 = pd.Series(Mie_900_Featured_Angles, name='Cal Mie Angles Local Max & Min')
C7 = pd.Series([(x * results0.fittedvalues[1]) + results0.fittedvalues[0] for x in Features_900_PN], name='Cal Fit Angles')
C8 = pd.Series(results0.fittedvalues[1], name='Cal Fit Slope')
C9 = pd.Series(results0.fittedvalues[0], name='Cal Fit Intercept')
All_Data = pd.concat([C0, C1, C2, C3, C4, C5, C6, C7, C8, C9], axis=1)
All_Data.to_csv(Save_Directory + '/Calibrated_Data_PSL900nm.txt')

# plot calibrated PSL 600nm data against theory
f6, ax6 = plt.subplots(figsize=(12, 6))
pt0, = ax6.plot(Mie_600_Spline_Angles, Mie_600_Spline_Intensity, color='red', linestyle='-', label='Mie 600nm PSL Spline vs. Theta')
pt1, = ax6.plot(PSL_600_Profiles_to_Angles, PSL_600_Savgol * Ratio_900_Avg * 0.30, color='purple', linestyle='-', label='Exp 600nm PSL Savgol Scaled vs. Theta')
pt2, = ax6.plot(PSL_600_Profiles_to_Angles, PSL_600_Savgol, color='green', linestyle='-', label='Exp 600nm PSL Savgol vs. Theta')
ax6.set_xlabel('Angles (\u00B0)', color='red')
ax6.set_ylabel('Intensity')
ax6.set_title('600nm Polystyrene Latex Sphere Raw Phase Function to Calibrated Phase Function')
ax6.tick_params(axis='x', labelcolor='red')
ax6.minorticks_on()
ax6.grid(True, which='both')
ax6.legend(loc=1)
ax6.set_yscale('log')
ax6a = ax6.twiny()
pt3, = ax6a.plot(PSL_600_PN, PSL_600_Savgol, color='blue', linestyle='-', label='Exp. Smoothed Intensity vs. PN')
ax6a.set_xlabel('Profile Numbers', color='blue')
ax6a.set_ylabel('Intensity')
ax6a.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2, pt3]
ax6.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax6.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + '600nm_PSL_Calibrated.pdf', format='pdf')
plt.show()


# plot calibrated PSL 800nm data against theory
f7, ax7 = plt.subplots(figsize=(12, 6))
pt0, = ax7.plot(Mie_800_Spline_Angles, Mie_800_Spline_Intensity, color='red', linestyle='-', label='Mie 800nm PSL Spline vs. Theta')
pt1, = ax7.plot(PSL_800_Profiles_to_Angles, PSL_800_Savgol * Ratio_900_Avg * 0.36, color='purple', linestyle='-', label='Exp 800nm PSL Savgol Scaled vs. Theta')
pt2, = ax7.plot(PSL_800_Profiles_to_Angles, PSL_800_Savgol, color='green', linestyle='-', label='Exp 800nm PSL Savgol vs. Theta')
ax7.set_xlabel('Angles (\u00B0)', color='red')
ax7.set_ylabel('Intensity')
ax7.set_title('800nm Polystyrene Latex Sphere Raw Phase Function to Calibrated Phase Function')
ax7.tick_params(axis='x', labelcolor='red')
ax7.minorticks_on()
ax7.grid(True, which='both')
ax7.legend(loc=1)
ax7.set_yscale('log')
ax7a = ax7.twiny()
pt3, = ax7a.plot(PSL_800_PN, PSL_800_Savgol, color='blue', linestyle='-', label='Exp. Smoothed Intensity vs. PN')
ax7a.set_xlabel('Profile Numbers', color='blue')
ax7a.set_ylabel('Intensity')
ax7a.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2, pt3]
ax7.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax7.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + '800nm_PSL_Calibrated.pdf', format='pdf')
plt.show()
'''