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
Save_Directory = '/home/austen/Desktop/2019-09-08_Analysis/'



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
d = 903
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


# calculate weighted average of phase function data based on weights determined from the lognormal distribution
for counter_0, m_i in enumerate(m_array):
    group = groups[counter_0]
    for counter_1, element_1 in enumerate(size_array):
        theta_mie, SL, SR, SU = ps.ScatteringFunction(m_i, w_n, element_1, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        pf_2darray.append(SL)
    pf_average = np.average(pf_2darray, axis=0, weights=log_dist)
    mie_data.append([group, m_i, pf_average])
    pf_2darray = []


# find all local maxima and minima in the 900nm PSL Mie scattering diagram
mie_df = np.array(mie_data)
mie_max_min_array = []
for counter, element in enumerate(m_array):
    mie_maximum = np.argmax(mie_data[counter][2], axis=0)
    mie_minimum = np.argmin(mie_data[counter][2], axis=0)
    mie_local_max = argrelmax(mie_data[counter][2], axis=0)
    mie_local_min = argrelmin(mie_data[counter][2], axis=0)
    mie_max_min_idx = np.sort(np.concatenate((mie_local_max, mie_local_min), axis=None))
    mie_max_min_array.append(mie_max_min_idx)
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


Mie_Theory_DF = pd.DataFrame()
Mie_Theory_DF['Theta'] = theta_mie
Mie_Theory_DF['Matheson'] = mie_data[0][2]
Mie_Theory_DF['Bateman'] = mie_data[1][2]
Mie_Theory_DF['Nikalov'] = mie_data[2][2]
Mie_Theory_DF['Ma'] = mie_data[3][2]
Mie_Theory_DF['Sultanova'] = mie_data[4][2]
Mie_Theory_DF['Kasarova'] = mie_data[5][2]
Mie_Theory_DF['Miles'] = mie_data[6][2]
Mie_Theory_DF['Jones'] = mie_data[7][2]
Mie_Theory_DF['Greenslade'] = mie_data[8][2]
Mie_Theory_DF['Gienger'] = mie_data[9][2]
#Mie_Theory_DF.to_csv(Save_Directory + 'Mie_Theory_DF.txt', sep=',')


theta_max_min_avg = np.mean([[theta_mie[x] for x in mie_max_min_array[0]], [theta_mie[x] for x in mie_max_min_array[1]], [theta_mie[x] for x in mie_max_min_array[2]], [theta_mie[x] for x in mie_max_min_array[3]], [theta_mie[x] for x in mie_max_min_array[4]], [theta_mie[x] for x in mie_max_min_array[5]], [theta_mie[x] for x in mie_max_min_array[6]], [theta_mie[x] for x in mie_max_min_array[7]], [theta_mie[x] for x in mie_max_min_array[8]], [theta_mie[x] for x in mie_max_min_array[9]]], axis=0)
pf_max_min_avg = np.mean([[Mie_PF_Matheson[x] for x in mie_max_min_array[0]], [Mie_PF_Bateman[x] for x in mie_max_min_array[1]], [Mie_PF_Nikalov[x] for x in mie_max_min_array[2]], [Mie_PF_Ma[x] for x in mie_max_min_array[3]], [Mie_PF_Sultanova[x] for x in mie_max_min_array[4]], [Mie_PF_Kasarova[x] for x in mie_max_min_array[5]], [Mie_PF_Miles[x] for x in mie_max_min_array[6]], [Mie_PF_Jones[x] for x in mie_max_min_array[7]], [Mie_PF_Greenslade[x] for x in mie_max_min_array[8]], [Mie_PF_Gienger[x] for x in mie_max_min_array[9]]], axis=0)
theta_max_min_std = np.std([[theta_mie[x] for x in mie_max_min_array[0]], [theta_mie[x] for x in mie_max_min_array[1]], [theta_mie[x] for x in mie_max_min_array[2]], [theta_mie[x] for x in mie_max_min_array[3]], [theta_mie[x] for x in mie_max_min_array[4]], [theta_mie[x] for x in mie_max_min_array[5]], [theta_mie[x] for x in mie_max_min_array[6]], [theta_mie[x] for x in mie_max_min_array[7]], [theta_mie[x] for x in mie_max_min_array[8]], [theta_mie[x] for x in mie_max_min_array[9]]], axis=0)
pf_max_min_std = np.std([[Mie_PF_Matheson[x] for x in mie_max_min_array[0]], [Mie_PF_Bateman[x] for x in mie_max_min_array[1]], [Mie_PF_Nikalov[x] for x in mie_max_min_array[2]], [Mie_PF_Ma[x] for x in mie_max_min_array[3]], [Mie_PF_Sultanova[x] for x in mie_max_min_array[4]], [Mie_PF_Kasarova[x] for x in mie_max_min_array[5]], [Mie_PF_Miles[x] for x in mie_max_min_array[6]], [Mie_PF_Jones[x] for x in mie_max_min_array[7]], [Mie_PF_Greenslade[x] for x in mie_max_min_array[8]], [Mie_PF_Gienger[x] for x in mie_max_min_array[9]]], axis=0)
theta_uncertainty = [theta_max_min_std[x] / theta_max_min_avg[x] * 100 for x in range(len(theta_max_min_avg))]


# figure with local max and minima and associated errors when averaging all theory theta
f0, ax0 = plt.subplots(figsize=(12, 7))
ax0.semilogy(theta_mie, Mie_PF_Matheson, label=groups[0])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[0]], [Mie_PF_Matheson[x] for x in mie_max_min_array[0]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Bateman, label=groups[1])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[1]], [Mie_PF_Bateman[x] for x in mie_max_min_array[1]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Nikalov, label=groups[2])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[2]], [Mie_PF_Nikalov[x] for x in mie_max_min_array[2]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Ma, label=groups[3])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[3]], [Mie_PF_Ma[x] for x in mie_max_min_array[3]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Sultanova, label=groups[4])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[4]], [Mie_PF_Sultanova[x] for x in mie_max_min_array[4]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Kasarova, label=groups[5])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[5]], [Mie_PF_Kasarova[x] for x in mie_max_min_array[5]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Miles, label=groups[6])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[6]], [Mie_PF_Miles[x] for x in mie_max_min_array[6]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Jones, label=groups[7])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[7]], [Mie_PF_Jones[x] for x in mie_max_min_array[7]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Greenslade, label=groups[8])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[8]], [Mie_PF_Greenslade[x] for x in mie_max_min_array[8]], color='black', marker='*', ms=3, ls=' ')
ax0.semilogy(theta_mie, Mie_PF_Gienger, label=groups[9])
ax0.semilogy([theta_mie[x] for x in mie_max_min_array[9]], [Mie_PF_Gienger[x] for x in mie_max_min_array[9]], color='black', marker='*', ms=3, ls=' ')
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


# import experiment data for 900nm PSL
#Exp_Directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/2019-09-06/2019-09-06_Analysis/90/SD_Particle.txt'
Exp_Directory = '/home/austen/Desktop/2019-09-08_Analysis/900nm_windows_on/SD_Particle.txt'
Exp_Data = pd.read_csv(Exp_Directory, delimiter=',', header=0)
Exp_Ray_PF = Exp_Data['N2 Intensity']
Exp_Ray_PN = np.asarray(Exp_Data['N2 Columns'])
#Exp_Particle_PF = np.asarray(Exp_Data['Sample Intensity'] - Exp_Ray_PF)
# if you don't want to perform a nitrogen/background gas subtraction use the below!
Exp_Particle_PF = np.asarray(Exp_Data['Sample Intensity'])
Exp_Particle_PF = Exp_Particle_PF[~np.isnan(Exp_Particle_PF)]
Exp_Particle_PN = Exp_Data['Sample Columns'] # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!
Exp_Particle_PN = Exp_Particle_PN[~np.isnan(Exp_Particle_PN)]


# smooth experimental scattering diagrams by savitzky golay to eliminate noise spikes!
Exp_Particle_PF_Savgol = savgol_filter(Exp_Particle_PF, window_length=151, polyorder=2, deriv=0)
Exp_Particle_PF_Savgol_Pchip = pchip_interpolate(xi=Exp_Particle_PN, yi=Exp_Particle_PF_Savgol, x=Exp_Particle_PN, der=0, axis=0)


# find all local maxima and minima for 900nm PSL in measured scattering diagram
# you gotta play with the order number to pick up all local max and min, then drop the features that don't belong
exp_max_min_array = []
exp_maximum = np.argmax(Exp_Particle_PF_Savgol_Pchip, axis=0)
exp_minimum = np.argmin(Exp_Particle_PF_Savgol_Pchip, axis=0)
exp_local_max = argrelmax(Exp_Particle_PF_Savgol_Pchip, axis=0, order=50)
exp_local_min = argrelmin(Exp_Particle_PF_Savgol_Pchip, axis=0, order=50)
exp_max_min_idx = np.sort(np.concatenate((exp_local_max, exp_local_min), axis=None))
#exp_max_min_idx = np.delete(exp_max_min_idx, [0])
exp_max_min_array = [Exp_Particle_PN[x] for x in exp_max_min_idx]
print('Mie Local Features Index: ', theta_max_min_avg)
print('Experiment Local Features Index: ', exp_max_min_array)


# do a linear fit to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
Exp_X_Vals_OLS = sm.add_constant(exp_max_min_array) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model1 = sm.OLS(theta_max_min_avg, Exp_X_Vals_OLS)
results1 = model1.fit()
print(results1.summary())

Mie_Cal_Ints = np.array([Mie_PF_Gienger[x] for x in mie_max_min_array[9]])
Exp_Cal_Ints = np.array([Exp_Particle_PF_Savgol[x] for x in exp_max_min_idx])
Exp_to_Mie_Ratio = ((Exp_Cal_Ints - Mie_Cal_Ints) / Mie_Cal_Ints) * 100

# plot imported data
f4 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax4a = f4.add_subplot(gs[0, 0])
ax4a.semilogy(theta_mie, mie_data[9][2], 'r-', label='900nm PSL')
ax4a.semilogy([theta_mie[x] for x in mie_max_min_array[9]], [mie_data[9][2][x] for x in mie_max_min_array[9]], marker='X', color='black', linestyle='None', label='Local Max & Min')
ax4a.set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax4a.set_xlabel('Angles (\u00B0)')
ax4a.set_ylabel('Intensity')
ax4a.legend(loc=1)
ax4a.grid(True)
ax4b = f4.add_subplot(gs[0, 1])
ax4b.semilogy(Exp_Particle_PN, Exp_Particle_PF, 'b-', label='900nm PSL')
#ax4b.semilogy(Exp_Particle_PN, Exp_Particle_PF_Savgol, 'y-', label='900nm PSL Smoothed')
#ax4b.semilogy(exp_max_min_array, [Exp_Particle_PF_Savgol[x] for x in exp_max_min_idx], marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax4b.semilogy(exp_max_min_array, [Exp_Particle_PF[x] for x in exp_max_min_idx], marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax4b.set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax4b.set_xlabel('Profile Number')
ax4b.set_ylabel('Intensity')
ax4b.legend(loc=1)
ax4b.grid(True)
ax4c = f4.add_subplot(gs[1, 0])
ax4c.plot(exp_max_min_array, theta_max_min_avg, marker='o', ls='', color='green', label='Angles vs Profile Numbers')
ax4c.plot(exp_max_min_array, results1.fittedvalues, color='black', linestyle='-', label='OLS: y = ' + str('{:.4f}'.format(results1.params[1])) + 'x + ' + str('{:.4f}'.format(results1.params[0])))
ax4c.set_title('Scattering Angle as a Function of Profile Number')
ax4c.set_xlabel('Profile Number')
ax4c.set_ylabel('Scattering Angle (\u00B0)')
ax4c.legend(loc=2)
ax4c.grid(True)
ax4d = f4.add_subplot(gs[1, 1])
ax4d.plot(theta_max_min_avg, Exp_to_Mie_Ratio, marker='^', color='purple', label='Mie:Exp Intensity Ratio vs. Angle')
ax4d.set_title('Mie:Measured Intensity Percent Difference as a \n Function of Local Max/Min Angle')
ax4d.set_xlabel('Scattering Angle (\u00B0)')
ax4d.set_ylabel('% Difference')
ax4d.legend(loc=1)
ax4d.grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + '900nm_Calibration.png', format='png')
plt.savefig(Save_Directory + '900nm_Calibration.pdf', format='pdf')
plt.show()

Calibrated_Theta = np.array([(results1.params[1] * x) + results1.params[0] for x in Exp_Particle_PN])
print('Angular Range: ', [Calibrated_Theta[0], Calibrated_Theta[-1]])
print('Angles: ', [(results1.params[1] * x) + results1.params[0] for x in Exp_Particle_PN])
print('Max Index: ', np.argmax(Exp_Particle_PF))

# this makes the mie data the same array length as the experimental data
Mie_PF_Matheson = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Matheson, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Bateman = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Bateman, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Nikalov = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Nikalov, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Ma = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Ma, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Sultanova = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Sultanova, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Kasarova = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Kasarova, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Miles = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Miles, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Jones = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Jones, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Greenslade = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Greenslade, x=Calibrated_Theta, der=0, axis=0)
Mie_PF_Gienger = pchip_interpolate(xi=theta_mie, yi=Mie_PF_Gienger, x=Calibrated_Theta, der=0, axis=0)

comparator = np.absolute((Exp_Particle_PF * 0.0010) - Mie_PF_Gienger)


# make a plot of the 900 spline data and the experimental data overlayed
f5, ax5 = plt.subplots(1, 2, figsize=(12, 6))
ax5[0].semilogy(theta_mie, np.array(mie_data[9][2]), color='red', linestyle='-', label='Mie Theory')
ax5[0].semilogy(Calibrated_Theta, Exp_Particle_PF * 0.0010, color='purple', linestyle='-', label='Calibrated Measurement')
ax5[0].set_xlabel('\u03b8 (\u00B0)')
ax5[0].set_ylabel('Intensity')
ax5[0].set_title('Theory and Calibrated Measurement')
#ax5.tick_params(axis='x')
ax5[0].minorticks_on()
ax5[0].grid(True)
ax5[0].legend(loc=1)
ax5[1].plot(Calibrated_Theta, comparator, color='black', linestyle='-', label='Calibrated Measurement')
ax5[1].set_xlabel('\u03b8 (\u00B0)')
ax5[1].set_ylabel('Intensity')
ax5[1].set_title('Absolute Value of the Difference Between\n Experiment and Mie Theory')
#ax5.tick_params(axis='x')
ax5[1].minorticks_on()
ax5[1].grid(True)
ax5[1].legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Validation.pdf', format='pdf')
plt.savefig(Save_Directory + 'Validation.png', format='png')
plt.show()

# Mie theory sample 2
Mie_Directory_2 = '/home/austen/Desktop/2019-09-08_Analysis/800nm/Mie_Theory_DF.txt'
Mie_Data_2 = pd.read_csv(Mie_Directory_2, delimiter=',')
Mie_Sample_PF_2 = Mie_Data_2['Gienger']

# Measurement 2
Exp_Directory_2 = '/home/austen/Desktop/2019-09-08_Analysis/800nm/SD_Particle.txt'
Exp_Data_2 = pd.read_csv(Exp_Directory_2, delimiter=',', header=0)
Exp_Ray_PF_2 = Exp_Data_2['N2 Intensity']
Exp_Ray_PN_2 = np.asarray(Exp_Data_2['N2 Columns'])
#Exp_Particle_PF = np.asarray(Exp_Data['Sample Intensity'] - Exp_Ray_PF)
# if you don't want to perform a nitrogen/background gas subtraction use the below!
Exp_Particle_PF_2 = np.asarray(Exp_Data_2['Sample Intensity'])
Exp_Particle_PN_2 = np.array(Exp_Data_2['Sample Columns'])
Calibrated_Theta_2 = (results1.params[1] * Exp_Particle_PN_2) + results1.params[0]


# Mie theory sample 3
Mie_Directory_3 = '/home/austen/Desktop/2019-09-08_Analysis/600nm/Mie_Theory_DF.txt'
Mie_Data_3 = pd.read_csv(Mie_Directory_3, delimiter=',')
Mie_Sample_PF_3 = np.array(Mie_Data_3['Gienger'])

# Measurement 3
Exp_Directory_3 = '/home/austen/Desktop/2019-09-08_Analysis/600nm/SD_Particle.txt'
Exp_Data_3 = pd.read_csv(Exp_Directory_3, delimiter=',', header=0)
Exp_Ray_PF_3 = Exp_Data_3['N2 Intensity']
Exp_Ray_PN_3 = np.asarray(Exp_Data_3['N2 Columns'])
#Exp_Particle_PF = np.asarray(Exp_Data['Sample Intensity'] - Exp_Ray_PF)
# if you don't want to perform a nitrogen/background gas subtraction use the below!
Exp_Particle_PF_3 = np.array(Exp_Data_3['Sample Intensity'])
Exp_Particle_PF_3 = Exp_Particle_PF_3[~np.isnan(Exp_Particle_PF_3)]
Exp_Particle_PN_3 = np.array(Exp_Data_3['Sample Columns'])
Exp_Particle_PN_3 = Exp_Particle_PN_3[~np.isnan(Exp_Particle_PN_3)]
Calibrated_Theta_3 = (results1.params[1] * Exp_Particle_PN_3) + results1.params[0]

'''
f6, ax6 = plt.subplots(figsize=(12, 6))
ax6.semilogy(Calibrated_Theta_2, Exp_Particle_PF_2 * .0010, color='purple', ls='-', label='Calibrated Measurement')
ax6.semilogy(theta_mie, Mie_Sample_PF_2, color='red', ls='-', label='Mie Theory')
ax6.set_title('Measurement Compared to Mie Theory')
ax6.set_xlabel('\u03b8')
ax6.set_ylabel('Intensity')
ax6.grid(True)
ax6.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Validation2.pdf', format='pdf')
plt.savefig(Save_Directory + 'Validation2.png', format='png')
plt.show()
'''
'''
f7, ax7 = plt.subplots(figsize=(12, 6))
ax7.semilogy(Calibrated_Theta_3, Exp_Particle_PF_3 * 0.0010, color='purple', ls='-', label='Calibrated Measurement')
ax7.semilogy(theta_mie, Mie_Sample_PF_3, color='red', ls='-', label='Mie Theory')
ax7.set_title('Measurement Compared to Mie Theory')
ax7.set_xlabel('\u03b8')
ax7.set_ylabel('Intensity')
ax7.grid(True)
ax7.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Validation3.pdf', format='pdf')
plt.savefig(Save_Directory + 'Validation3.png', format='png')
plt.show()
'''
print(Calibrated_Theta_3)
print(Mie_Sample_PF_3)
# xi is small theory theta axis, yi = small theory intensities, x=new thetas to compute at intensities at, same size as measurement PN axis!
pchip_3 = pchip_interpolate(xi=theta_mie, yi=Mie_Sample_PF_3, x=Calibrated_Theta_3, der=0, axis=0)
comparator_3 = np.absolute((Exp_Particle_PF_3 * 0.0008) - pchip_3)
# make a plot of the 900 spline data and the experimental data overlayed
f8, ax8 = plt.subplots(1, 2, figsize=(12, 6))
ax8[0].semilogy(theta_mie, Mie_Sample_PF_3, color='red', linestyle='-', label='Mie Theory')
ax8[0].semilogy(Calibrated_Theta_3, Exp_Particle_PF_3 * 0.0008, color='purple', linestyle='-', label='Calibrated Measurement')
ax8[0].set_xlabel('\u03b8 (\u00B0)')
ax8[0].set_ylabel('Intensity')
ax8[0].set_title('Theory and Calibrated Measurement')
#ax5.tick_params(axis='x')
ax8[0].minorticks_on()
ax8[0].grid(True)
ax8[0].legend(loc=1)
ax8[1].plot(Calibrated_Theta_3, comparator_3, color='black', linestyle='-', label='Calibrated Measurement')
ax8[1].set_xlabel('\u03b8 (\u00B0)')
ax8[1].set_ylabel('Intensity')
ax8[1].set_title('Absolute Value of the Difference Between\n Experiment and Mie Theory')
#ax5.tick_params(axis='x')
ax8[1].minorticks_on()
ax8[1].grid(True)
ax8[1].legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Validation4.pdf', format='pdf')
plt.savefig(Save_Directory + 'Validation4.png', format='png')
plt.show()
#'''