'''
Austen K. Sruggs
date created:  09-04-2019
Description: script solves for local maxima and minima in
Mie theory and measured scattering diagrams and finds their
indicesThen plots the Mie theory angle of the local maxima
and minima as a function of the profile number, does a pixel to angle calibration
and then attempts to apply a lens correction to the scattering anlge axis by using
the rayleigh scattering data

PSLs:
Mean    Mean Uncertainty     Size Dist Sigma
600nm     9nm                     10.0nm
800nm     14nm                    5.6nm
903nm     12nm                    4.1nm
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
from datetime import date

# import N2 Rayleigh scattering data
Save_Directory = '/home/sm3/Desktop/Recent/'

today = date.today()
today_string = str(today.strftime("%b-%d-%Y"))

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
sigma_g = 1.005
sig = 4.1
# Particle diameter, geometric mean of the particle diameter
d = 900
# wavelength
w_n = 663
# CRI
m_array = [m0[2], m1[2], m2[2], m3[2], m4[2], m5[2], m6[2], m7[2], m8[2], m9[2]]
groups = ['Matheson 1952', 'Bateman 1959', 'Nikalov 2000', 'Ma 2003', 'Sultanova 2003', 'Kasarova 2006', 'Miles 2010', 'Jones 2013', 'Greenslade 2017', 'Gienger 2017']
# size array
size_array = np.arange(d-100, d+100, 2)

# Gaussian distribution function
def Gaussian(x, N, mu, sigma):
   return N * 1/(sigma * sqrt(2*pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# log normal distribution function
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))

# create gaussian distribution values in array
gaus_dist = np.array([Gaussian(element, concentration, d, 4.1) for element in size_array])

# create log normal distribution values in array
log_dist = np.array([LogNormal(element, d, sigma_g, concentration) for element in size_array])

# LogNormal and Guassian Distributions
fA, axA = plt.subplots(figsize=(12, 6))
axA.plot(size_array, gaus_dist, 'r-', label='Gaussian')
axA.plot(size_array, log_dist, 'b-', label='Log Normal')
axA.set_title('Possible Distributions')
axA.set_xlabel('Size Bins')
axA.set_ylabel('Particle Concentrations')
axA.grid(True)
axA.legend(loc=1)
plt.tight_layout()
plt.show()

# pre-allocate arrays
SL_average = []
SR_average = []
SU_average = []

SL_2darray = []
SR_2darray = []
SU_2darray = []
mie_data = []


# calculate weighted average of phase function data based on weights determined from the lognormal distribution
for counter_0, m_i in enumerate(m_array):
    group = groups[counter_0]
    for counter_1, element_1 in enumerate(size_array):
        theta_mie, SL, SR, SU = ps.ScatteringFunction(m_i, w_n, element_1, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SL_2darray.append(SL)
        SR_2darray.append(SR)
        SU_2darray.append(SU)
    SL_average = np.average(SL_2darray, axis=0, weights=log_dist)
    SR_average = np.average(SR_2darray, axis=0, weights=log_dist)
    SU_average = np.average(SU_2darray, axis=0, weights=log_dist)
    mie_data.append([group, m_i, SL_average, SR_average, SU_average])
    SL_2darray = []
    SR_2darray = []
    SU_2darray = []



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


Mie_Theory_DF = pd.DataFrame()
Mie_Theory_DF['Theta'] = theta_mie
Mie_Theory_DF['Matheson SL'] = mie_data[0][2]
Mie_Theory_DF['Bateman SL'] = mie_data[1][2]
Mie_Theory_DF['Nikalov SL'] = mie_data[2][2]
Mie_Theory_DF['Ma SL'] = mie_data[3][2]
Mie_Theory_DF['Sultanova SL'] = mie_data[4][2]
Mie_Theory_DF['Kasarova SL'] = mie_data[5][2]
Mie_Theory_DF['Miles SL'] = mie_data[6][2]
Mie_Theory_DF['Jones SL'] = mie_data[7][2]
Mie_Theory_DF['Greenslade SL'] = mie_data[8][2]
Mie_Theory_DF['Gienger SL'] = mie_data[9][2]

Mie_Theory_DF['Matheson SR'] = mie_data[0][3]
Mie_Theory_DF['Bateman SR'] = mie_data[1][3]
Mie_Theory_DF['Nikalov SR'] = mie_data[2][3]
Mie_Theory_DF['Ma SR'] = mie_data[3][3]
Mie_Theory_DF['Sultanova SR'] = mie_data[4][3]
Mie_Theory_DF['Kasarova SR'] = mie_data[5][3]
Mie_Theory_DF['Miles SR'] = mie_data[6][3]
Mie_Theory_DF['Jones SR'] = mie_data[7][3]
Mie_Theory_DF['Greenslade SR'] = mie_data[8][3]
Mie_Theory_DF['Gienger SR'] = mie_data[9][3]

Mie_Theory_DF['Matheson SU'] = mie_data[0][4]
Mie_Theory_DF['Bateman SU'] = mie_data[1][4]
Mie_Theory_DF['Nikalov SU'] = mie_data[2][4]
Mie_Theory_DF['Ma SU'] = mie_data[3][4]
Mie_Theory_DF['Sultanova SU'] = mie_data[4][4]
Mie_Theory_DF['Kasarova SU'] = mie_data[5][4]
Mie_Theory_DF['Miles SU'] = mie_data[6][4]
Mie_Theory_DF['Jones SU'] = mie_data[7][4]
Mie_Theory_DF['Greenslade SU'] = mie_data[8][4]
Mie_Theory_DF['Gienger SU'] = mie_data[9][4]
Mie_Theory_DF.to_csv(Save_Directory + 'Mie_Theory_DF.txt', sep=',')


theta_max_min_avg = np.mean([[theta_mie[x] for x in mie_max_min_array[0]], [theta_mie[x] for x in mie_max_min_array[1]], [theta_mie[x] for x in mie_max_min_array[2]], [theta_mie[x] for x in mie_max_min_array[3]], [theta_mie[x] for x in mie_max_min_array[4]], [theta_mie[x] for x in mie_max_min_array[5]], [theta_mie[x] for x in mie_max_min_array[6]], [theta_mie[x] for x in mie_max_min_array[7]], [theta_mie[x] for x in mie_max_min_array[8]], [theta_mie[x] for x in mie_max_min_array[9]]], axis=0)
pf_max_min_avg = np.mean([[Mie_Theory_DF['Matheson SL'][x] for x in mie_max_min_array[0]], [Mie_Theory_DF['Bateman SL'][x] for x in mie_max_min_array[1]], [Mie_Theory_DF['Nikalov SL'][x] for x in mie_max_min_array[2]], [Mie_Theory_DF['Ma SL'][x] for x in mie_max_min_array[3]], [Mie_Theory_DF['Sultanova SL'][x] for x in mie_max_min_array[4]], [Mie_Theory_DF['Kasarova SL'][x] for x in mie_max_min_array[5]], [Mie_Theory_DF['Miles SL'][x] for x in mie_max_min_array[6]], [Mie_Theory_DF['Jones SL'][x] for x in mie_max_min_array[7]], [Mie_Theory_DF['Greenslade SL'][x] for x in mie_max_min_array[8]], [Mie_Theory_DF['Gienger SL'][x] for x in mie_max_min_array[9]]], axis=0)
theta_max_min_std = np.std([[theta_mie[x] for x in mie_max_min_array[0]], [theta_mie[x] for x in mie_max_min_array[1]], [theta_mie[x] for x in mie_max_min_array[2]], [theta_mie[x] for x in mie_max_min_array[3]], [theta_mie[x] for x in mie_max_min_array[4]], [theta_mie[x] for x in mie_max_min_array[5]], [theta_mie[x] for x in mie_max_min_array[6]], [theta_mie[x] for x in mie_max_min_array[7]], [theta_mie[x] for x in mie_max_min_array[8]], [theta_mie[x] for x in mie_max_min_array[9]]], axis=0)
pf_max_min_std = np.std([[Mie_Theory_DF['Matheson SL'][x] for x in mie_max_min_array[0]], [Mie_Theory_DF['Bateman SL'][x] for x in mie_max_min_array[1]], [Mie_Theory_DF['Nikalov SL'][x] for x in mie_max_min_array[2]], [Mie_Theory_DF['Ma SL'][x] for x in mie_max_min_array[3]], [Mie_Theory_DF['Sultanova SL'][x] for x in mie_max_min_array[4]], [Mie_Theory_DF['Kasarova SL'][x] for x in mie_max_min_array[5]], [Mie_Theory_DF['Miles SL'][x] for x in mie_max_min_array[6]], [Mie_Theory_DF['Jones SL'][x] for x in mie_max_min_array[7]], [Mie_Theory_DF['Greenslade SL'][x] for x in mie_max_min_array[8]], [Mie_Theory_DF['Gienger SL'][x] for x in mie_max_min_array[9]]], axis=0)
theta_uncertainty = [theta_max_min_std[x] / theta_max_min_avg[x] * 100 for x in range(len(theta_max_min_avg))]


# figure with local max and minima and associated errors when averaging all theory theta
f0a, ax0a = plt.subplots(figsize=(12, 7))
ax0a.semilogy(theta_mie, Mie_Theory_DF['Matheson SL'], label=groups[0])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[0]], [Mie_Theory_DF['Matheson SL'][x] for x in mie_max_min_array[0]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Bateman SL'], label=groups[1])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[1]], [Mie_Theory_DF['Bateman SL'][x] for x in mie_max_min_array[1]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Nikalov SL'], label=groups[2])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[2]], [Mie_Theory_DF['Nikalov SL'][x] for x in mie_max_min_array[2]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Ma SL'], label=groups[3])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[3]], [ Mie_Theory_DF['Ma SL'][x] for x in mie_max_min_array[3]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Sultanova SL'], label=groups[4])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[4]], [Mie_Theory_DF['Sultanova SL'][x] for x in mie_max_min_array[4]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Kasarova SL'], label=groups[5])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[5]], [Mie_Theory_DF['Kasarova SL'][x] for x in mie_max_min_array[5]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Miles SL'], label=groups[6])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[6]], [Mie_Theory_DF['Miles SL'][x] for x in mie_max_min_array[6]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Jones SL'], label=groups[7])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[7]], [Mie_Theory_DF['Jones SL'][x] for x in mie_max_min_array[7]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Greenslade SL'], label=groups[8])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[8]], [Mie_Theory_DF['Greenslade SL'][x] for x in mie_max_min_array[8]], color='black', marker='*', ms=3, ls=' ')
ax0a.semilogy(theta_mie, Mie_Theory_DF['Gienger SL'], label=groups[9])
ax0a.semilogy([theta_mie[x] for x in mie_max_min_array[9]], [Mie_Theory_DF['Gienger SL'][x] for x in mie_max_min_array[9]], color='black', marker='*', ms=3, ls=' ')
ax0a.errorbar(theta_max_min_avg, pf_max_min_avg, xerr=2*np.array(theta_max_min_std), yerr=2*np.array(pf_max_min_std), color='black', ls=' ', capsize=2.5, label='Maxima & Minima\n with Error in \u03b8')
ax0a.set_title('PSL SL Phase Functions Calculated from Various Refractive Index\n Values Given in the Literature')
ax0a.set_xlabel('\u03b8')
ax0a.set_ylabel('Intensity')
ax0a.grid(True)
ax0a.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Mie_SL.png', format='png')
plt.savefig(Save_Directory + 'Mie_SL.pdf', format='pdf')
plt.show()


f0b, ax0b = plt.subplots(figsize=(12, 7))
ax0b.semilogy(theta_mie, Mie_Theory_DF['Matheson SR'], label=groups[0])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Bateman SR'], label=groups[1])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Nikalov SR'], label=groups[2])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Ma SR'], label=groups[3])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Sultanova SR'], label=groups[4])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Kasarova SR'], label=groups[5])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Miles SR'], label=groups[6])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Jones SR'], label=groups[7])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Greenslade SR'], label=groups[8])
ax0b.semilogy(theta_mie, Mie_Theory_DF['Gienger SR'], label=groups[9])
ax0b.set_title('PSL SR Phase Functions Calculated from Various Refractive Index\n Values Given in the Literature')
ax0b.set_xlabel('\u03b8')
ax0b.set_ylabel('Intensity')
ax0b.grid(True)
ax0b.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Mie_SR.png', format='png')
plt.savefig(Save_Directory + 'Mie_SR.pdf', format='pdf')
plt.show()


f0c, ax0c = plt.subplots(figsize=(12, 7))
ax0c.semilogy(theta_mie, Mie_Theory_DF['Matheson SU'], label=groups[0])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Bateman SU'], label=groups[1])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Nikalov SU'], label=groups[2])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Ma SU'], label=groups[3])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Sultanova SU'], label=groups[4])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Kasarova SU'], label=groups[5])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Miles SU'], label=groups[6])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Jones SU'], label=groups[7])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Greenslade SU'], label=groups[8])
ax0c.semilogy(theta_mie, Mie_Theory_DF['Gienger SU'], label=groups[9])
ax0c.set_title('PSL SU Phase Functions Calculated from Various Refractive Index\n Values Given in the Literature')
ax0c.set_xlabel('\u03b8')
ax0c.set_ylabel('Intensity')
ax0c.grid(True)
ax0c.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Mie_SU.png', format='png')
plt.savefig(Save_Directory + 'Mie_SU.pdf', format='pdf')
plt.show()
