'''
Austen K. Scruggs
10-31-2018
Description: Mie theory script on a gaussian size distribution that computes the scattering
diagram for Parallel, Perpendicular, and Unpolarized Light!

PSLs:
Mean    Mean Uncertainty     Size Dist Sigma
600nm     9nm                     10.0nm
800nm     14nm                    5.6nm
903nm     12nm                    4.1nm
'''

import numpy as np
import PyMieScatt as ps
from math import pi, sqrt, log10
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.signal import savgol_filter

Save_Directory = '/home/austen/Documents/04-16-2019 Analysis'
Data_Directory = '/home/austen/Documents/04-16-2019 Analysis/SD_Particle_600nmPSL.txt'

# import experimental data
#Data = pd.read_csv(Data_Directory, sep=',', header=0)
# Particle diameter, geometric mean of the particle diameter
d = 600
# particle size standard deviation
sigma_s = 10.0
# define Gaussian function
def Gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# size distribution plot
size_axis = np.arange(d - (sigma_s * 3), d + (sigma_s * 3), 1)
Gaussian_Data = Gaussian(size_axis, mu=d, sigma=sigma_s)
#print(sp.integrate.simps(Gaussian(size_axis, mu=d, sigma=sigma_g), size_axis, dx=1))
f, ax = plt.subplots(figsize=(6, 6))
ax.plot(size_axis, Gaussian_Data, 'b-', label='Gaussian Dist. \u03bc=' + str(d) + ', \u03c3=' + str(sigma_s))
ax.set_xlabel('particle diameter (nm)')
ax.set_ylabel('Normalized $dN/Log_{10}(D)$')
ax.set_title('Distributions Used for Mie Theory Calculations')
ax.grid(True)
plt.legend(loc=1)
plt.savefig(Save_Directory + 'Mie_Distributions.pdf', format='pdf')
plt.show()

'''
We will used the normalized distribution intensities at each particle diameter as weights 
when we sum all our scattering diagrams, here we just collect the distribution information in array of arrays
'''

Weights_Gaussian = []
size_array = []
weights_array = []
for counter, element in enumerate(range(len(Gaussian_Data))):
    Weights_Gaussian.append([size_axis[counter], Gaussian_Data[counter]])
    size_array.append(size_axis[counter])
    weights_array.append(Gaussian_Data[counter])

# wavelength in centimeters
w_c = 663E-7

# wavelength in microns
w_u = .663

# wavelength in nanometers
w_n = 663

# wavelength in angstroms
w_a = 6630

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

# n of refractive index for PSL as cauchy equation
n_cauchy0 = A0 + (B0 / w_u ** 2) + (C0 / (w_u ** 4))
n_cauchy1 = A1 + (B1 / w_c ** 2)
n_cauchy2 = np.sqrt(A2 + (B2 * w_u ** 2) + (C2 / w_u ** 2) + (D2 / (w_u ** 4)) + (E2 / (w_u ** 6)) + (F2 / (w_u ** 8)))
n_cauchy3 = A3 + (B3 / w_u ** 2) + (C3 / (w_u ** 4))
n_cauchy4 = np.sqrt(A4 + (B4 * w_u ** 2) + (C4 / w_u ** 2) + (D4 / (w_u ** 4)) + (E4 / (w_u ** 6)) + (F4 / (w_u ** 8)))
n_cauchy5 = np.sqrt(A5 + (B5 * w_u ** 2) + (C5 / w_u ** 2) + (D5 / (w_u ** 4)) + (E5 / (w_u ** 6)) + (F5 / (w_u ** 8)))
n_cauchy6 = A6 + (B6 / w_u ** 2) + (C6 / (w_u ** 4))
n_cauchy7 = A7 + (B7 / w_n ** 2) + (C7 / (w_n ** 4))
n_cauchy8 = A8 + (B8 / w_u ** 2) + (C8 / (w_u ** 4))
n_sellmeier9 = np.sqrt(1 + ((B9 * w_n ** 2)/(w_n ** 2 - wav9 ** 2)))

# basic statistics
n_all_groups = [n_cauchy0, n_cauchy1, n_cauchy2, n_cauchy3, n_cauchy4, n_cauchy5, n_cauchy6, n_cauchy7, n_cauchy8, n_sellmeier9]
n_mean = np.mean(n_all_groups)
n_percentiles = np.percentile(n_all_groups, [0, 25, 50, 75, 100])


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

fig2, ax2 = plt.subplots(1, 3, figsize=(20, 7))
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
ax2[1].plot(663, n_cauchy0, marker='o', ms=3, ls='', label='Matheson et. al. 1952')
ax2[1].plot(663, n_cauchy1, marker='o', ms=3, ls='', label='Bateman et. al. 1959')
ax2[1].plot(663, n_cauchy2, marker='o', ms=3, ls='', label='Nikalov et. al. 2000')
ax2[1].plot(663, n_cauchy3, marker='o', ms=3, ls='', label='Ma et. al. 2003')
ax2[1].plot(663, n_cauchy4, marker='o', ms=3, ls='', label='Sultanova et. al. 2003')
ax2[1].plot(663, n_cauchy5, marker='o', ms=3, ls='', label='Kasarova et. al. 2006')
ax2[1].plot(663, n_cauchy6, marker='o', ms=3, ls='', label='Miles et. al. 2010')
ax2[1].plot(663, n_cauchy7, marker='o', ms=3, ls='', label='Jones et. al. 2013')
ax2[1].plot(663, n_cauchy8, marker='o', ms=3, ls='', label='Greenslade et. al. 2017')
ax2[1].plot(663, n_sellmeier9, marker='o', ms=3, ls='', label='Gienger et. al. 2017')
ax2[1].set_title('All the Groups Values at 663nm')
ax2[1].set_xlabel('Wavelength (nm)')
ax2[1].set_ylabel('n')
ax2[1].grid(True)
ax2[1].legend(loc=1)
ax2[2].boxplot(n_all_groups)
ax2[2].set_xticklabels(['Wavelength 663nm'])
ax2[2].set_title('Box Plot Statistics of All the Groups Values at 663nm')
ax2[2].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'N_vs_Wav.pdf', format='pdf')
plt.show()


# k, imaginary part of RI
k0 = .0003j
k1 = .0003j
k2 = .0003j
k3 = .0003j
k4 = .0003j
k5 = .0003j
k6 = .0003j
k7 = .0003j
k8 = .0003j
k9 = .0003j

# create RI
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

# pre-allocate 2darrays
theta_2darray_0 = []
SL_2darray_0 = []
SR_2darray_0 = []
SU_2darray_0 = []

theta_2darray_1 = []
SL_2darray_1 = []
SR_2darray_1 = []
SU_2darray_1 = []

theta_2darray_2 = []
SL_2darray_2 = []
SR_2darray_2 = []
SU_2darray_2 = []

theta_2darray_3 = []
SL_2darray_3 = []
SR_2darray_3 = []
SU_2darray_3 = []

theta_2darray_4 = []
SL_2darray_4 = []
SR_2darray_4 = []
SU_2darray_4 = []

theta_2darray_5 = []
SL_2darray_5 = []
SR_2darray_5 = []
SU_2darray_5 = []

theta_2darray_6 = []
SL_2darray_6 = []
SR_2darray_6 = []
SU_2darray_6 = []

theta_2darray_7 = []
SL_2darray_7 = []
SR_2darray_7 = []
SU_2darray_7 = []

theta_2darray_8 = []
SL_2darray_8 = []
SR_2darray_8 = []
SU_2darray_8 = []

theta_2darray_9 = []
SL_2darray_9 = []
SR_2darray_9 = []
SU_2darray_9 = []

# note everything is in theta space, if you want q space read online documentation
for element in size_array:
    theta_0, SL_0, SR_0, SU_0 = ps.ScatteringFunction(m0, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_0.append(theta_0)
    SL_2darray_0.append(SL_0)
    SR_2darray_0.append(SR_0)
    SU_2darray_0.append(SU_0)
    theta_1, SL_1, SR_1, SU_1 = ps.ScatteringFunction(m1, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_1.append(theta_1)
    SL_2darray_1.append(SL_1)
    SR_2darray_1.append(SR_1)
    SU_2darray_1.append(SU_1)
    theta_2, SL_2, SR_2, SU_2 = ps.ScatteringFunction(m2, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_2.append(theta_2)
    SL_2darray_2.append(SL_2)
    SR_2darray_2.append(SR_2)
    SU_2darray_2.append(SU_2)
    theta_3, SL_3, SR_3, SU_3 = ps.ScatteringFunction(m3, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_3.append(theta_3)
    SL_2darray_3.append(SL_3)
    SR_2darray_3.append(SR_3)
    SU_2darray_3.append(SU_3)
    theta_4, SL_4, SR_4, SU_4 = ps.ScatteringFunction(m4, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_4.append(theta_4)
    SL_2darray_4.append(SL_4)
    SR_2darray_4.append(SR_4)
    SU_2darray_4.append(SU_4)
    theta_5, SL_5, SR_5, SU_5 = ps.ScatteringFunction(m5, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_5.append(theta_5)
    SL_2darray_5.append(SL_5)
    SR_2darray_5.append(SR_5)
    SU_2darray_5.append(SU_5)
    theta_6, SL_6, SR_6, SU_6 = ps.ScatteringFunction(m6, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_6.append(theta_6)
    SL_2darray_6.append(SL_6)
    SR_2darray_6.append(SR_6)
    SU_2darray_6.append(SU_6)
    theta_7, SL_7, SR_7, SU_7 = ps.ScatteringFunction(m7, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_7.append(theta_7)
    SL_2darray_7.append(SL_7)
    SR_2darray_7.append(SR_7)
    SU_2darray_7.append(SU_7)
    theta_8, SL_8, SR_8, SU_8 = ps.ScatteringFunction(m8, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_8.append(theta_8)
    SL_2darray_8.append(SL_8)
    SR_2darray_8.append(SR_8)
    SU_2darray_8.append(SU_8)
    theta_9, SL_9, SR_9, SU_9 = ps.ScatteringFunction(m9, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_9.append(theta_9)
    SL_2darray_9.append(SL_9)
    SR_2darray_9.append(SR_9)
    SU_2darray_9.append(SU_9)

SL0 = np.average(SL_2darray_0, axis=0, weights=weights_array)
SR0 = np.average(SR_2darray_0, axis=0, weights=weights_array)
SU0 = np.average(SU_2darray_0, axis=0, weights=weights_array)

SL1 = np.average(SL_2darray_1, axis=0, weights=weights_array)
SR1 = np.average(SR_2darray_1, axis=0, weights=weights_array)
SU1 = np.average(SU_2darray_1, axis=0, weights=weights_array)

SL2 = np.average(SL_2darray_2, axis=0, weights=weights_array)
SR2 = np.average(SR_2darray_2, axis=0, weights=weights_array)
SU2 = np.average(SU_2darray_2, axis=0, weights=weights_array)

SL3 = np.average(SL_2darray_3, axis=0, weights=weights_array)
SR3 = np.average(SR_2darray_3, axis=0, weights=weights_array)
SU3 = np.average(SU_2darray_3, axis=0, weights=weights_array)

SL4 = np.average(SL_2darray_4, axis=0, weights=weights_array)
SR4 = np.average(SR_2darray_4, axis=0, weights=weights_array)
SU4 = np.average(SU_2darray_4, axis=0, weights=weights_array)

SL5 = np.average(SL_2darray_5, axis=0, weights=weights_array)
SR5 = np.average(SR_2darray_5, axis=0, weights=weights_array)
SU5 = np.average(SU_2darray_5, axis=0, weights=weights_array)

SL6 = np.average(SL_2darray_6, axis=0, weights=weights_array)
SR6 = np.average(SR_2darray_6, axis=0, weights=weights_array)
SU6 = np.average(SU_2darray_6, axis=0, weights=weights_array)

SL7 = np.average(SL_2darray_7, axis=0, weights=weights_array)
SR7 = np.average(SR_2darray_7, axis=0, weights=weights_array)
SU7 = np.average(SU_2darray_7, axis=0, weights=weights_array)

SL8 = np.average(SL_2darray_8, axis=0, weights=weights_array)
SR8 = np.average(SR_2darray_8, axis=0, weights=weights_array)
SU8 = np.average(SU_2darray_8, axis=0, weights=weights_array)

SL9 = np.average(SL_2darray_9, axis=0, weights=weights_array)
SR9 = np.average(SR_2darray_9, axis=0, weights=weights_array)
SU9 = np.average(SU_2darray_9, axis=0, weights=weights_array)

# closing any open plots
plt.close('all')

# For plotting data use the parameters below and remove comment hash (#) in the plot code below
#slope = 0.21116737541023334
#intercept = -47.97208663718679
#intensity_scalar = .0036050

'''
# scattering diagram figure 0
fig0, ax0 = plt.subplots(figsize=(12, 7))
pt0, = ax0.semilogy(theta_1, SL1, 'b', ls='dashdot', lw=1, label="Perpendicular Polarization Ma 2003")
pt1, = ax0.semilogy(theta_1, SR1, 'r', ls='dashed', lw=1, label="Parallel Polarization Ma 2003")
pt2, = ax0.semilogy(theta_1, SU1, 'g', ls=':', lw=1, label="Unpolarized/Circular Ma 2003")
#pt3, = ax0.semilogy((np.asarray(Data['Columns']) * slope) + intercept, (np.asarray(Data['Sample Intensity']) - np.asarray(Data['Nitrogen Intensity'])) * intensity_scalar, 'k', ls='-', lw=1, label='PSL 600nm Cal. Data')
#ax0.tick_params(which='both', direction='in')
ax0.set_xlabel("\n \u03b8", fontsize=16)
ax0.set_ylabel(r"Intensity ($\mathregular{|S|^2}$)",fontsize=16,labelpad=10)
fig0.suptitle("Scattering Intensity Functions \n", fontsize=18)
#ax0b = plt.twiny()
#pt4, = ax0b.semilogy(Data['Columns'], np.asarray(Data['Sample Intensity']) - np.asarray(Data['Nitrogen Intensity']), 'c', ls='-', lw=1, label='Circularly Pol. Meas.')
#ax0b.set_xlabel('Profile Numbers', color='c', fontsize=16)
#ax0b.tick_params(axis='x', labelcolor='c')
pt = [pt0, pt1, pt2]
#ax0b.legend(pt, [pt_.get_label() for pt_ in pt], fontsize=14, loc='best')
ax0.legend(pt, [pt_.get_label() for pt_ in pt], fontsize=14, loc='best')
ax0.xaxis.set_major_locator(plt.MaxNLocator(10))
ax0.grid(True)
#plt.gcf().subplots_adjust(bottom=0.15)
plt.subplots_adjust(top=0.85)
# this must be done so that the figure title (suptitle) is spaced apart from the top x axis!
plt.tight_layout(rect=[0.01, 0.05, 0.915, 0.95])
plt.savefig(Save_Directory + 'MieTheory.pdf', format='pdf')
plt.show()
'''

fig3, ax3 = plt.subplots(1, 3, figsize=(20, 7))
ax3[0].semilogy(theta_0, SL0, ls='-', lw=1, label="Perp. Pol. Matheson et al 1952")
ax3[0].semilogy(theta_1, SL1, ls='-', lw=1, label="Perp. Pol. Bateman et al 1959")
ax3[0].semilogy(theta_2, SL2, ls='-', lw=1, label="Perp. Pol. Nikalov et al 2000")
ax3[0].semilogy(theta_3, SL3, ls='-', lw=1, label="Perp. Pol. Ma et al 2003")
ax3[0].semilogy(theta_4, SL4, ls='-', lw=1, label="Perp. Pol. Sultanova et al 2003")
ax3[0].semilogy(theta_5, SL5, ls='-', lw=1, label="Perp. Pol. Kasarova et al 2006")
ax3[0].semilogy(theta_6, SL6, ls='-', lw=1, label="Perp. Pol. Miles et al 2010")
ax3[0].semilogy(theta_7, SL7, ls='-', lw=1, label="Perp. Pol. Jones et al 2013")
ax3[0].semilogy(theta_8, SL8, ls='-', lw=1, label="Perp. Pol. Greensalde et al 2017")
ax3[0].semilogy(theta_9, SL9, ls='-', lw=1, label="Perp. Pol. Geinger et al 2017")
ax3[0].set_xlabel("ϴ", fontsize=16)
ax3[0].set_ylabel(r"Intensity ($\mathregular{|S|^2}$)",fontsize=16,labelpad=10)
ax3[0].set_title('Scattering Diagram \n Perpendicular Polarized Light', fontsize=18)
ax3[0].grid(True)
ax3[0].legend(loc=1)
ax3[1].semilogy(theta_0, SR0, ls='-', lw=1, label="Par. Pol. Matheson et al 1952")
ax3[1].semilogy(theta_1, SR1, ls='-', lw=1, label="Par. Pol. Bateman et al 1959")
ax3[1].semilogy(theta_2, SR2, ls='-', lw=1, label="Par. Pol. Nikalov et al 2000")
ax3[1].semilogy(theta_3, SR3, ls='-', lw=1, label="Par. Pol. Ma et al 2003")
ax3[1].semilogy(theta_4, SR4, ls='-', lw=1, label="Par. Pol. Sultanova et al 2003")
ax3[1].semilogy(theta_5, SR5, ls='-', lw=1, label="Par. Pol. Kasarova et al 2006")
ax3[1].semilogy(theta_6, SR6, ls='-', lw=1, label="Par. Pol. Miles et al 2010")
ax3[1].semilogy(theta_7, SR7, ls='-', lw=1, label="Par. Pol. Jones et al 2013")
ax3[1].semilogy(theta_8, SR8, ls='-', lw=1, label="Par. Pol. Greensalde et al 2017")
ax3[1].semilogy(theta_9, SR9, ls='-', lw=1, label="Par. Pol. Geinger et al 2017")
ax3[1].set_xlabel("ϴ", fontsize=16)
ax3[1].set_ylabel(r"Intensity ($\mathregular{|S|^2}$)",fontsize=16,labelpad=10)
ax3[1].set_title('Scattering Diagram \n Parallel Polarized Light', fontsize=18)
ax3[1].grid(True)
ax3[1].legend(loc=1)
ax3[2].semilogy(theta_0, SU0, ls='-', lw=1, label="Un./Cir. Pol. Matheson et al 1952")
ax3[2].semilogy(theta_1, SU1, ls='-', lw=1, label="Un./Cir. Pol. Bateman et al 1959")
ax3[2].semilogy(theta_2, SU2, ls='-', lw=1, label="Un./Cir. Pol. Nikalov et al 2000")
ax3[2].semilogy(theta_3, SU3, ls='-', lw=1, label="Un./Cir. Pol. Ma et al 2003")
ax3[2].semilogy(theta_4, SU4, ls='-', lw=1, label="Un./Cir. Pol. Sultanova et al 2003")
ax3[2].semilogy(theta_5, SU5, ls='-', lw=1, label="Un./Cir. Pol. Kasarova et al 2006")
ax3[2].semilogy(theta_6, SU6, ls='-', lw=1, label="Un./Cir. Pol. Miles et al 2010")
ax3[2].semilogy(theta_7, SU7, ls='-', lw=1, label="Un./Cir. Pol. Jones et al 2013")
ax3[2].semilogy(theta_8, SU8, ls='-', lw=1, label="Un./Cir. Pol. Greenslade et al 2017")
ax3[2].semilogy(theta_9, SU9, ls='-', lw=1, label="Un./Cir. Pol. Geinger et al 2017")
#ax3[2].semilogy(Data['PN to Angle'], intensity_scalar * np.array(Data['Exp Smoothed Intensity']), color='black', ls='-', lw=3, label="Un/Cir. Pol. Experiment")
#Val_Intensity = np.array(Data['Sample Intensity'])
#Val_PN = Data['Columns']
#Val_Smoothed_Intensity = savgol_filter(Val_Intensity, window_length=151, polyorder=2, deriv=0)
#Val_Pchip = pchip_interpolate(Val_PN, Val_Smoothed_Intensity, Val_PN, der=0, axis=0)
#ax3[2].semilogy((slope * Val_PN) + intercept, intensity_scalar * Val_Pchip, color='black', ls='-', lw=3, label="Un/Cir. Pol. Experiment")
ax3[2].set_xlabel("ϴ", fontsize=16)
ax3[2].set_ylabel(r"Intensity ($\mathregular{|S|^2}$)",fontsize=16,labelpad=10)
ax3[2].set_title('Scattering Diagram \n UnPolarized & Circularly Polarized Light', fontsize=18)
ax3[2].grid(True)
ax3[2].legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Mie_PhaseFunctions.pdf', format='pdf')
plt.show()

Mie_Data = pd.DataFrame()
Mie_Data['Theta Matheson 1952'] = theta_0
Mie_Data['SL Matheson 1952'] = SL0
Mie_Data['SR Matheson 1952'] = SR0
Mie_Data['SU Matheson 1952'] = SU0
Mie_Data['Theta Bateman 1959'] = theta_1
Mie_Data['SL Bateman 1959'] = SL1
Mie_Data['SR Bateman 1959'] = SR1
Mie_Data['SU Bateman 1959'] = SU1
Mie_Data['Theta Nikalov 2000'] = theta_2
Mie_Data['SL Nikalov 2000'] = SL2
Mie_Data['SR Nikalov 2000'] = SR2
Mie_Data['SU Nikalov 2000'] = SU2
Mie_Data['Theta Ma 2003'] = theta_3
Mie_Data['SL Ma 2003'] = SL3
Mie_Data['SR Ma 2003'] = SR3
Mie_Data['SU Ma 2003'] = SU3
Mie_Data['Theta Sultanova 2003'] = theta_4
Mie_Data['SL Sultanova 2003'] = SL4
Mie_Data['SR Sultanova 2003'] = SR4
Mie_Data['SU Sultanova 2003'] = SU4
Mie_Data['Theta Kasarova 2006'] = theta_5
Mie_Data['SL Kasarova 2006'] = SL5
Mie_Data['SR Kasarova 2006'] = SR5
Mie_Data['SU Kasarova 2006'] = SU5
Mie_Data['Theta Miles 2010'] = theta_6
Mie_Data['SL Miles 2010'] = SL6
Mie_Data['SR Miles 2010'] = SR6
Mie_Data['SU Miles 2010'] = SU6
Mie_Data['Theta Jones 2013'] = theta_7
Mie_Data['SL Jones 2013'] = SL7
Mie_Data['SR Jones 2013'] = SR7
Mie_Data['SU Jones 2013'] = SU7
Mie_Data['Theta Greenslade 2017'] = theta_8
Mie_Data['SL Greenslade 2017'] = SL8
Mie_Data['SR Greenslade 2017'] = SR8
Mie_Data['SU Greenslade 2017'] = SU8
Mie_Data['Theta Gienger 2017'] = theta_9
Mie_Data['SL Gienger 2017'] = SL9
Mie_Data['SR Gienger 2017'] = SR9
Mie_Data['SU Gienger 2017'] = SU9



Mie_Data.to_csv(Save_Directory + '/PSL600nm_MieTheory.txt')
