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
from scipy.stats import chisquare
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

Save_Directory = '/home/austen/Documents/'
Data_Directory = '/home/austen/Documents/01-23-2019_Analysis/SD_Offline_800.txt'

# import experimental data
#Data = pd.read_csv(Data_Directory, sep=',', header=0)
# Particle diameter, geometric mean of the particle diameter
d = 800
# particle size standard deviation
sigma_s = 5.6
# define Gaussian function
def Gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# size distribution plot
size_axis = np.arange(d - (sigma_s * 3), d + (sigma_s * 3), 1)
Gaussian_Data = Gaussian(size_axis, mu=d, sigma=sigma_s)
#print(sp.integrate.simps(Gaussian(size_axis, mu=d, sigma=sigma_g), size_axis, dx=1))
f1, ax1 = plt.subplots(figsize=(6, 6))
ax1.plot(size_axis, Gaussian_Data, 'b-', label='Gaussian Dist. \u03bc=' + str(d) + ', \u03c3=' + str(sigma_s))
ax1.set_xlabel('particle diameter (nm)')
ax1.set_ylabel('Normalized $dN/Log_{10}(D)$')
ax1.set_title('Distributions Used for Mie Theory Calculations')
ax1.grid(True)
plt.legend(loc=1)
plt.savefig(Save_Directory + 'Mie_Distributions.pdf', format='pdf')
plt.savefig(Save_Directory + 'Mie_Distributions.png', format='png')
#plt.show()
plt.close()

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

# n of refractive index for PSL as cauchy equation at a specific wavelength
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
plt.savefig(Save_Directory + 'N_vs_Wav.png', format='png')
#plt.show()
plt.close()

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
for size in size_array:
    theta_0, SL_0, SR_0, SU_0 = ps.ScatteringFunction(m0, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_0.append(theta_0)
    SL_2darray_0.append(SL_0)
    SR_2darray_0.append(SR_0)
    SU_2darray_0.append(SU_0)
    theta_1, SL_1, SR_1, SU_1 = ps.ScatteringFunction(m1, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_1.append(theta_1)
    SL_2darray_1.append(SL_1)
    SR_2darray_1.append(SR_1)
    SU_2darray_1.append(SU_1)
    theta_2, SL_2, SR_2, SU_2 = ps.ScatteringFunction(m2, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_2.append(theta_2)
    SL_2darray_2.append(SL_2)
    SR_2darray_2.append(SR_2)
    SU_2darray_2.append(SU_2)
    theta_3, SL_3, SR_3, SU_3 = ps.ScatteringFunction(m3, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_3.append(theta_3)
    SL_2darray_3.append(SL_3)
    SR_2darray_3.append(SR_3)
    SU_2darray_3.append(SU_3)
    theta_4, SL_4, SR_4, SU_4 = ps.ScatteringFunction(m4, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_4.append(theta_4)
    SL_2darray_4.append(SL_4)
    SR_2darray_4.append(SR_4)
    SU_2darray_4.append(SU_4)
    theta_5, SL_5, SR_5, SU_5 = ps.ScatteringFunction(m5, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_5.append(theta_5)
    SL_2darray_5.append(SL_5)
    SR_2darray_5.append(SR_5)
    SU_2darray_5.append(SU_5)
    theta_6, SL_6, SR_6, SU_6 = ps.ScatteringFunction(m6, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_6.append(theta_6)
    SL_2darray_6.append(SL_6)
    SR_2darray_6.append(SR_6)
    SU_2darray_6.append(SU_6)
    theta_7, SL_7, SR_7, SU_7 = ps.ScatteringFunction(m7, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_7.append(theta_7)
    SL_2darray_7.append(SL_7)
    SR_2darray_7.append(SR_7)
    SU_2darray_7.append(SU_7)
    theta_8, SL_8, SR_8, SU_8 = ps.ScatteringFunction(m8, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_2darray_8.append(theta_8)
    SL_2darray_8.append(SL_8)
    SR_2darray_8.append(SR_8)
    SU_2darray_8.append(SU_8)
    theta_9, SL_9, SR_9, SU_9 = ps.ScatteringFunction(m9, w_n, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
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
plt.savefig(Save_Directory + 'Mie_PhaseFunctions.png', format='png')
#plt.show()
plt.close()

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


Mie_Data.to_csv(Save_Directory + '/PSL800nm_MieTheory.txt')

# Now for the size distribution retrieval
experiment_data = pd.read_csv(Data_Directory)
experiment_SU = savgol_filter(np.array(experiment_data['Sample Intensity']) - np.array(experiment_data['Nitrogen Intensity']), window_length=151, polyorder=2, deriv=0)
experiment_PN = np.array(experiment_data['Columns'])
slope = 0.2045
intercept = -43.9764
experiment_Theta = (slope * experiment_PN) + intercept

# number of means and standard deviations looked at
num = 25
#number of points looked at on the distribution for weighted average
num2 = 25
# make size space and size sigma space
d_array = np.linspace(d-25.0, d+25.0, num)
print('particle distribution mean diameters evaluated: ', d_array)
sigma_array = np.linspace(1.0, 50.0, num)
print('particle distribution sigmas evaluated: ', sigma_array)

# looping body for Chi minimization to determine optimum size distribution
# the below is commented out, I will probably need to use it again if I build in multiple wavelengths
'''
# wavelength in centimeters
w_c = 663E-7
# wavelength in microns
w_u = .663
# wavelength in nanometers
w_n = 663
# wavelength in angstroms
w_a = 6630

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
popt_Nikalov, pcov_Nikalov = curve_fit(cauchy_6term, w_Nikalov_microns, n_RI_Nikalov, p0=[2.44675093, -1.011623E-3, 2.840749E-2, -3.761631E-4, 8.193491E-5, 9.055861E-4])
# print(popt_Nikalov)
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
'''
Chi_2darray_0 = []
Chi_2darray_1 = []
Chi_2darray_2 = []
Chi_2darray_3 = []
Chi_2darray_4 = []
Chi_2darray_5 = []
Chi_2darray_6 = []
Chi_2darray_7 = []
Chi_2darray_8 = []
Chi_2darray_9 = []
for counter0, s in enumerate(d_array):
    percent_done = (counter0 / len(d_array)) * 100
    print('calculation % complete: ', percent_done)
    for counter1, sigma in enumerate(sigma_array):
        W_G = []
        s_arr = []
        w_arr = []
        x = np.linspace(s-3*sigma, s-3*sigma, num2)
        G = Gaussian(x, s, sigma)

        # pre-allocate 2darrays
        theta_2darray_0_ret = []
        #SL_2darray_0_ret = []
        #SR_2darray_0_ret = []
        SU_2darray_0_ret = []

        theta_2darray_1_ret = []
        #SL_2darray_1_ret = []
        #SR_2darray_1_ret = []
        SU_2darray_1_ret = []

        theta_2darray_2_ret = []
        #SL_2darray_2_ret = []
        #SR_2darray_2_ret = []
        SU_2darray_2_ret = []

        theta_2darray_3_ret = []
        #SL_2darray_3_ret = []
        #SR_2darray_3_ret = []
        SU_2darray_3_ret = []

        theta_2darray_4_ret = []
        #SL_2darray_4_ret = []
        #SR_2darray_4_ret = []
        SU_2darray_4_ret = []

        theta_2darray_5_ret = []
        #SL_2darray_5_ret = []
        #SR_2darray_5_ret = []
        SU_2darray_5_ret = []

        theta_2darray_6_ret = []
        #SL_2darray_6_ret = []
        #SR_2darray_6_ret = []
        SU_2darray_6_ret = []

        theta_2darray_7_ret = []
        #SL_2darray_7_ret = []
        #SR_2darray_7_ret = []
        SU_2darray_7_ret = []

        theta_2darray_8_ret = []
        #SL_2darray_8_ret = []
        #SR_2darray_8_ret = []
        SU_2darray_8_ret = []

        theta_2darray_9_ret = []
        #SL_2darray_9_ret = []
        #SR_2darray_9_ret = []
        SU_2darray_9_ret = []

        # note everything is in theta space, if you want q space read online documentation
        for size_bin in x:
            theta_0_ret, SL_0_ret, SR_0_ret, SU_0_ret = ps.ScatteringFunction(m0, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_0_ret.append(theta_0_ret)
            #SL_2darray_0_ret.append(SL_0_ret)
            #SR_2darray_0_ret.append(SR_0_ret)
            SU_2darray_0_ret.append(SU_0_ret)

            theta_1_ret, SL_1_ret, SR_1_ret, SU_1_ret = ps.ScatteringFunction(m1, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_1_ret.append(theta_1_ret)
            #SL_2darray_1_ret.append(SL_1_ret)
            #SR_2darray_1_ret.append(SR_1_ret)
            SU_2darray_1_ret.append(SU_1_ret)

            theta_2_ret, SL_2_ret, SR_2_ret, SU_2_ret = ps.ScatteringFunction(m2, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_2_ret.append(theta_2_ret)
            #SL_2darray_2_ret.append(SL_2_ret)
            #SR_2darray_2_ret.append(SR_2_ret)
            SU_2darray_2_ret.append(SU_2_ret)

            theta_3_ret, SL_3_ret, SR_3_ret, SU_3_ret = ps.ScatteringFunction(m3, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_3_ret.append(theta_3_ret)
            #SL_2darray_3_ret.append(SL_3_ret)
            #SR_2darray_3_ret.append(SR_3_ret)
            SU_2darray_3_ret.append(SU_3_ret)

            theta_4_ret, SL_4_ret, SR_4_ret, SU_4_ret = ps.ScatteringFunction(m4, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_4_ret.append(theta_4_ret)
            #SL_2darray_4_ret.append(SL_4_ret)
            #SR_2darray_4_ret.append(SR_4_ret)
            SU_2darray_4_ret.append(SU_4_ret)

            theta_5_ret, SL_5_ret, SR_5_ret, SU_5_ret = ps.ScatteringFunction(m5, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_5_ret.append(theta_5_ret)
            #SL_2darray_5_ret.append(SL_5_ret)
            #SR_2darray_5_ret.append(SR_5_ret)
            SU_2darray_5_ret.append(SU_5_ret)

            theta_6_ret, SL_6_ret, SR_6_ret, SU_6_ret = ps.ScatteringFunction(m6, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_6_ret.append(theta_6_ret)
            #SL_2darray_6_ret.append(SL_6_ret)
            #SR_2darray_6_ret.append(SR_6_ret)
            SU_2darray_6_ret.append(SU_6_ret)

            theta_7_ret, SL_7_ret, SR_7_ret, SU_7_ret = ps.ScatteringFunction(m7, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_7_ret.append(theta_7_ret)
            #SL_2darray_7_ret.append(SL_7_ret)
            #SR_2darray_7_ret.append(SR_7_ret)
            SU_2darray_7_ret.append(SU_7_ret)

            theta_8_ret, SL_8_ret, SR_8_ret, SU_8_ret = ps.ScatteringFunction(m8, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_8_ret.append(theta_8_ret)
            #SL_2darray_8_ret.append(SL_8_ret)
            #SR_2darray_8_ret.append(SR_8_ret)
            SU_2darray_8_ret.append(SU_8_ret)

            theta_9_ret, SL_9_ret, SR_9_ret, SU_9_ret = ps.ScatteringFunction(m9, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            theta_2darray_9_ret.append(theta_9_ret)
            #SL_2darray_9_ret.append(SL_9_ret)
            #SR_2darray_9_ret.append(SR_9_ret)
            SU_2darray_9_ret.append(SU_9_ret)


        print('starting calculation on particle (diameter, sigma): ', [s, sigma])
        #SL0_ret = np.average(SL_2darray_0_ret, axis=0, weights=w_a)
        #SR0_ret = np.average(SR_2darray_0_ret, axis=0, weights=w_a)
        SU0_ret = np.average(SU_2darray_0_ret, axis=0, weights=G)
        Theta_0_ret = np.average(theta_2darray_0, axis=0)
        SU0_ret_pchip = pchip_interpolate(Theta_0_ret, SU0_ret, experiment_Theta, der=0, axis=0)
        scalar_0 = SU0_ret_pchip[np.argmin(SU0_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_0 = scalar_0 * experiment_SU
        Chi_SU_WA_0, Pval_SU_WA_0 = chisquare(f_obs=experiment_SU_scaled_0[250:len(experiment_SU_scaled_0)-1], f_exp=SU0_ret_pchip[250:len(experiment_SU_scaled_0)-1])
        Chi_2darray_0.append([m0, w_n, x, G, s, sigma, scalar_0, Chi_SU_WA_0, Pval_SU_WA_0, experiment_Theta, experiment_SU, Theta_0_ret, SU0_ret_pchip])

        #SL1_ret = np.average(SL_2darray_1_ret, axis=0, weights=w_a)
        #SR1_ret = np.average(SR_2darray_1_ret, axis=0, weights=w_a)
        SU1_ret = np.average(SU_2darray_1_ret, axis=0, weights=G)
        Theta_1_ret = np.average(theta_2darray_1, axis=0)
        SU1_ret_pchip = pchip_interpolate(Theta_1_ret, SU1_ret, experiment_Theta, der=0, axis=0)
        scalar_1 = SU1_ret_pchip[np.argmin(SU1_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_1 = scalar_1 * experiment_SU
        Chi_SU_WA_1, Pval_SU_WA_1 = chisquare(f_obs=experiment_SU_scaled_1[250:len(experiment_SU_scaled_1)-1], f_exp=SU1_ret_pchip[250:len(experiment_SU_scaled_1)-1])
        Chi_2darray_1.append([m1, w_n, x, G, s, sigma, scalar_1, Chi_SU_WA_1, Pval_SU_WA_1, experiment_Theta, experiment_SU, Theta_1_ret, SU1_ret_pchip])

        #SL2_ret = np.average(SL_2darray_2_ret, axis=0, weights=w_a)
        #SR2_ret = np.average(SR_2darray_2_ret, axis=0, weights=w_a)
        SU2_ret = np.average(SU_2darray_2_ret, axis=0, weights=G)
        Theta_2_ret = np.average(theta_2darray_2, axis=0)
        SU2_ret_pchip = pchip_interpolate(Theta_2_ret, SU2_ret, experiment_Theta, der=0, axis=0)
        scalar_2 = SU2_ret_pchip[np.argmin(SU2_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_2 = scalar_2 * experiment_SU
        Chi_SU_WA_2, Pval_SU_WA_2 = chisquare(f_obs=experiment_SU_scaled_2[250:len(experiment_SU_scaled_2)-1], f_exp=SU2_ret_pchip[250:len(experiment_SU_scaled_2)-1])
        Chi_2darray_2.append([m2, w_n, x, G, s, sigma, scalar_2, Chi_SU_WA_2, Pval_SU_WA_2, experiment_Theta, experiment_SU, Theta_2_ret, SU2_ret_pchip])

        #SL3_ret = np.average(SL_2darray_3_ret, axis=0, weights=w_a)
        #SR3_ret = np.average(SR_2darray_3_ret, axis=0, weights=w_a)
        SU3_ret = np.average(SU_2darray_3_ret, axis=0, weights=G)
        Theta_3_ret = np.average(theta_2darray_3, axis=0)
        SU3_ret_pchip = pchip_interpolate(Theta_3_ret, SU3_ret, experiment_Theta, der=0, axis=0)
        scalar_3 = SU3_ret_pchip[np.argmin(SU3_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_3 = scalar_3 * experiment_SU
        Chi_SU_WA_3, Pval_SU_WA_3 = chisquare(f_obs=experiment_SU_scaled_3[250:len(experiment_SU_scaled_3)-1], f_exp=SU3_ret_pchip[250:len(experiment_SU_scaled_3)-1])
        Chi_2darray_3.append([m3, w_n, x, G, s, sigma, scalar_3, Chi_SU_WA_3, Pval_SU_WA_3, experiment_Theta, experiment_SU, Theta_3_ret, SU3_ret_pchip])

        #SL4_ret = np.average(SL_2darray_4_ret, axis=0, weights=w_a)
        #SR4_ret = np.average(SR_2darray_4_ret, axis=0, weights=w_a)
        SU4_ret = np.average(SU_2darray_4_ret, axis=0, weights=G)
        Theta_4_ret = np.average(theta_2darray_4, axis=0)
        SU4_ret_pchip = pchip_interpolate(Theta_4_ret, SU4_ret, experiment_Theta, der=0, axis=0)
        scalar_4 = SU4_ret_pchip[np.argmin(SU4_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_4 = scalar_4 * experiment_SU
        Chi_SU_WA_4, Pval_SU_WA_4 = chisquare(f_obs=experiment_SU_scaled_4[250:len(experiment_SU_scaled_4)-1], f_exp=SU4_ret_pchip[250:len(experiment_SU_scaled_4)-1])
        Chi_2darray_4.append([m4, w_n, x, G, s, sigma,  scalar_4, Chi_SU_WA_4, Pval_SU_WA_4, experiment_Theta, experiment_SU, Theta_4_ret, SU4_ret_pchip])

        #SL5_ret = np.average(SL_2darray_5_ret, axis=0, weights=w_a)
        #SR5_ret = np.average(SR_2darray_5_ret, axis=0, weights=w_a)
        SU5_ret = np.average(SU_2darray_5_ret, axis=0, weights=G)
        Theta_5_ret = np.average(theta_2darray_5, axis=0)
        SU5_ret_pchip = pchip_interpolate(Theta_5_ret, SU5_ret, experiment_Theta, der=0, axis=0)
        scalar_5 = SU5_ret_pchip[np.argmin(SU5_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_5 = scalar_5 * experiment_SU
        Chi_SU_WA_5, Pval_SU_WA_5 = chisquare(f_obs=experiment_SU_scaled_5[250:len(experiment_SU_scaled_5)-1], f_exp=SU5_ret_pchip[250:len(experiment_SU_scaled_5)-1])
        Chi_2darray_5.append([m5, w_n, x, G, s, sigma, scalar_5, Chi_SU_WA_5, Pval_SU_WA_5, experiment_Theta, experiment_SU, Theta_5_ret, SU5_ret_pchip])

        #SL6_ret = np.average(SL_2darray_6_ret, axis=0, weights=w_a)
        #SR6_ret = np.average(SR_2darray_6_ret, axis=0, weights=w_a)
        SU6_ret = np.average(SU_2darray_6_ret, axis=0, weights=G)
        Theta_6_ret = np.average(theta_2darray_6, axis=0)
        SU6_ret_pchip = pchip_interpolate(Theta_6_ret, SU6_ret, experiment_Theta, der=0, axis=0)
        scalar_6 = SU6_ret_pchip[np.argmin(SU6_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_6 = scalar_6 * experiment_SU
        Chi_SU_WA_6, Pval_SU_WA_6 = chisquare(f_obs=experiment_SU_scaled_6[250:len(experiment_SU_scaled_6)-1], f_exp=SU6_ret_pchip[250:len(experiment_SU_scaled_6)-1])
        Chi_2darray_6.append([m6, w_n, x, G, s, sigma, scalar_6, Chi_SU_WA_6, Pval_SU_WA_6, experiment_Theta, experiment_SU, Theta_6_ret, SU6_ret_pchip])

        #SL7_ret = np.average(SL_2darray_7_ret, axis=0, weights=w_a)
        #SR7_ret = np.average(SR_2darray_7_ret, axis=0, weights=w_a)
        SU7_ret = np.average(SU_2darray_7_ret, axis=0, weights=G)
        Theta_7_ret = np.average(theta_2darray_7, axis=0)
        SU7_ret_pchip = pchip_interpolate(Theta_7_ret, SU7_ret, experiment_Theta, der=0, axis=0)
        scalar_7 = SU7_ret_pchip[np.argmin(SU7_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_7 = scalar_7 * experiment_SU
        Chi_SU_WA_7, Pval_SU_WA_7 = chisquare(f_obs=experiment_SU_scaled_7[250:len(experiment_SU_scaled_7)-1], f_exp=SU7_ret_pchip[250:len(experiment_SU_scaled_7)-1])
        Chi_2darray_7.append([m7, w_n, x, G, s, sigma, scalar_7, Chi_SU_WA_7, Pval_SU_WA_7, experiment_Theta, experiment_SU, Theta_7_ret, SU7_ret_pchip])

        #SL8_ret = np.average(SL_2darray_8_ret, axis=0, weights=w_a)
        #SR8_ret = np.average(SR_2darray_8_ret, axis=0, weights=w_a)
        SU8_ret = np.average(SU_2darray_8_ret, axis=0, weights=G)
        Theta_8_ret = np.average(theta_2darray_8, axis=0)
        SU8_ret_pchip = pchip_interpolate(Theta_8_ret, SU8_ret, experiment_Theta, der=0, axis=0)
        scalar_8 = SU8_ret_pchip[np.argmin(SU8_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_8 = scalar_8 * experiment_SU
        Chi_SU_WA_8, Pval_SU_WA_8 = chisquare(f_obs=experiment_SU_scaled_8[250:len(experiment_SU_scaled_8)-1], f_exp=SU8_ret_pchip[250:len(experiment_SU_scaled_8)-1])
        Chi_2darray_8.append([m8, w_n, x, G, s, sigma, scalar_8, Chi_SU_WA_8, Pval_SU_WA_8, experiment_Theta, experiment_SU, Theta_8_ret, SU8_ret_pchip])

        #SL9_ret = np.average(SL_2darray_9_ret, axis=0, weights=w_a)
        #SR9_ret = np.average(SR_2darray_9_ret, axis=0, weights=w_a)
        SU9_ret = np.average(SU_2darray_9_ret, axis=0, weights=G)
        Theta_9_ret = np.average(theta_2darray_9, axis=0)
        SU9_ret_pchip = pchip_interpolate(Theta_9_ret, SU9_ret, experiment_Theta, der=0, axis=0)
        scalar_9 = SU9_ret_pchip[np.argmin(SU9_ret_pchip)] / experiment_SU[np.argmin(experiment_SU)]
        experiment_SU_scaled_9 = scalar_9 * experiment_SU
        Chi_SU_WA_9, Pval_SU_WA_9 = chisquare(f_obs=experiment_SU_scaled_9[250:len(experiment_SU_scaled_9)-1], f_exp=SU9_ret_pchip[250:len(experiment_SU_scaled_9)-1])
        Chi_2darray_9.append([m9, w_n, x, G, s, sigma, scalar_9, Chi_SU_WA_9, Pval_SU_WA_9, experiment_Theta, experiment_SU, Theta_9_ret, SU9_ret_pchip])
print('finished!')


print(np.array(Chi_2darray_0).shape)

Chi_DF_0 = pd.DataFrame(np.array(Chi_2darray_0), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_1 = pd.DataFrame(np.array(Chi_2darray_1), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_2 = pd.DataFrame(np.array(Chi_2darray_2), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_3 = pd.DataFrame(np.array(Chi_2darray_3), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_4 = pd.DataFrame(np.array(Chi_2darray_4), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_5 = pd.DataFrame(np.array(Chi_2darray_5), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_6 = pd.DataFrame(np.array(Chi_2darray_6), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_7 = pd.DataFrame(np.array(Chi_2darray_7), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_8 = pd.DataFrame(np.array(Chi_2darray_8), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])
Chi_DF_9 = pd.DataFrame(np.array(Chi_2darray_9), columns=['Real RI', 'Wavelength', 'Gaussian X', 'Gaussian Y', 'Mu', 'Sigma', 'Scalar', 'Chi', 'Pval', 'Exp Theta', 'Exp SU', 'Mie Theta', 'Mie SU'])

# create look up tables
Chi_DF_0.to_csv(Save_Directory + 'Matheson_1952_Lookup_Table.txt', sep=',', header=0)
Chi_DF_1.to_csv(Save_Directory + 'Bateman_1959_Lookup_Table.txt', sep=',', header=0)
Chi_DF_2.to_csv(Save_Directory + 'Nikalov_2000_Lookup_Table.txt', sep=',', header=0)
Chi_DF_3.to_csv(Save_Directory + 'Ma_2003_Lookup_Table.txt', sep=',', header=0)
Chi_DF_4.to_csv(Save_Directory + 'Sultanova_2003_Lookup_Table.txt', sep=',', header=0)
Chi_DF_5.to_csv(Save_Directory + 'Kasarova_2006_Lookup_Table.txt', sep=',', header=0)
Chi_DF_6.to_csv(Save_Directory + 'Miles_2010_Lookup_Table.txt', sep=',', header=0)
Chi_DF_7.to_csv(Save_Directory + 'Jones_2013_Lookup_Table.txt', sep=',', header=0)
Chi_DF_8.to_csv(Save_Directory + 'Greenslade_2017_Lookup_Table.txt', sep=',', header=0)
Chi_DF_9.to_csv(Save_Directory + 'Gienger_2017_Lookup_Table.txt', sep=',', header=0)
#print(np.array(Chi_DF_0['Mie SU'])[0])

# Find Global Minimum amongst all comparisons
DF_List = [Chi_DF_0, Chi_DF_1, Chi_DF_2, Chi_DF_3, Chi_DF_4, Chi_DF_5, Chi_DF_6, Chi_DF_7, Chi_DF_8, Chi_DF_9]
Group_List  = ['Matheson et. al. 1952', 'Bateman et. al. 1959', 'Nikalov et. al. 2000', 'Ma et. al. 2003', 'Sultanova et. al. 2003', 'Kasarova et. al. 2006', 'Miles et. al. 2010', 'Jones et. al. 2013', 'Greenslade et. al. 2017', 'Gienger et. al. 2017']
Global_Chi_Min = []
for count, DF in enumerate(DF_List):
    i = np.argmin(np.array(DF['Chi']))
    Global_Chi_Min.append(np.array(DF['Chi'])[i])

min_i = np.argmin(np.array(Global_Chi_Min))
print('Group with lowest Chi: ', Group_List[min_i])
print('Real refactive index of: ', n_all_groups[min_i])



fig4 = plt.figure( figsize=(12, 7))
ax4a = fig4.add_subplot(224, projection='3d')
min_idx_0 = np.argmin(np.array(Chi_DF_0['Chi']))
ax4a.scatter(np.array(Chi_DF_0['Mu']), np.array(Chi_DF_0['Sigma']), np.array(Chi_DF_0['Chi']), cmap='autumn_r')
ax4a.set_title('Chi Minimization Space')
ax4a.set_xlabel('\u03bc')
ax4a.set_ylabel('\u03c3')
ax4a.set_zlabel('$|\u03a7|^2$')
ax4a.grid(True)
ax4b = fig4.add_subplot(222)
ax4b.plot(np.array(Chi_DF_0['Gaussian X'])[min_idx_0], np.array(Chi_DF_0['Gaussian Y'])[min_idx_0], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_0['Mu'])[min_idx_0]) + ', \u03c3=' + str(np.array(Chi_DF_0['Sigma'])[min_idx_0]))
ax4b.set_title('Gaussian Size Distribution')
ax4b.set_xlabel('Diameter (nm)')
ax4b.set_ylabel('Normalized Gaussian')
ax4b.grid(True)
ax4b.legend(loc=1)
ax4c = fig4.add_subplot(121)
ax4c.semilogy(np.array(Chi_DF_0['Exp Theta'])[min_idx_0], np.array(Chi_DF_0['Exp SU'])[min_idx_0] * np.array(Chi_DF_0['Scalar'])[min_idx_0], color='red', ls='-', label='Measurement')
ax4c.semilogy(np.array(Chi_DF_0['Exp Theta'])[min_idx_0][250:len(experiment_SU_scaled_0)-1], np.array(Chi_DF_0['Exp SU'])[min_idx_0][250:len(experiment_SU_scaled_0)-1] * np.array(Chi_DF_0['Scalar'])[min_idx_0], color='orange', ls='-', lw=3, label='Measurement Subset')
ax4c.semilogy(np.array(Chi_DF_0['Exp Theta'])[min_idx_0], np.array(Chi_DF_0['Mie SU'])[min_idx_0], color='blue', ls='-', label='Best Mie Theory')
ax4c.semilogy(np.array(Chi_DF_0['Exp Theta'])[min_idx_0][250:len(experiment_SU_scaled_0)-1], np.array(Chi_DF_0['Mie SU'])[min_idx_0][250:len(experiment_SU_scaled_0)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax4c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax4c.set_xlabel("ϴ")
ax4c.set_ylabel('$|SU|^2$')
ax4c.grid(True)
ax4c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Matheson et. al. 1952')
plt.savefig(Save_Directory + 'Matheson_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Matheson_Min.png', format='png')
plt.close()

fig5 = plt.figure(figsize=(12, 7))
min_idx_1 = np.argmin(np.array(Chi_DF_1['Chi']))
ax5a = fig5.add_subplot(224, projection='3d')
ax5a.scatter(np.array(Chi_DF_1['Mu']), np.array(Chi_DF_1['Sigma']), np.array(Chi_DF_1['Chi']), cmap='autumn_r')
ax5a.set_title('Chi Minimization Space')
ax5a.set_xlabel('\u03bc')
ax5a.set_ylabel('\u03c3')
ax5a.set_zlabel('$|\u03a7|^2$')
ax5a.grid(True)
ax5b = fig5.add_subplot(222)
ax5b.plot(np.array(Chi_DF_1['Gaussian X'])[min_idx_1], np.array(Chi_DF_1['Gaussian Y'])[min_idx_1], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_1['Mu'])[min_idx_1]) + ', \u03c3=' + str(np.array(Chi_DF_1['Sigma'])[min_idx_1]))
ax5b.set_title('Gaussian Size Distribution')
ax5b.set_xlabel('Diameter (nm)')
ax5b.set_ylabel('Normalized Gaussian')
ax5b.grid(True)
ax5b.legend(loc=1)
ax5c = fig5.add_subplot(121)
ax5c.semilogy(np.array(Chi_DF_1['Exp Theta'])[min_idx_1], np.array(Chi_DF_1['Exp SU'])[min_idx_1] * np.array(Chi_DF_1['Scalar'])[min_idx_1], color='orange', ls='-', label='Measurement')
ax5c.semilogy(np.array(Chi_DF_1['Exp Theta'])[min_idx_1][250:len(experiment_SU_scaled_1)-1], np.array(Chi_DF_1['Exp SU'])[min_idx_1][250:len(experiment_SU_scaled_1)-1] * np.array(Chi_DF_1['Scalar'])[min_idx_1], color='orange', ls='-', lw=3, label='Measurement Subset')
ax5c.semilogy(np.array(Chi_DF_1['Exp Theta'])[min_idx_1], np.array(Chi_DF_1['Mie SU'])[min_idx_1], color='red', ls='-', label='Best Mie Theory')
ax5c.semilogy(np.array(Chi_DF_1['Exp Theta'])[min_idx_1][250:len(experiment_SU_scaled_1)-1], np.array(Chi_DF_1['Mie SU'])[min_idx_1][250:len(experiment_SU_scaled_1)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax5c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax5c.set_xlabel("ϴ")
ax5c.set_ylabel('$|SU|^2$')
ax5c.grid(True)
ax5c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Bateman et. al. 1959')
plt.savefig(Save_Directory + 'Bateman_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Bateman_Min.png', format='png')
plt.close()

fig6 = plt.figure(figsize=(12, 7))
min_idx_2 = np.argmin(np.array(Chi_DF_2['Chi']))
ax6a = fig6.add_subplot(224, projection='3d')
ax6a.scatter(np.array(Chi_DF_2['Mu']), np.array(Chi_DF_2['Sigma']), np.array(Chi_DF_2['Chi']), cmap='autumn_r')
ax6a.set_title('Chi Minimization Space')
ax6a.set_xlabel('\u03bc')
ax6a.set_ylabel('\u03c3')
ax6a.set_zlabel('$|\u03a7|^2$')
ax6a.grid(True)
ax6b = fig6.add_subplot(222)
ax6b.plot(np.array(Chi_DF_2['Gaussian X'])[min_idx_2], np.array(Chi_DF_2['Gaussian Y'])[min_idx_2], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_2['Mu'])[min_idx_2]) + ', \u03c3=' + str(np.array(Chi_DF_2['Sigma'])[min_idx_2]))
ax6b.set_title('Gaussian Size Distribution')
ax6b.set_xlabel('Diameter (nm)')
ax6b.set_ylabel('Normalized Gaussian')
ax6b.grid(True)
ax6b.legend(loc=1)
ax6c = fig6.add_subplot(121)
ax6c.semilogy(np.array(Chi_DF_2['Exp Theta'])[min_idx_2], np.array(Chi_DF_2['Exp SU'])[min_idx_2] * np.array(Chi_DF_2['Scalar'])[min_idx_2], color='orange', ls='-', lw=3, label='Measurement')
ax6c.semilogy(np.array(Chi_DF_2['Exp Theta'])[min_idx_2][250:len(experiment_SU_scaled_2)-1], np.array(Chi_DF_2['Exp SU'])[min_idx_2][250:len(experiment_SU_scaled_2)-1] * np.array(Chi_DF_2['Scalar'])[min_idx_2], color='orange', ls='-', label='Measurement Subset')
ax6c.semilogy(np.array(Chi_DF_2['Exp Theta'])[min_idx_2], np.array(Chi_DF_2['Mie SU'])[min_idx_2], color='red', ls='-', label='Best Mie Theory')
ax6c.semilogy(np.array(Chi_DF_2['Exp Theta'])[min_idx_2][250:len(experiment_SU_scaled_2)-1], np.array(Chi_DF_2['Mie SU'])[min_idx_2][250:len(experiment_SU_scaled_2)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax6c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax6c.set_xlabel("ϴ")
ax6c.set_ylabel('$|SU|^2$')
ax6c.grid(True)
ax6c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Nikalov et. al. 2000')
plt.savefig(Save_Directory + 'Nikalov_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Nikalov_Min.png', format='png')
plt.close()

fig7 = plt.figure(figsize=(12, 7))
min_idx_3 = np.argmin(np.array(Chi_DF_3['Chi']))
ax7a = fig7.add_subplot(224, projection='3d')
ax7a.scatter(np.array(Chi_DF_3['Mu']), np.array(Chi_DF_3['Sigma']), np.array(Chi_DF_3['Chi']), cmap='autumn_r')
ax7a.set_title('Chi Minimization Space')
ax7a.set_xlabel('\u03bc')
ax7a.set_ylabel('\u03c3')
ax7a.set_zlabel('$|\u03a7|^2$')
ax7a.grid(True)
ax7b = fig7.add_subplot(222)
ax7b.plot(np.array(Chi_DF_3['Gaussian X'])[min_idx_3], np.array(Chi_DF_3['Gaussian Y'])[min_idx_3], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_3['Mu'])[min_idx_3]) + ', \u03c3=' + str(np.array(Chi_DF_3['Sigma'])[min_idx_3]))
ax7b.set_title('Gaussian Size Distribution')
ax7b.set_xlabel('Diameter (nm)')
ax7b.set_ylabel('Normalized Gaussian')
ax7b.grid(True)
ax7b.legend(loc=1)
ax7c = fig7.add_subplot(121)
ax7c.semilogy(np.array(Chi_DF_3['Exp Theta'])[min_idx_3], np.array(Chi_DF_3['Exp SU'])[min_idx_3] * np.array(Chi_DF_3['Scalar'])[min_idx_3], color='orange', ls='-', label='Measurement')
ax7c.semilogy(np.array(Chi_DF_3['Exp Theta'])[min_idx_3][250:len(experiment_SU_scaled_3)-1], np.array(Chi_DF_3['Exp SU'])[min_idx_3][250:len(experiment_SU_scaled_3)-1] * np.array(Chi_DF_3['Scalar'])[min_idx_3], color='orange', ls='-', lw=3, label='Measurement Subset')
ax7c.semilogy(np.array(Chi_DF_3['Exp Theta'])[min_idx_3], np.array(Chi_DF_3['Mie SU'])[min_idx_3], color='red', ls='-', label='Best Mie Theory')
ax7c.semilogy(np.array(Chi_DF_3['Exp Theta'])[min_idx_3][250:len(experiment_SU_scaled_3)-1], np.array(Chi_DF_3['Mie SU'])[min_idx_3][250:len(experiment_SU_scaled_3)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax7c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax7c.set_xlabel("ϴ")
ax7c.set_ylabel('$|SU|^2$')
ax7c.grid(True)
ax7c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Ma et. al. 2003')
plt.savefig(Save_Directory + 'Ma_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Ma_Min.png', format='png')
plt.close()

fig8 = plt.figure(figsize=(12, 7))
min_idx_4 = np.argmin(np.array(Chi_DF_4['Chi']))
ax8a = fig8.add_subplot(224, projection='3d')
ax8a.scatter(np.array(Chi_DF_4['Mu']), np.array(Chi_DF_4['Sigma']), np.array(Chi_DF_4['Chi']), cmap='autumn_r')
ax8a.set_title('Chi Minimization Space')
ax8a.set_xlabel('\u03bc')
ax8a.set_ylabel('\u03c3')
ax8a.set_zlabel('$|\u03a7|^2$')
ax8a.grid(True)
ax8b = fig8.add_subplot(222)
ax8b.plot(np.array(Chi_DF_4['Gaussian X'])[min_idx_4], np.array(Chi_DF_4['Gaussian Y'])[min_idx_4], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_4['Mu'])[min_idx_4]) + ', \u03c3=' + str(np.array(Chi_DF_4['Sigma'])[min_idx_4]))
ax8b.set_title('Gaussian Size Distribution')
ax8b.set_xlabel('Diameter (nm)')
ax8b.set_ylabel('Normalized Gaussian')
ax8b.grid(True)
ax8b.legend(loc=1)
ax8c = fig8.add_subplot(121)
ax8c.semilogy(np.array(Chi_DF_4['Exp Theta'])[min_idx_4], np.array(Chi_DF_4['Exp SU'])[min_idx_4] * np.array(Chi_DF_4['Scalar'])[min_idx_4], color='orange', ls='-', label='Measurement')
ax8c.semilogy(np.array(Chi_DF_4['Exp Theta'])[min_idx_4][250:len(experiment_SU_scaled_4)-1], np.array(Chi_DF_4['Exp SU'])[min_idx_4][250:len(experiment_SU_scaled_4)-1] * np.array(Chi_DF_4['Scalar'])[min_idx_4], color='orange', ls='-', lw=3, label='Measurement Subset')
ax8c.semilogy(np.array(Chi_DF_4['Exp Theta'])[min_idx_4], np.array(Chi_DF_4['Mie SU'])[min_idx_4], color='red', ls='-', label='Best Mie Theory')
ax8c.semilogy(np.array(Chi_DF_4['Exp Theta'])[min_idx_4][250:len(experiment_SU_scaled_4)-1], np.array(Chi_DF_4['Mie SU'])[min_idx_4][250:len(experiment_SU_scaled_4)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax8c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax8c.set_xlabel("ϴ")
ax8c.set_ylabel('$|SU|^2$')
ax8c.grid(True)
ax8c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Sultanova et. al. 2003')
plt.savefig(Save_Directory + 'Sultanova_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Sultanova_Min.png', format='png')
plt.close()

fig9 = plt.figure(figsize=(12, 7))
min_idx_5 = np.argmin(np.array(Chi_DF_5['Chi']))
ax9a = fig9.add_subplot(224, projection='3d')
ax9a.scatter(np.array(Chi_DF_5['Mu']), np.array(Chi_DF_5['Sigma']), np.array(Chi_DF_5['Chi']), cmap='autumn_r')
ax9a.set_title('Chi Minimization Space')
ax9a.set_xlabel('\u03bc')
ax9a.set_ylabel('\u03c3')
ax9a.set_zlabel('$|\u03a7|^2$')
ax9a.grid(True)
ax9b = fig9.add_subplot(222)
ax9b.plot(np.array(Chi_DF_5['Gaussian X'])[min_idx_5], np.array(Chi_DF_5['Gaussian Y'])[min_idx_5], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_5['Mu'])[min_idx_5]) + ', \u03c3=' + str(np.array(Chi_DF_5['Sigma'])[min_idx_5]))
ax9b.set_title('Gaussian Size Distribution')
ax9b.set_xlabel('Diameter (nm)')
ax9b.set_ylabel('Normalized Gaussian')
ax9b.grid(True)
ax9b.legend(loc=1)
ax9c = fig9.add_subplot(121)
ax9c.semilogy(np.array(Chi_DF_5['Exp Theta'])[min_idx_5], np.array(Chi_DF_5['Exp SU'])[min_idx_5] * np.array(Chi_DF_5['Scalar'])[min_idx_5], color='orange', ls='-', label='Measurement')
ax9c.semilogy(np.array(Chi_DF_5['Exp Theta'])[min_idx_5][250:len(experiment_SU_scaled_5)-1], np.array(Chi_DF_5['Exp SU'])[min_idx_5][250:len(experiment_SU_scaled_5)-1] * np.array(Chi_DF_5['Scalar'])[min_idx_5], color='orange', ls='-', lw=3, label='Measurement Subset')
ax9c.semilogy(np.array(Chi_DF_5['Exp Theta'])[min_idx_5], np.array(Chi_DF_0['Mie SU'])[min_idx_5], color='red', ls='-', label='Best Mie Theory')
ax9c.semilogy(np.array(Chi_DF_5['Exp Theta'])[min_idx_5][250:len(experiment_SU_scaled_5)-1], np.array(Chi_DF_5['Mie SU'])[min_idx_5][250:len(experiment_SU_scaled_5)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax9c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax9c.set_xlabel("ϴ")
ax9c.set_ylabel('$|SU|^2$')
ax9c.grid(True)
ax9c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Kasarova et. al. 2006')
plt.savefig(Save_Directory + 'Kasarova_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Kasarova_Min.png', format='png')
plt.close()

fig10 = plt.figure(figsize=(12, 7))
min_idx_6 = np.argmin(np.array(Chi_DF_6['Chi']))
ax10a = fig10.add_subplot(224, projection='3d')
ax10a.scatter(np.array(Chi_DF_6['Mu']), np.array(Chi_DF_6['Sigma']), np.array(Chi_DF_6['Chi']), cmap='autumn_r')
ax10a.set_title('Chi Minimization Space')
ax10a.set_xlabel('\u03bc')
ax10a.set_ylabel('\u03c3')
ax10a.set_zlabel('$|\u03a7|^2$')
ax10a.grid(True)
ax10b = fig10.add_subplot(222)
ax10b.plot(np.array(Chi_DF_6['Gaussian X'])[min_idx_6], np.array(Chi_DF_6['Gaussian Y'])[min_idx_6], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_6['Mu'])[min_idx_6]) + ', \u03c3=' + str(np.array(Chi_DF_6['Sigma'])[min_idx_6]))
ax10b.set_title('Gaussian Size Distribution')
ax10b.set_xlabel('Diameter (nm)')
ax10b.set_ylabel('Normalized Gaussian')
ax10b.grid(True)
ax10b.legend(loc=1)
ax10c = fig10.add_subplot(121)
ax10c.semilogy(np.array(Chi_DF_6['Exp Theta'])[min_idx_6], np.array(Chi_DF_6['Exp SU'])[min_idx_6] * np.array(Chi_DF_6['Scalar'])[min_idx_6], color='orange', ls='-', label='Measurement')
ax10c.semilogy(np.array(Chi_DF_6['Exp Theta'])[min_idx_6][250:len(experiment_SU_scaled_6)-1], np.array(Chi_DF_6['Exp SU'])[min_idx_6][250:len(experiment_SU_scaled_6)-1] * np.array(Chi_DF_6['Scalar'])[min_idx_6], color='orange', ls='-', lw=3, label='Measurement Subset')
ax10c.semilogy(np.array(Chi_DF_6['Exp Theta'])[min_idx_6], np.array(Chi_DF_6['Mie SU'])[min_idx_6], color='red', ls='-', label='Best Mie Theory')
ax10c.semilogy(np.array(Chi_DF_6['Exp Theta'])[min_idx_6][250:len(experiment_SU_scaled_6)-1], np.array(Chi_DF_6['Mie SU'])[min_idx_6][250:len(experiment_SU_scaled_6)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax10c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax10c.set_xlabel("ϴ")
ax10c.set_ylabel('$|SU|^2$')
ax10c.grid(True)
ax10c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Miles et. al. 2010')
plt.savefig(Save_Directory + 'Miles_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Miles_Min.png', format='png')
plt.close()

fig11 = plt.figure(figsize=(12, 7))
min_idx_7 = np.argmin(np.array(Chi_DF_7['Chi']))
ax11a = fig11.add_subplot(224, projection='3d')
ax11a.scatter(np.array(Chi_DF_7['Mu']), np.array(Chi_DF_7['Sigma']), np.array(Chi_DF_7['Chi']), cmap='autumn_r')
ax11a.set_title('Chi Minimization Space')
ax11a.set_xlabel('\u03bc')
ax11a.set_ylabel('\u03c3')
ax11a.set_zlabel('$|\u03a7|^2$')
ax11a.grid(True)
ax11b = fig11.add_subplot(222)
ax11b.plot(np.array(Chi_DF_7['Gaussian X'])[min_idx_7], np.array(Chi_DF_7['Gaussian Y'])[min_idx_7], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_7['Mu'])[min_idx_7]) + ', \u03c3=' + str(np.array(Chi_DF_7['Sigma'])[min_idx_7]))
ax11b.set_title('Gaussian Size Distribution')
ax11b.set_xlabel('Diameter (nm)')
ax11b.set_ylabel('Normalized Gaussian')
ax11b.grid(True)
ax11b.legend(loc=1)
ax11c = fig11.add_subplot(121)
ax11c.semilogy(np.array(Chi_DF_7['Exp Theta'])[min_idx_7], np.array(Chi_DF_7['Exp SU'])[min_idx_7] * np.array(Chi_DF_7['Scalar'])[min_idx_7], color='orange', ls='-', label='Measurement')
ax11c.semilogy(np.array(Chi_DF_7['Exp Theta'])[min_idx_7][250:len(experiment_SU_scaled_7)-1], np.array(Chi_DF_7['Exp SU'])[min_idx_7][250:len(experiment_SU_scaled_7)-1] * np.array(Chi_DF_7['Scalar'])[min_idx_7], color='orange', ls='-', lw=3, label='Measurement Subset')
ax11c.semilogy(np.array(Chi_DF_7['Exp Theta'])[min_idx_7], np.array(Chi_DF_7['Mie SU'])[min_idx_7], color='red', ls='-', label='Best Mie Theory')
ax11c.semilogy(np.array(Chi_DF_7['Exp Theta'])[min_idx_7][250:len(experiment_SU_scaled_7)-1], np.array(Chi_DF_7['Mie SU'])[min_idx_7][250:len(experiment_SU_scaled_7)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax11c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax11c.set_xlabel("ϴ")
ax11c.set_ylabel('$|SU|^2$')
ax11c.grid(True)
ax11c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Jones et. al. 2013')
plt.savefig(Save_Directory + 'Jones_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Jones_Min.png', format='png')
plt.close()

fig12 = plt.figure(figsize=(12, 7))
min_idx_8 = np.argmin(np.array(Chi_DF_8['Chi']))
ax12a = fig12.add_subplot(224, projection='3d')
ax12a.scatter(np.array(Chi_DF_8['Mu']), np.array(Chi_DF_8['Sigma']), np.array(Chi_DF_8['Chi']), cmap='autumn_r')
ax12a.set_title('Chi Minimization Space')
ax12a.set_xlabel('\u03bc')
ax12a.set_ylabel('\u03c3')
ax12a.set_zlabel('$|\u03a7|^2$')
ax12a.grid(True)
ax12b = fig12.add_subplot(222)
ax12b.plot(np.array(Chi_DF_8['Gaussian X'])[min_idx_8], np.array(Chi_DF_8['Gaussian Y'])[min_idx_8], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_8['Mu'])[min_idx_8]) + ', \u03c3=' + str(np.array(Chi_DF_8['Sigma'])[min_idx_8]))
ax12b.set_title('Gaussian Size Distribution')
ax12b.set_xlabel('Diameter (nm)')
ax12b.set_ylabel('Normalized Gaussian')
ax12b.grid(True)
ax12b.legend(loc=1)
ax12c = fig12.add_subplot(121)
ax12c.semilogy(np.array(Chi_DF_8['Exp Theta'])[min_idx_8], np.array(Chi_DF_8['Exp SU'])[min_idx_8] * np.array(Chi_DF_8['Scalar'])[min_idx_8], color='orange', ls='-', label='Measurement')
ax12c.semilogy(np.array(Chi_DF_8['Exp Theta'])[min_idx_8][250:len(experiment_SU_scaled_8)-1], np.array(Chi_DF_8['Exp SU'])[min_idx_8][250:len(experiment_SU_scaled_8)-1] * np.array(Chi_DF_8['Scalar'])[min_idx_8], color='orange', ls='-', lw=3, label='Measurement Subset')
ax12c.semilogy(np.array(Chi_DF_8['Exp Theta'])[min_idx_8], np.array(Chi_DF_8['Mie SU'])[min_idx_8], color='red', ls='-', label='Best Mie Theory')
ax12c.semilogy(np.array(Chi_DF_8['Exp Theta'])[min_idx_8][250:len(experiment_SU_scaled_8)-1], np.array(Chi_DF_8['Mie SU'])[min_idx_8][250:len(experiment_SU_scaled_8)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax12c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax12c.set_xlabel("ϴ")
ax12c.set_ylabel('$|SU|^2$')
ax12c.grid(True)
ax12c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Greenslade et. al. 2017')
plt.savefig(Save_Directory + 'Greenslade_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Greenslade_Min.png', format='png')
plt.close()

fig13 = plt.figure(figsize=(12, 7))
min_idx_9 = np.argmin(np.array(Chi_DF_9['Chi']))
ax13a = fig13.add_subplot(224, projection='3d')
ax13a.scatter(np.array(Chi_DF_9['Mu']), np.array(Chi_DF_9['Sigma']), np.array(Chi_DF_9['Chi']), cmap='autumn_r')
ax13a.set_title('Chi Minimization Space')
ax13a.set_xlabel('\u03bc')
ax13a.set_ylabel('\u03c3')
ax13a.set_zlabel('$|\u03a7|^2$')
ax13a.grid(True)
ax13b = fig13.add_subplot(222)
ax13b.plot(np.array(Chi_DF_9['Gaussian X'])[min_idx_9], np.array(Chi_DF_9['Gaussian Y'])[min_idx_9], color='blue', ls='-', label='Gaussian Dist. \u03bc=' + str(np.array(Chi_DF_9['Mu'])[min_idx_9]) + ', \u03c3=' + str(np.array(Chi_DF_9['Sigma'])[min_idx_9]))
ax13b.set_title('Gaussian Size Distribution')
ax13b.set_xlabel('Diameter (nm)')
ax13b.set_ylabel('Normalized Gaussian')
ax13b.grid(True)
ax13b.legend(loc=1)
ax13c = fig13.add_subplot(121)
ax13c.semilogy(np.array(Chi_DF_9['Exp Theta'])[min_idx_9], np.array(Chi_DF_9['Exp SU'])[min_idx_9] * np.array(Chi_DF_9['Scalar'])[min_idx_9], color='orange', ls='-', label='Measurement')
ax13c.semilogy(np.array(Chi_DF_9['Exp Theta'])[min_idx_9][250:len(experiment_SU_scaled_9)-1], np.array(Chi_DF_9['Exp SU'])[min_idx_9][250:len(experiment_SU_scaled_9)-1] * np.array(Chi_DF_9['Scalar'])[min_idx_9], color='orange', ls='-', lw=3, label='Measurement Subset')
ax13c.semilogy(np.array(Chi_DF_9['Exp Theta'])[min_idx_9], np.array(Chi_DF_9['Mie SU'])[min_idx_9], color='red', ls='-', label='Best Mie Theory')
ax13c.semilogy(np.array(Chi_DF_9['Exp Theta'])[min_idx_9][250:len(experiment_SU_scaled_9)-1], np.array(Chi_DF_9['Mie SU'])[min_idx_9][250:len(experiment_SU_scaled_9)-1], color='aqua', ls='-', lw=3, label='Best Mie Theory Subset')
ax13c.set_title('Minimization of Measured Phase Function \n and Mie Phase Function ')
ax13c.set_xlabel("ϴ")
ax13c.set_ylabel('$|SU|^2$')
ax13c.grid(True)
ax13c.legend(loc=1)
plt.suptitle('Measurement vs. Mie Calculation Minimization Results \n Grienger et. al. 2017')
plt.savefig(Save_Directory + 'Grienger_Min.pdf', format='pdf')
plt.savefig(Save_Directory + 'Grienger_Min.png', format='png')
plt.close()







