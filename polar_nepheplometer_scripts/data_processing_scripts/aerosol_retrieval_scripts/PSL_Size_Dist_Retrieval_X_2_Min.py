'''
Austen K. Scruggs
10-15-2018
Description:
This script calculates a weighted averaged scattering diagram for various size distribution widths (sigma) of particles.
For each size distriubition, a Chi square minimization is conducted at each of the Minima in the PSL scattering diagram
in order to obtain a theoretical scattering diagram with the correct size distribution to optimize the agreement between
measured scattering diagrams and Mie theory scattering diagrams.
'''

import numpy as np
import PyMieScatt as ps
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from scipy.interpolate import PchipInterpolator
from scipy.stats import chisquare
from scipy.signal import savgol_filter

Save_Directory = '/home/austen/Documents/'
Exp_Directory = '/home/austen/Documents/Good_Data/PSL_600nm_T5/SD_Offline.txt'

# import measured data
data = pd.read_csv(Exp_Directory, sep=',', header=0)
SD_Exp_Int = np.array(data['Sample Intensity']) - np.array(data['Nitrogen Intensity'])
SD_Exp_PN = np.array(data['Columns'])

# calibration to go from profile number to scattering angle
slope = 0.2056
# the intercept is 11.1% different used to be 45.6927, not sure why, gonna calibrate again w 900s soon
intercept = -40.6927
Exp_Theta = (slope * SD_Exp_PN) + intercept

# smooth experimental scattering diagrams by savitzky golay to eliminate noise spikes!
Exp_Smoothed_Intensity = savgol_filter(SD_Exp_Int, window_length=151, polyorder=2, deriv=0)
Exp_Smoothed_Intensity_Normed = Exp_Smoothed_Intensity / np.linalg.norm(Exp_Smoothed_Intensity)

# First we will calculate the particle size distribution and plot it
# Particle diameter, geometric mean of the particle diameter
d = np.arange(600.0, 630.0, 2.0)
# particle size standard deviation
sigma_s = np.arange(10.0, 50.0, 2.0)
# wavelength
wavelength = [663]


def Gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))



# pre-allocate arrays
theta_2darray = []
SL_2darray = []
SR_2darray = []
SU_2darray = []

DF = pd.DataFrame()
sizes_ndarray = []
weights_ndarray = []
chi_array = []
sigma_array = []
mu_array = []
wav_array = []
sd_ndarray = []
sd_normed_ndarray = []
scalar_array = []
# Double for loop for Xi square minimization
for element_w in wavelength:
    for element_d in d:
        for element_s in sigma_s:
            # create distribution size/width based on sigma
            size_axis = np.arange(element_d - (3 * element_s), element_d + (3 * element_s), 2)
            # create gaussian
            Gaussian_Data = Gaussian(size_axis, mu=element_d, sigma=element_s)
            weights_array = Gaussian_Data
            # store gaussian in 2darray
            #gaussian_data_ndarray.append(Gaussian_Data)

            # refractive index
            m_medium = 1.000277
            # Cauchy parameters for PSL, check greenslader paper for more RI
            A = 1.5725
            B = 0.0031080
            C = 0.00034779
            # n of refractive index for PSL as cauchy equation
            n_cauchy = A + (B / element_w ** 2) + (C / (element_w ** 4))
            m = n_cauchy + 0.0005j
            # particle wavenumber calculation
            k = (pi * m_medium) / element_w
            # size parameter calculation
            X = k * element_d
            # scattering angles
            theta = np.arange(0, 181, 1)
            rads = [x * (pi / 180.0) for x in theta]
            mu = np.cos(rads)
            # the loop below creates a 2darray filled with a column of sizes and a column of weights at that size
            for element_g in weights_array:
                theta, SL, SR, SU = ps.ScatteringFunction(m, element_w, element_d, nMedium=m_medium, minAngle=0, maxAngle=180,
                                                          angularResolution=0.2, space='theta', angleMeasure='degrees',
                                                          normalization=None)
                theta_2darray.append(theta)
                SL_2darray.append(np.array(SL) * element_g)
                SR_2darray.append(np.array(SR) * element_g)
                SU_2darray.append(np.array(SU) * element_g)

            Mie_Theta = np.average(theta_2darray, axis=0)
            SL_WA = np.average(SL_2darray, axis=0)
            SR_WA = np.average(SR_2darray, axis=0)
            SU_WA = np.average(SU_2darray, axis=0)

            SL_WA_Pchip_Fit = PchipInterpolator(Mie_Theta, SL_WA)
            SR_WA_Pchip_Fit = PchipInterpolator(Mie_Theta, SR_WA)
            SU_WA_Pchip_Fit = PchipInterpolator(Mie_Theta, SU_WA)

            SL_WA_Pchip_Data = SL_WA_Pchip_Fit(Exp_Theta)
            SR_WA_Pchip_Data = SR_WA_Pchip_Fit(Exp_Theta)
            SU_WA_Pchip_Data = SU_WA_Pchip_Fit(Exp_Theta)

            SL_WA_Pchip_Data_Normed = SL_WA_Pchip_Data / np.linalg.norm(SL_WA_Pchip_Data)
            SR_WA_Pchip_Data_Normed = SR_WA_Pchip_Data / np.linalg.norm(SR_WA_Pchip_Data)
            SU_WA_Pchip_Data_Normed = SU_WA_Pchip_Data / np.linalg.norm(SU_WA_Pchip_Data)
            scalar = SU_WA_Pchip_Data_Normed[360] / Exp_Smoothed_Intensity_Normed[360]
            # since we are only using the circularly polarized data, only the SU data matters
            Chi_SU_WA, Pval_SU_WA = chisquare(f_obs=SU_WA_Pchip_Data_Normed[350:-1], f_exp=scalar * Exp_Smoothed_Intensity_Normed[350:-1])

            chi_array.append(Chi_SU_WA)
            sizes_ndarray.append(size_axis)
            weights_ndarray.append(weights_array)
            wav_array.append(element_w)
            mu_array.append(element_d)
            sigma_array.append(element_s)
            sd_ndarray.append(SU_WA_Pchip_Data)
            sd_normed_ndarray.append(SU_WA_Pchip_Data_Normed)
            scalar_array.append(scalar)

            theta_2darray = []
            SL_2darray = []
            SR_2darray = []
            SU_2darray = []


DF['Wavelength'] = wav_array
DF['Gaussian Mu'] = mu_array
DF['Gaussian Sigma'] = sigma_array
DF['Xi^2'] = Chi_SU_WA
DF['Pval'] = Pval_SU_WA
DF['Scalar'] = scalar_array
DF['Size Range'] = sizes_ndarray
DF['Weights @ Size Range'] = weights_ndarray
DF['SU'] = sd_ndarray
DF['SU Normalized'] = sd_normed_ndarray
DF.to_csv(Save_Directory + '/' + 'Chi_Square_DF.txt', sep=',')

chi_min_index = np.argmin(chi_array)
chi_min = chi_array[chi_min_index]
size_range_best = sizes_ndarray[chi_min_index]
print('best size range: ', size_range_best)
weights_best = weights_ndarray[chi_min_index]
print('best gaussian: ', weights_best)
mu_best = mu_array[chi_min_index]
sigma_best = sigma_array[chi_min_index]
SU_best = sd_ndarray[chi_min_index]
SU_best_normed = sd_normed_ndarray[chi_min_index]
scalar_best = scalar_array[chi_min_index]
print('scalar: ', scalar_best)



f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].plot(size_range_best, weights_best, label='Gaussian Dist. \u03bc=' + str(mu_best) + ', \u03c3=' + str(sigma_best))
ax0[0].set_xlabel('particle diameter (nm)')
ax0[0].set_ylabel('Normalized $dN/Log_{10}(D)$')
ax0[0].set_title('Best Fit Gaussian Size Distribution Optimizing \n Agreement Between Mie Theory and Experiment')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].semilogy(Exp_Theta[350:-1], Exp_Smoothed_Intensity_Normed[350:-1], color='lawngreen', ls='-', marker='*', label='Exp. PSL 600nm Subset')
ax0[1].semilogy(Exp_Theta, Exp_Smoothed_Intensity_Normed, color='green', ls='-', label='Exp. PSL 600nm')
ax0[1].semilogy(Exp_Theta[350:-1], SU_best_normed[350:-1], color='cyan', ls='-', marker='*', label='Best SU Subset')
ax0[1].semilogy(Exp_Theta, SU_best_normed, color='blue', ls='-', label='Best SU')
ax0[1].semilogy(Exp_Theta, scalar_best * Exp_Smoothed_Intensity_Normed, color='purple', ls='-', label='Exp. PSL 600nm Scaled')
ax0[1].semilogy(Exp_Theta[366], Exp_Smoothed_Intensity_Normed[366], color='red', ls='', marker='^', ms=10, label='Scalar Point 1')
ax0[1].semilogy(Exp_Theta[366], SU_best_normed[366], color='orange', ls='', marker='^', ms=10, label='Scalar Point 2')
ax0[1].set_xlabel('\u03b8')
ax0[1].set_ylabel('Normalized SU')
ax0[1].set_title('Scattering Diagram')
ax0[1].grid(True)
ax0[1].legend(loc=1)
plt.savefig(Save_Directory + 'Best_SU.pdf', format='pdf')
plt.show()

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot_trisurf(sigma_array, mu_array, chi_array, label='$\u03c7^2$ Min. Surface', cmap='inferno')
ax1.plot_trisurf(sigma_best, mu_best, chi_min, label='Surface Minimum', marker='X', ms='7', color='lime')
ax1.set_xlabel('\u03c3')
ax1.set_ylabel('\u03bc')
ax1.set_zlabel('$\u03c7^2$')
ax1.set_title('$\u03c7^2$ Minimization Surface')
ax1.grid(True)
plt.savefig(Save_Directory + 'X^2_Min_Surface.pdf', format='pdf')
plt.show()
