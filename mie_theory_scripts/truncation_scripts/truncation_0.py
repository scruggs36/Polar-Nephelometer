'''
Austen K. Scruggs
08-22-2019
Descripiton: Attempt to play with fitting Mie Models to Experimental Data in order to minimize truncation error,
I don't want to figure out how to represent my phase functions with a sum of Legendre Polynomials with constraints
'''

import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as ps
from scipy.interpolate import pchip, UnivariateSpline, interp1d
from math import sqrt, pi, log

# Mie theory constants----------------------------------------------------------------------------------------
num_d = 1000
# geometric standard deviation
sigma_g = 1.005
# Particle diameter, geometric mean of the particle diameter
d = 903
# wavelength
w_n = 663
# CRI
m = 1.59 + 0.0j
# size array
size_array = np.arange(880, 930, 1)
# generate phase function data---------------------------------------------------------------------------------
# truncation, total 361 points (0-180 degrees at resolution of 0.5)
truncation = 20
# added noise constants
noise_mu = np.mean(0)
noise_sigma = 5.0

def LogNormal(size, mu, gsd, N):
    return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    #return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))

log_dist = np.array([LogNormal(element, d, sigma_g, num_d) for element in size_array])

# plot distribution
f0, ax0 = plt.subplots(figsize=(6, 12))
ax0.plot(size_array, log_dist, 'r-', label='lognormal dist.: \u03bc=' + str(d) + ' $\u03c3_{g}$=' + str(sigma_g))
ax0.set_xlabel('particle diameter (nm)')
ax0.set_ylabel('dN/dD')
ax0.set_title('Log Normal Distribution')
ax0.grid(True)
ax0.legend(loc=1)
#plt.savefig(data_path + 'size_distribution.pdf', format='pdf')
plt.show()

# pre-allocate
pf_2darray = []
for element in size_array:
    theta, SL, SR, SU = ps.ScatteringFunction(m, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    pf_2darray.append(SU)

pf_average = np.average(pf_2darray, axis=0, weights=log_dist)
pf_trunc = pf_average[truncation:-1*truncation]
# creating noise in phase functions

noise = np.random.normal(noise_mu, noise_sigma, len(pf_average))
# Noise up the original signal
pf_noise = np.add(pf_average, np.multiply(pf_average, noise / 100))[truncation:-1*truncation]
theta_noise = theta[truncation:-1*truncation]
# pchip of data without noise, just incase it becomes useful
pchip_fit = pchip(x=theta_noise, y=pf_trunc, axis=0, extrapolate=None)
pf_pchip = pchip_fit(theta)
# linear extrapolation of noisy data with nearest neighbor method
interp1d_nearest = interp1d(theta_noise, pf_trunc, kind='nearest', fill_value='extrapolate')
interp1d_slinear = interp1d(theta_noise, pf_trunc, kind='slinear', fill_value='extrapolate')
interp1d_quadratic = interp1d(theta_noise, pf_trunc, kind='quadratic', fill_value='extrapolate')
interp1d_cubic = interp1d(theta_noise, pf_trunc, kind='cubic', fill_value='extrapolate')
pf_extrapolate_0 = interp1d_nearest(theta)
pf_extrapolate_1 = interp1d_slinear(theta)
pf_extrapolate_2 = interp1d_quadratic(theta)
pf_extrapolate_3 = interp1d_cubic(theta)
# univariate spline extrapolation

# plot phase functions that comprise the weighted average phase function
f1, ax1 = plt.subplots(1, 2, figsize=(6, 12))
for counter, element in enumerate(pf_2darray):
    ax1[0].semilogy(theta, element, label='P.F. @ size: ' + str(size_array[counter]))
ax1[0].legend(loc=1, ncol=3)
ax1[0].set_title('Phase Functions Used to Create Weighted Average Phase Function')
ax1[0].set_xlabel('\u03b8 (\u00b0)')
ax1[0].set_ylabel('Intensity')
ax1[0].grid(True)
ax1[1].semilogy(theta, pf_average, color='red', ls='-', label='Calc. Phase Function')
ax1[1].semilogy(theta_noise, pf_noise, color='black', ls='-', label='Calc. Phase Function w Noise + Trunc ')
ax1[1].semilogy(theta, pf_pchip, color='blue', ls='-', label='Pchip Phase Function')
ax1[1].semilogy(theta, pf_extrapolate_0, color='green', ls='-', label='Nearest Neighbor')
ax1[1].semilogy(theta, pf_extrapolate_1, color='orange', ls='-', label='Univariate Spline')
ax1[1].semilogy(theta, pf_extrapolate_2, color='aqua', ls='-', label='Quadratic Spline')
ax1[1].semilogy(theta, pf_extrapolate_3, color='purple', ls='-', label='Cubic Spline')
ax1[1].legend(loc=1)
ax1[1].set_title('Weighted Average Phase Function')
ax1[1].set_xlabel('\u03b8 (\u00b0)')
ax1[1].set_ylabel('Intensity')
ax1[1].grid(True)
#plt.savefig(data_path + 'generated_pfs.pdf', format='pdf')
plt.show()
