'''
Austen K. Scruggs
11-22-2019
Description: Retrieve the CRI and size distribution from the phase functions of 900nm, 800nm, and 600nm
PSL phase functions simultaneousely, this is based upon a genetic algorithm by Brian Barkey, without the genetic
algorithm selection part... we may implement a monte carlo apporach or something...
'''

import PyMieScatt as ps
import numpy as np
import pandas as pd
import scipy.optimize as so
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate


# data & save directories
# 0.5 lamda = Perpendicular
SR_dir = '/home/austen/Desktop/2019-11-21_Analysis/PSL/900/1s/0.5lamda/SD_Particle.txt'
# 0 lamda = Parallel
SL_dir = '/home/austen/Desktop/2019-11-21_Analysis/PSL/900/1s/0lamda/SD_Particle.txt'
# save directory
Save_dir = '/home/austen/Desktop/2019-11-21_Analysis/'

# import data
SL_data = pd.read_csv(SL_dir, sep=',', header=0)
SL_data = SL_data.dropna()
SR_data = pd.read_csv(SR_dir, sep=',', header=0)
SR_data = SR_data.dropna()
SL_exp = SL_data['Sample Intensity']
SL_exp_n = SL_exp / np.linalg.norm(np.array(SL_exp))
SR_exp = SR_data['Sample Intensity']
SR_exp_n = SR_exp / np.linalg.norm(np.array(SR_exp))

# theta experimental latest calibration: 11-21-2019
slope = 0.2049
intercept = -2.7594
SL_theta = [(slope * x) + intercept for x in SL_data['Sample Columns']]
SR_theta = [(slope * x) + intercept for x in SR_data['Sample Columns']]

# Functions List
# the lognormal function has been modified, N must be set to 1 based on the Barkey paper
def LogNormal(size, mu, sigma, N):
        return N / (((2 * np.pi)**(1/2)) * size * sigma) * np.exp((-1 * ((np.log(size) - np.log(mu)) ** 2)) / (2 * sigma ** 2))


def Gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# parameters definding the scattering solution mr (real RI), mu (mean of lognormal size dist), sigma (geom std of lognormal size dist)
def Dist_Weighted_SL(wavelength, m, mu, sigma, N):
    SL_MAT = []
    bin_vec = []
    size_array = np.linspace(mu - 10.0 *sigma, mu + 10.0 * sigma, 1)
    for size in size_array:
        bin_count = LogNormal(size, mu, sigma, N)
        bin_vec.append(bin_count)
        theta, SL, SR, SU = ps.ScatteringFunction(m, wavelength, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SL_MAT.append(SL)
    weights = np.array(bin_vec) / np.linalg.norm(np.array(bin_vec))
    SL = np.average(SL_MAT, axis=0, weights=weights)
    SL = pchip_interpolate(theta, SL, SL_theta, der=0, axis=0)
    SL = SL / np.linalg.norm(SL)
    return theta, SL


# parameters definding the scattering solution mr (real RI), mu (mean of lognormal size dist), sigma (geom std of lognormal size dist)
def Dist_Weighted_SR(wavelength, m, mu, sigma, N):
    SR_MAT = []
    bin_vec = []
    size_array = np.linspace(mu - 10.0 *sigma, mu + 10.0 * sigma, 1)
    for size in size_array:
        bin_count = LogNormal(size, mu, sigma, N)
        bin_vec.append(bin_count)
        theta, SL, SR, SU = ps.ScatteringFunction(m, wavelength, size, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SR_MAT.append(SR)
    weights = np.array(bin_vec) / np.linalg.norm(np.array(bin_vec))
    SR = np.average(SR_MAT, axis=0, weights=weights)
    SR = pchip_interpolate(theta, SR, SR_theta, der=0, axis=0)
    SR = SR / np.linalg.norm(SR)
    return theta, SR


# Brian Barkey function to minimize in Genetic Algorithm
def FL(wavelength, m, mu, sigma):
    return 1 - ((np.sum(np.subtract(np.log(Dist_Weighted_SL(m, wavelength, mu, sigma, 1)[1]), np.log(SL_exp_n))))/len(SL_exp_n))


def FR(wavelength, m, mu, sigma):
    return 1 - ((np.sum(np.subtract(np.log(Dist_Weighted_SR(m, wavelength, mu, sigma, 1)[1]), np.log(SR_exp_n))))/len(SR_exp_n))


def FT(X):
    m, mu, sigma = X
    FL = ((np.sum(np.square(np.subtract(np.log10(Dist_Weighted_SL(663, m, mu, sigma, 1)[1]), np.log10(SL_exp_n))))))
    FR = ((np.sum(np.square(np.subtract(np.log10(Dist_Weighted_SR(663, m, mu, sigma, 1)[1]), np.log10(SR_exp_n))))))
    return (FL + FR)/2

# simplex minimization
res = so.minimize(FT, x0=np.array([1.59, 903.0, 4.1]), method='Nelder-Mead', options={'maxiter':160})

print(res.success)
print(res.status)
print(res.message)
print('Solution Vector', res.x)
print('Number of Iterations', res.nit)
print(res.fun)

f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].semilogy(SL_theta, SL_exp_n, color='black', ls='-', label='Parallel Polarization Measurement')
ax0[0].semilogy(SL_theta, Dist_Weighted_SL(663, res.x[0], res.x[1], res.x[2], 1)[1], color='red', ls='-', label='minimized parameters: \n' + 'n = ' + str('{:.3f}'.format(res.x[0])) + ', \u03bc = ' + str('{:.3f}'.format(res.x[1])) + ', \u03c3 = ' + str('{:.3f}'.format(res.x[2])))
ax0[0].semilogy(SL_theta, Dist_Weighted_SL(663, 1.59, 903, 4.1, 1)[1], color='green', ls='-', label='Thermo Scientific Specs: \n' + 'n = ' + str('{:.2f}'.format(1.59)) + ', \u03bc = ' + str('{:.0f}'.format(903)) + ', \u03c3 = ' + str('{:.1f}'.format(4.1)))
ax0[0].set_title('Parallel Polarization \n Measurement and Minimization Result')
ax0[0].set_xlabel('\u0398')
ax0[0].set_ylabel('$S_{11}[Normalized]$')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].semilogy(SR_theta, SR_exp_n, color='black', ls='-', label='Perpendicular Polarization Measurement')
ax0[1].semilogy(SR_theta, Dist_Weighted_SR(663, res.x[0], res.x[1], res.x[2], 1)[1], color='red', ls='-', label='minimized parameters: \n' + 'n = ' + str('{:.3f}'.format(res.x[0])) + ', \u03bc = ' + str('{:.3f}'.format(res.x[1])) + ', \u03c3 = ' + str('{:.3f}'.format(res.x[2])))
ax0[1].semilogy(SR_theta, Dist_Weighted_SR(663, 1.59, 903, 4.1, 1)[1], color='green', ls='-', label='Thermo Scientific Specs: \n' + 'n = ' + str('{:.2f}'.format(1.59)) + ', \u03bc = ' + str('{:.0f}'.format(903)) + ', \u03c3 = ' + str('{:.1f}'.format(4.1)))
ax0[1].set_title('Perpendicular Polarization \n Measurement and Minimization Result')
ax0[1].set_xlabel('\u0398')
ax0[1].set_ylabel('$S_{12}[Normalized]$')
ax0[1].grid(True)
ax0[1].legend(loc=1)
plt.tight_layout()
plt.savefig(Save_dir + 'Simplex_Result.png', format='png')
plt.show()

