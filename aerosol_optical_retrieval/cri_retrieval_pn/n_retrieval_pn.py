'''
Austen K. Scruggs
07-19-2019
Description: Retrieve n for a given lognormal size distribution
'''

import PyMieScatt as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, pi
from scipy.stats import chisquare
path = '/home/austen/Desktop/'
# generate artificial data, this is the creation of the log normal distribution
# total particle concentration
concentration = 1000
# geometric standard deviation
sigma_g = 1.05
# Particle diameter, geometric mean of the particle diameter
d= 900
# wavelength
w_n = 663
# CRI
m = 1.59 + 0.0j
# write function for LogNormal distribution (same as Tami Bonds)
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))

# create distribution data
sizes = np.arange(700, 1110, 10)
log_dist = [LogNormal(element, d, sigma_g, concentration) for element in sizes]

# plot distribution
f0, ax0 = plt.subplots(figsize=(12, 12))
ax0.plot(sizes, log_dist, 'r-', label='lognormal dist.: \u03bc=' + str(d) + ' $\u03c3_{g}$=' + str(sigma_g))
ax0.set_xlabel('particle diameter (nm)')
ax0.set_ylabel('dN/dD')
ax0.set_title('Log Normal Distribution')
ax0.grid(True)
ax0.legend(loc=1)
#plt.savefig(Save_Directory + 'Mie_Distributions.pdf', format='pdf')
plt.show()

# generate phase function data
pf_2darray = []

for counter, element in enumerate(sizes):
    theta, SL, SR, SU = ps.ScatteringFunction(m, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    # creating noise in phase functions
    noise_mu = np.mean(SU)
    noise_sigma = 1
    noise = np.random.normal(noise_mu, noise_sigma, len(SU))
    # Noise up the original signal
    SU_noise = SU + noise
    pf_2darray.append(SU_noise)

# plot phase functions that comprise the weighted average phase function
f1, ax1 = plt.subplots(figsize=(12, 12))
for counter, element in enumerate(pf_2darray):
    ax1.semilogy(theta, element, label='P.F. @ size: ' + str(sizes[counter]))
ax1.legend(loc=1, ncol=3)
ax1.set_title('Phase Functions Used to Create Weighted Average Phase Function')
ax1.set_xlabel('\u03b8 (\u0b00)')
ax1.set_ylabel('Intensity')
ax1.grid(True)
plt.show()

# calculate weighted average of phase function data based on weights determined from the lognormal distribution
pf_average = np.average(pf_2darray, axis=0, weights=log_dist)

# plot weighted average phase function
f2, ax2 = plt.subplots(figsize=(12,12))
ax2.semilogy(theta, pf_average, label='Weighted Average Phase Function')
ax2.legend(loc=1)
ax2.set_title('Weighted Average Phase Function')
ax2.set_xlabel('\u03b8 (\u00b0)')
ax2.set_ylabel('Intensity')
ax2.grid(True)
plt.show()

# testing retrieval with data generated from mie theory with added noise
cri_n_space = np.arange(1.54, 1.59, .01)
cri_k_space = np.arange(0.0, 0.1, 0.02)
pf_2darray_R = []
m_R_space = []
pf_space = []
chi_square_space = []
p_val_space = []
for n in cri_n_space:
    for k in cri_k_space:
        m_R = complex(n, k)
        print(m_R)
        m_R_space.append(m_R)
        for element in sizes:
            theta_R, SL_R, SR_R, SU_R = ps.ScatteringFunction(m_R, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            pf_2darray_R.append(SU_R)
        pf_average_R = np.average(pf_2darray_R, axis=0, weights=log_dist)
        pf_2darray_R = []
        pf_space.append(pf_average_R)
        chi_val, p_val = chisquare(f_obs=pf_average, f_exp=pf_average_R)
        chi_square_space.append(chi_val)
        p_val_space.append(p_val)

'''
df1 = pd.DataFrame()
df['CRI'] = m_R_space
df1['Chi Square Statistic'] = chi_square_space
df1['P Val'] = p_val_space
df2['Theta'] = theta_R
df2['Phase Function Space'] = pf_space
df1.to_csv(path + 'data.txt')
'''

f3, ax3 = plt.subplots(1, 2, figsize=(12, 12))
for counter, element in enumerate(pf_space):
    ax3[0].plot(theta_R, element, label='PF @ m = ' + str(m_R_space[counter]))
ax3[0].set_title('Phase Functions at Various Refractive Indices')
ax3[0].set_xlabel('\u03b8 (\u0b00)')
ax3[0].set_ylabel('Intensity')
ax3[0].legend(loc=1)
ax3[0].grid(True)
ax3[1].plot(m_R_space, chi_square_space, color='black', marker='o', ls=' ', label='Chi stat. vs. m')
ax3[1].set_xlabel('Complex Refractive Index')
ax3[1].set_ylabel('Chi Square Stat.')
ax3[1].legend(loc=1)
ax3[1].grid(True)
plt.show()


