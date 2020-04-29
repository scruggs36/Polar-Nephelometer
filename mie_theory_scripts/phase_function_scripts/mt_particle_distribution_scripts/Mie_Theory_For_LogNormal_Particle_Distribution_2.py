'''
Austen K. Scruggs
09-13-2018
Description: This is a Mie Theory script that a computes scattering diagram
taking into account identical particles in a given gaussian size distribution
'''

import numpy as np
import PyMieScatt as PMS
from math import pi, sqrt, log10
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

Save_Directory = '/home/austen/Desktop/Recent/'
# Particle diameter, geometric mean of the particle diameter
d= 600
# geometric standard deviation
sigma_g = 1.005
# wavelength
wavelength = 663.0
# refractive index of medium
m_medium = 1.000277
# Cauchy parameters for PSL particles
A = 1.5725
B = 0.0031080
C = 0.00034779
# n of refractive index for PSL as cauchy equation
n_cauchy = A + (B / wavelength ** 2) + (C / (wavelength ** 4))
#complex refractive index
#CRI = complex(n_cauchy, 0.0005)
m_array = np.arange(1.3, 1.7,.001)
k_array = np.array(0.00, .70, .001)
# particle wavenumber calculation
#k = (pi * m_medium) / wavelength
# size parameter calculation
#X = k * d
# scattering angles mu (cosine weighted)
#theta = np.arange(0, 181, 1)
#rads = [x * (pi / 180.0) for x in theta]
#mu = np.cos(rads)
# font sizes for figures
f_title = 24
f_axes = 18
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']

# write function for LogNormal distribution (same as Tami Bonds)
def LogNormal(diam, mu, gsd, N):
    return (np.exp(-1 * ((np.log10(diam/mu)) ** 2) / (2 * log10(gsd) ** 2)) / (sqrt(2 * pi) * log10(gsd)))*N

LogNormal_Size_Range = np.arange(1, 1000, 10)
LogNormal_Counts = LogNormal(LogNormal_Size_Range, mu=d, gsd=sigma_g, N=1)
LogNormal_Weights = LogNormal(LogNormal_Size_Range, mu=d, gsd=sigma_g, N=1)/np.linalg.norm(LogNormal(LogNormal_Size_Range, mu=d, gsd=sigma_g, N=1))

for m in m_array():
    for k in k_array():
        CRI = complex(m, k)
        SL_2darray = []
        SR_2darray = []
        SU_2darray = []
for size in LogNormal_Size_Range:
    theta, SL, SR, SU = PMS.ScatteringFunction(CRI, wavelength, size, nMedium=1.0, minAngle=0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='t')
    SL_2darray.append(SL)
    SR_2darray.append(SR)
    SU_2darray.append(SU)


SL = np.average(SL_2darray, axis=0, weights=LogNormal_Weights)
SR = np.average(SR_2darray, axis=0, weights=LogNormal_Weights)
SU = np.average(SU_2darray, axis=0, weights=LogNormal_Weights)
DLP = (-1*(SL - SR))/(SL+SR)

# plot size distribution
f0, ax0 = plt.subplots(figsize=(8, 6))
ax0.plot(LogNormal_Size_Range, LogNormal_Counts, color='red', ls='-', label='LogNormal Dist.: \u03bc=' + str(d) + 'nm, ' + '\u03c3=' + str(sigma_g))
ax0.set_xlabel('particle diameter (nm)', fontsize=f_axes)
ax0.set_ylabel('dN/Log(D)', fontsize=f_axes)
ax0.set_title('LogNormal Distribution', fontsize=f_title)
ax0.grid(True)
ax0.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(Save_Directory + 'Mie_LogNormal.png', format='png')
plt.show()


# plot SL (Parallel polarization to the scattering plane)
f1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(theta, SL, color='orange', ls='-', label='SL')
ax1.set_xlabel('\u03b8(\u00b0)', fontsize=f_axes)
ax1.set_ylabel('Normalized Radiance', fontsize=f_axes)
ax1.set_title('Phase Function for Incident Radiation Polarized\n Parallel to the Scattering Plane', fontsize=f_title)
ax1.grid(True)
ax1.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(Save_Directory + 'SL_LogNormal.png', format='png')
plt.show()

# plot SR
f2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(theta, SR, color='green', ls='-', label='SR')
ax2.set_xlabel('\u03b8(\u00b0)', fontsize=f_axes)
ax2.set_ylabel('Normalized Radiance', fontsize=f_axes)
ax2.set_title('Phase Function for Incident Radiation Polarized\n Perpendicular to the Scattering Plane', fontsize=f_title)
ax2.grid(True)
ax2.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(Save_Directory + 'SR_LogNormal.png', format='png')
plt.show()

# plot SU
f3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(theta, SU, color='blue', ls='-', label='SU')
ax3.set_xlabel('\u03b8(\u00b0)', fontsize=f_axes)
ax3.set_ylabel('Normalized Radiance', fontsize=f_axes)
ax3.set_title('Phase Function for Unpolarized Incident Radiation\n Relative to the Scattering Plane', fontsize=f_title)
ax3.grid(True)
ax3.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(Save_Directory + 'SU_LogNormal.png', format='png')
plt.show()

# plot DLP
f4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(theta, DLP, color='purple', ls='-', label='DLP')
ax4.set_xlabel('\u03b8(\u00b0)', fontsize=f_axes)
ax4.set_ylabel('Normalized Radiance', fontsize=f_axes)
ax4.set_title('Degree of Linear Polarization', fontsize=f_title)
ax4.grid(True)
ax4.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(Save_Directory + 'DLP_LogNormal.png', format='png')
plt.show()

