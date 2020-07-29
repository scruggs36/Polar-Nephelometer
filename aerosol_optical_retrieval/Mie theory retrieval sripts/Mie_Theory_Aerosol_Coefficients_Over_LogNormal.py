'''
Austen K. Scruggs
05-18-2020
Testing PyMieScatt calculation of optical coefficients over lognormal distribution
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, linprog, curve_fit
from math import pi, sqrt, log
from scipy.interpolate import pchip_interpolate
import PyMieScatt as PMS
# values match that of OSA publication by Bernard Kaplan, Guy Ledanois, and Bernard Dre´ villon
# Mueller matrix of dense polystyrene latex sphere suspensions: measurements and Monte Carlo simulation
'''
m_val = complex(1.586, 0.00)
wav_val = 515.0
geomean_val = 404
geostdev_val = 1.0005
nmedium_val = 1.33
Nparticles_val = 1000
'''
m_val = complex(1.586, 0.00)
wav_val = 515.0
geomean_val = 404
geostdev_val = 1.0005
nmedium_val = 1.33
Nparticles_val = 1000




def LogNormal(size, mu, gsd, N):
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (N / (np.sqrt(2 * pi) * size * np.log(gsd))) * np.exp(-1 * ((np.log(size) - np.log(mu)) ** 2) / (2 * np.log(gsd) ** 2))


coefficients, bins, counts = PMS.Mie_Lognormal(m=m_val, wavelength=wav_val, geoStdDev=geostdev_val, geoMean=geomean_val, numberOfParticles=Nparticles_val, nMedium=nmedium_val, lower=1, upper=1000, numberOfBins=10000, returnDistribution=True, asDict=True)
# units are in inverse meters originally for the cross section over the distribution
print('cross section dist: ', (coefficients['Bext']/np.sum(counts)) * 1E-8, 'cm^2')
x_sections = PMS.MieQ(m=m_val, wavelength=wav_val, diameter=geomean_val, nMedium=nmedium_val, asCrossSection=True, asDict=True)
print(coefficients)
# a cubic nanometer is 1E-14 cubic centimeters, units are in cubic nanometers originally for the cross section of the monodisperse calculation
print('cross section mono: ', x_sections['Cext'] * 1E-14, 'cm^2')
print(x_sections)
#print(len(bins))
#print(len(counts))

'''
for counter, element in enumerate(bins):
    if counter <= len(bins) - 2:
        print(bins[counter+1] - bins[counter])
'''

popt, pcov = curve_fit(LogNormal, bins, counts, p0=[800.00, 1.5, 1000])
print(popt)


'''
plt.bar(bins, counts, label='Distribution')
plt.plot(bins, LogNormal(bins, *popt), color='red', ls='-', label='fit')
plt.show()
'''

# this prooves that the number density essentially scales the data
bins_ln = bins
counts_ln_1 = LogNormal(size=bins_ln, mu=geomean_val, gsd=geostdev_val, N=100)
counts_ln_2 = LogNormal(size=bins_ln, mu=geomean_val, gsd=geostdev_val, N=1000)


theta_mie_1, SL_mie_1, SR_mie_1, SU_mie_1 = PMS.SF_SD(m=m_val, wavelength=wav_val, dp=bins_ln, ndp=counts_ln_1, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
theta_mie_1 = [x * 180.0/pi for x in theta_mie_1]
theta_mie_2, SL_mie_2, SR_mie_2, SU_mie_2 = PMS.SF_SD(m=m_val, wavelength=wav_val, dp=bins_ln, ndp=counts_ln_2, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
theta_mie_2 = [x * 180.0/pi for x in theta_mie_2]

f0, ax0 = plt.subplots(1, 3, figsize=(18, 6))
ax0[0].bar(bins_ln, counts_ln_1, color='red', label='N=100')
ax0[0].bar(bins_ln, counts_ln_2, color='blue', label='N=1000')
ax0[0].set_title('Distributions')
ax0[0].set_xlabel('Theta')
ax0[0].set_ylabel('Intensity')
ax0[0].legend(loc=1)
ax0[0].grid(True, which='both')
ax0[1].semilogy(theta_mie_1, SL_mie_1, color='red', ls='-', label='N=100')
ax0[1].semilogy(theta_mie_2, SL_mie_2, color='blue', ls='-', label='N=1000')
ax0[1].set_title('SL')
ax0[1].set_xlabel('Theta')
ax0[1].set_ylabel('Intensity')
ax0[1].legend(loc=1)
ax0[1].grid(True, which='both')
ax0[2].semilogy(theta_mie_1, SR_mie_1, color='red', ls='-', label='N=100')
ax0[2].semilogy(theta_mie_2, SR_mie_2, color='blue', ls='-', label='N=1000')
ax0[2].set_title('SR')
ax0[2].set_xlabel('Theta')
ax0[2].set_ylabel('Intensity')
ax0[2].legend(loc=1)
ax0[2].grid(True, which='both')
plt.show()