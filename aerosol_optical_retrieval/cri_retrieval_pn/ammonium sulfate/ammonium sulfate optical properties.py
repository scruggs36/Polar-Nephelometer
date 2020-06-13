'''
Austen K. Scruggs
Date: 06/09/2020
Description: compiling literature values for ammonium sulfate
'''
# m = 1.52 ammonium sulfate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PyMieScatt as PMS
from math import sqrt, pi, log

directory = '/home/austen/Desktop/Recent/'

SL_DF = pd.read_csv(directory + 'SL_DF_NLLS_Tmat_Mie.txt', sep=',', header=0)
SR_DF = pd.read_csv(directory + 'SR_DF_NLLS_Tmat_Mie.txt', sep=',', header=0)

def LogNormal(size, mu, gsd, N):
    if size ==0.0:
        return 0.0
    else:
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))


# setting some constants
# diameter, radius, and  wavelength in nanometers
diameter_val = 903.0
gsd_val = 1.005
size_bins = np.arange(800.0, 1000.0, 1.0)
N_val = 1000
bin_counts = [LogNormal(size=i, mu=diameter_val, gsd=gsd_val, N=N_val) for i in size_bins]
wavelength_val = 663.0
m_val = complex(1.59, 0)
n_medium_val = 1.0
# 0.6 to less than 1.00 is a prolate top, greater than 1.00 to 1.66 is an oblate top, and 1.00 is spherical particle for ax_r
ax_r_val = 1.00
theta_scattered = np.arange(0.0, 180.5, 0.5)
theta_mie, SL_mie, SR_mie, SU_mie = PMS.SF_SD(m_val, wavelength=wavelength_val, dp=size_bins, ndp=bin_counts, nMedium=n_medium_val, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
theta_mie = np.array(theta_mie) * (180.0/pi)
SL_mie_norm = SL_mie
SR_mie_norm = SR_mie
#SL_mie_norm = SL_mie / np.sum(SL_mie)
#SR_mie_norm = SR_mie / np.sum(SR_mie)


soln_mie, bins_mie, counts_mie = PMS.Mie_Lognormal(m=m_val, wavelength=wavelength_val, geoStdDev=gsd_val, geoMean=diameter_val, nMedium=n_medium_val, numberOfParticles=N_val, lower=800.0, upper=1000.0, numberOfBins=201, asDict=True, returnDistribution=True)
print('-------Mie Results-------')
print('Mie ext coefficient: ', soln_mie['Bext'])
print('Mie ext x-section: ', (soln_mie['Bext']/1E8)/np.sum(counts_mie))
print('Mie sca coefficient: ', soln_mie['Bsca'])
print('Mie sca x-section: ', (soln_mie['Bsca']/1E8)/np.sum(counts_mie))
print('Mie SL Sum: ',np.sum(SL_mie_norm))

print(np.array(SL_DF['N']))



