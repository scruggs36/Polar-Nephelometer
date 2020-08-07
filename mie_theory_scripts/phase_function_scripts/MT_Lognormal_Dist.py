'''
Austen K. Scruggs
11-08-2019
Description: Squalane
'''


import numpy as np
import PyMieScatt as ps
from math import pi, sqrt, log10
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.signal import savgol_filter


Save_Directory = '/home/austen/Desktop/Recent/'
#Save_Mie = Save_Directory + '/Squalane900nm_MieTheory.txt'
#Data_Directory = '/home/austen/Documents/04-16-2019 Analysis/SD_Particle_803nmPSL.txt'



# function
def LogNormal(diam, mu, gsd, N):
    return N / (((2 * pi)**(1/2)) * diam * np.log(gsd)) * np.exp((-1 * ((np.log(diam) - np.log(mu)) ** 2)) / (2 * np.log(gsd) ** 2))

def Gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

def cauchy_4term(wav, A_4term, B_4term, C_4term):
    return A_4term + (B_4term / wav ** 2) + (C_4term / (wav ** 4))


# import experimental data
#Data = pd.read_csv(Data_Directory, sep=',', header=0)
# Particle diameter, geometric mean of the particle diameter
d = 900.0
# particle size standard deviation
sigma_s = 1.05
# define Gaussian function
# wavelength
w_n = 663
# CRI Squalane Cauchy Parameters
A = 1.43694
B = 3677.64
C = 8.28899E7
#m = cauchy_4term(663, A, B, C) + 0.00j
m = 1.525
print('Refractive Index: ', m)
# number density
N = 300


# size distribution plot
#size_axis = np.arange(1, 2000, 100), d - (sigma_s * 3) - 200
size_axis = np.arange(1, d + (sigma_s * 3) + 200, 1)
Size_Data = [LogNormal(x, mu=d, gsd=sigma_s, N=N) for x in size_axis]
#print(sp.integrate.simps(Gaussian(size_axis, mu=d, sigma=sigma_g), size_axis, dx=1))
f, ax = plt.subplots(figsize=(6, 6))
ax.plot(size_axis, Size_Data, 'b-', label='Gaussian Dist. \u03bc=' + str(d) + ', \u03c3=' + str(sigma_s))
ax.set_xlabel('particle diameter (nm)')
ax.set_ylabel('Normalized $dN/Log_{10}(D)$')
ax.set_title('Distributions Used for Mie Theory Calculations')
ax.grid(True)
plt.legend(loc=1)
plt.savefig(Save_Directory + 'Mie_Distributions_' + str(int(d)) + '.png', format='png')
plt.show()


Weights_Gaussian = []
size_array = []
weights_array = []
for counter, element in enumerate(range(len(Size_Data))):
    Weights_Gaussian.append([size_axis[counter], Size_Data[counter]])
    size_array.append(size_axis[counter])
    weights_array.append(Size_Data[counter])


SL_2darray = []
SR_2darray = []
SU_2darray = []
for element in size_array:
    theta, SL, SR, SU = ps.ScatteringFunction(m, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    SL_2darray.append(SL)
    SR_2darray.append(SR)
    SU_2darray.append(SU)

SL = np.average(SL_2darray, axis=0, weights=weights_array)
SR = np.average(SR_2darray, axis=0, weights=weights_array)
SU = np.average(SU_2darray, axis=0, weights=weights_array)

DF = pd.DataFrame()
DF['Theta'] = theta
DF['SL'] = SL
DF['SR'] = SR
DF['SU'] = SU
DF.to_csv(Save_Directory + 'MT_AS_' + str(int(d)) + '.txt')

fig3, ax3 = plt.subplots(figsize=(20, 7))
ax3.semilogy(theta, SL, ls='-', lw=1, label="SL")
ax3.semilogy(theta, SR, ls='-', lw=1, label="SR")
ax3.semilogy(theta, SU, ls='-', lw=1, label="SU")
ax3.set_xlabel("ϴ", fontsize=16)
ax3.set_ylabel(r"Intensity ($\mathregular{|S|^2}$)",fontsize=16,labelpad=10)
ax3.set_title('Phase Functions at Various Incident Polarizations of Light', fontsize=18)
ax3.grid(True)
ax3.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'MT' + str(int(d)) + '.png', format='png')
plt.show()
