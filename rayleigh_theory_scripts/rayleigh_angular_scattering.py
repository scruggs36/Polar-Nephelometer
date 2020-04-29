'''
Austen K. Scruggs
04-09-2019
Description: I want to be able to calculate the rayleigh scattering phase function from
known rayleigh scattering theory.
'''

import PyMieScatt as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos


# these functions integrate (0 - 2pi) to equal the same thing (rayleigh scattering cross section) as confirmed by wolfram
def Rayleigh_Gas1_Intensity(theta, I0, wav, alpha, R):
    rads = (theta * pi)/180
    I = I0 * (((8 * pi**4) * (alpha**2)) / (wav**4 * R**2)) * (1 + cos(rads)**2)
    return I

def Rayleigh_Gas2_Intensity(theta, I0, wav, alpha):
    rads = (theta * pi)/180
    I = I0 * (((128 * pi**5) * (alpha**2)) / (3 * wav**4)) * (3/(16 * pi)) * (1 + cos(rads)**2)
    return I


# this function works correctly according to pennendorf
def Rayleigh_Phase_Function(theta):
    rads = (theta * pi) / 180
    pcos = 0.7629 * (1 + 0.932 * cos(rads)**2)
    return pcos


def Angular_Scattering_Cross_Section(theta, wav, ns, Ns, pn):
    q = ((ns ** 2) - 1) ** 2
    anisotropic_coeff = (((pi**2) * q * 2 * (2 + pn)) / ((wav**4) * (Ns**2) * (6 - (7 * pn))))
    sigma_angle_anisotropic = anisotropic_coeff * Rayleigh_Phase_Function(theta)
    isotropic_coeff = (pi**2 / 2) * (q / (wav**4 * (Ns**2)))
    sigma_angle_isotropic = isotropic_coeff * (1 + cos((theta * pi)/ 180.0)**2)
    print('coefficients: ', [isotropic_coeff, anisotropic_coeff])
    return sigma_angle_anisotropic, sigma_angle_isotropic



# nitrogen optical parameters, for some reason the refractive index needs to be upped by 8 orders of mag in the paper
ns_n2 = 1.00029839**2
ns_minus_1_squared_n2 = (1.00029839**2 - 1)**2
print(ns_minus_1_squared_n2)
pn_n2 = .0305
# D. Spelsberg et al 1994
a_N2 = 11.74


# air (n^2 - 1)^2, these are correct, when you calculate based on the formula
ns_air = 1.00027773
ns_minus_1_squared_air = (1.00027773**2 - 1)**2
print(ns_minus_1_squared_air)
pn_air = .035
R_dist = 1


# other pertinent parameters
# concentration
N = 2.54743E19
# angular range
angles = np.arange(0, 181, 1)
# wavelength for some reason on penndorf given in microns, it has to be converted to cm in the calculation
micron = .550E-4
frequency = 1/(micron *10**-2)
wavenumbers = (((6.626E-34 * 3.0E8)/(micron * 10**-2))/1.62E-19) * 8065.54
print('wavenumbers: ', wavenumbers)

# carbon dioxide optical parameters
co2_kings = ((1.1364 + 25.3) * 10**-12) * frequency**2
ns_co2 = (1.1427E6 * ((5799.25/(128908.9**2 - frequency**2)) + (120.05/(89223.8**2 - frequency**2)) + (5.3334/(75037.5**2 - frequency**2)) + (4.3244/(67837.7**2 - frequency**2)) + (0.1218145E-4/(2418.136 - frequency**2))) + 1)
pn_co2 = .0805
print('n co2: ', ns_co2)

# this was our check, the math is right finally!
#I1_air = np.array([Rayleigh_Gas1_Intensity(x, I_0, micron, a_air, R_dist) for x in angles])
#I2_air = np.array([Rayleigh_Gas2_Intensity(x, I_0, micron, a_air) for x in angles])
#I1_n2 = np.array([Rayleigh_Gas1_Intensity(x, I_0, micron, a_N2, R_dist) for x in angles])
#I2_n2 = np.array([Rayleigh_Gas2_Intensity(x, I_0, micron, a_N2) for x in angles])
co2_sigma_anisotropic_theta = np.array([Angular_Scattering_Cross_Section(theta=x, wav=micron, ns=ns_co2, Ns=N, pn=pn_co2)[0] for x in angles])
co2_sigma_isotropic_theta = np.array([Angular_Scattering_Cross_Section(theta=x, wav=micron, ns=ns_co2, Ns=N, pn=pn_co2)[1] for x in angles])
air_sigma_anisotropic_theta = np.array([Angular_Scattering_Cross_Section(theta=x, wav=micron, ns=ns_air, Ns=N, pn=pn_air)[0] for x in angles])
air_sigma_isotropic_theta = np.array([Angular_Scattering_Cross_Section(theta=x, wav=micron, ns=ns_air, Ns=N, pn=pn_air)[1] for x in angles])
n2_sigma_anisotropic_theta = np.array([Angular_Scattering_Cross_Section(theta=x, wav=micron, ns=ns_n2, Ns=N, pn=pn_n2)[0] for x in angles])
n2_sigma_isotropic_theta = np.array([Angular_Scattering_Cross_Section(theta=x, wav=micron, ns=ns_n2, Ns=N, pn=pn_n2)[1] for x in angles])

print(air_sigma_anisotropic_theta)
# font sizes for figures
f_title = 24
f_axes = 18
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
f0, ax0 = plt.subplots(figsize=(12, 6))
#ax0.plot(angles, air_sigma_isotropic_theta, color='aqua', ls='-', label='Air Isotropic')
#ax0.plot(angles, air_sigma_anisotropic_theta, color='blue', label='Air Anisotropic')
#ax0.plot(angles, n2_sigma_isotropic_theta, color='lawngreen', ls='-', label='$N_2$ Isotropic')
#ax0.plot(angles, n2_sigma_anisotropic_theta, color='green', label='$N_2$ Anisotropic')
#ax0.plot(angles, co2_sigma_isotropic_theta, color='blue', ls='-', label='$CO_2$ Isotropic')
ax0.plot(angles, np.repeat(co2_sigma_anisotropic_theta[0], len(co2_sigma_anisotropic_theta)), color='red', ls='-', label='$CO_2$ Isotropic')
ax0.plot(angles, co2_sigma_anisotropic_theta, color='blue', label='$CO_2$ SL')
ax0.set_xlabel('\u0398')
ax0.set_ylabel('$\u03c3_\u0398$')
ax0.set_title('Gas Scattering Cross-Section as a Function of Light Scattering Angle')
ax0.grid(True)
ax0.legend(loc=1)
f0.savefig('/home/austen/Desktop/Recent/Rayleigh_PF.png', format='png')
plt.show()


DF = pd.DataFrame()
DF['Theta'] = angles
DF['CO2 isotropic'] = co2_sigma_isotropic_theta
DF['CO2 anisotropic'] = co2_sigma_anisotropic_theta
DF['N2 isotropic'] = n2_sigma_isotropic_theta
DF['N2 anisotropic'] = n2_sigma_anisotropic_theta
DF['Air isotropic'] = air_sigma_isotropic_theta
DF['Air anisotropic'] = air_sigma_anisotropic_theta
DF.to_csv('/home/austen/Desktop/Recent/Rayleigh_PF.txt', sep=',')