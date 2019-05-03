'''
Austen K. Scruggs
04-09-2019
Description: I want to be able to calculate the rayleigh scattering phase function from
known rayleigh scattering theory.
'''

import PyMieScatt as ps
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

# this function is from the penendorf publication
#def Angular_Scattering_Cross_Section(theta, wav, Ns, ns, pn):
def Angular_Scattering_Cross_Section(theta, wav, ns, Ns, pn):
    rads = (theta * pi)/ 180
    sigma_angle_anisotropic = ((ns * (pi**2) * 2 * (2 + pn)) / (wav**4 * (Ns)**2 * (6 - 7 * pn))) * (0.7629 * (1 + 0.932 * cos(rads)**2))
    sigma_angle_isotropic = ((pi**2 / 2) * (ns) / (wav**4 * (Ns**2))) * (1 + cos(rads)**2)
    return sigma_angle_anisotropic, sigma_angle_isotropic

def Rayleigh_Scattering_Cross_Section(theta):
    rads = (theta * pi) / 180

# this function works correctly according to pennendorf
def Rayleigh_Phase_Function(theta):
    rads = (theta * pi) / 180
    pcos = 0.7629 * (1 + 0.932 * cos(rads)**2)
    return pcos



# intensity, power per unit area (W/cm^2)
I_0 = 0.1
# nitrogen optical parameters, for some reason the refractive index needs to be upped by 8 orders of mag in the paper
ns_n2 = (1.00029839**2 - 1)**2
print(ns_n2)
pn_n2 = .0305
# D. Spelsberg et al 1994
a_N2 = 11.74
# concentration
N = 2.54743E19
# angular range
angles = np.arange(0, 181, 1)
# wavelength for some reason on penndorf given in microns, it has to be converted to cm in the calculation
micron = .550
wav = micron * 10**-4
# air (n^2 - 1)^2, these are correct, when you calculate based on the formula
ns_air = (1.00027773**2 - 1)**2
print(ns_air)
pn_air = .035
R_dist = 1
# this was our check, the math is right finally!
#print(np.array([Rayleigh_Phase_Function(x) for x in angles]))
#print(np.array([Angular_Scattering_Cross_Section(x, wav, ns_air, N, pn_air) for x in angles]))

I1_n2 = np.array([Rayleigh_Gas1_Intensity(x, I_0, wav, a_N2, R_dist) for x in angles])
I2_n2 = np.array([Rayleigh_Gas2_Intensity(x, I_0, wav, a_N2) for x in angles])
sigma_anisotropic_theta = np.array([Angular_Scattering_Cross_Section(x, wav, ns_n2, N, pn_n2)[1] for x in angles])
print(sigma_anisotropic_theta)
sigma_isotropic_theta = np.array([Angular_Scattering_Cross_Section(x, wav, ns_n2, N, pn_n2)[0] for x in angles])

f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].plot(angles, I1_n2, 'r-', label='I1')
ax0[0].plot(angles, I2_n2, 'b-', label='I2')
ax0[1].plot(angles, sigma_isotropic_theta, color='orange', ls='-', label='isotropic')
ax0[1].plot(angles, sigma_anisotropic_theta, 'g-', label='anisotropic')
ax0[0].set_xlabel('\u0398')
ax0[0].set_ylabel('Intensity (W/M)')
ax0[0].set_title('Rayleigh Scattering Phase Function')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].set_xlabel('\u0398')
ax0[1].set_ylabel('$\u03c3_\u0398$')
ax0[1].set_title('Scattering Cross-Section \n as a Function of Angle')
ax0[1].grid(True)
ax0[1].legend(loc=1)
f0.savefig('/home/austen/Documents/Rayleigh_PF.png', format='png')
plt.show()

