'''
Austen K. Scruggs
10-1-2018
Description: This is the first attempt at correcting the scattering diagram data for the lens
it utilizes the rayleigh scattering data.
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# import data
Rayleigh_Directory = '/home/austen/Documents/SD_avg_15s_N2.txt'
Save_Directory = '/home/austen/Documents/'
Rayleigh_Data = pd.read_csv(Rayleigh_Directory, delimiter=',', header=0)
Rayleigh_Int = Rayleigh_Data['SD N2']
Rayleigh_PN = np.arange(0, len(Rayleigh_Int), 1)

# fit to a quadratic
T1 = 100
T2 = 700
coeffs = np.polynomial.polynomial.polyfit(Rayleigh_PN[T1:T2], Rayleigh_Int[T1:T2], deg=2)
fit = np.polynomial.polynomial.polyval(Rayleigh_PN[T1:T2], coeffs)
fit_min = fit[np.argmin(fit)]
# parameters from linear calibration conducted like manfred et al.
slope = 0.206
intercept = 5.922
# fit the Delta Angle vs. Angle Function, note that Delta Angle 2 is the lens correction applied after the linear calibration
PN_To_Angle = Rayleigh_PN[T1:T2] * slope + intercept
Delta_Angle = (fit/fit_min) * slope - slope
coeffs2 = np.polynomial.polynomial.polyfit(PN_To_Angle, Delta_Angle, deg=2)
fit2 = np.polynomial.polynomial.polyval(PN_To_Angle, coeffs2)
Angle_Sign_Flip = PN_To_Angle[np.argmin(fit2)]
Delta_Angle_2 = []
for counter, element in enumerate(PN_To_Angle):
    if element > Angle_Sign_Flip:
        Delta_Angle_2.append(Delta_Angle[counter] * -1)
    else:
        Delta_Angle_2.append(Delta_Angle[counter])


# Make pretty plots
f0, ax0 = plt.subplots(1, 3, figsize=(10, 6))
ax0[0].plot(Rayleigh_PN, Rayleigh_Int, color='blue', linestyle='-', label='N2 15s Exposure Full')
ax0[0].plot(Rayleigh_PN[T1:T2], Rayleigh_Int[T1:T2], color='orange', linestyle='-', label='N2 15s Exposure Truncated')
ax0[0].plot(Rayleigh_PN[T1:T2], fit, color='red', linestyle='-', label='Polynomial Fit')
ax0[0].set_xlabel('Profile Number')
ax0[0].set_ylabel('Intensity (DN)')
ax0[0].set_title('Nitrogen Rayleigh Scattering Data')
ax0[0].legend(loc=1)
ax0[0].grid(True)
ax0[1].plot(Rayleigh_PN[T1:T2], Delta_Angle, color='purple', linestyle='-', label='\u0394\u00B0/\u0394PN')
ax0[1].set_xlabel('Profile Number')
ax0[1].set_ylabel('|\u0394\u00B0|')
ax0[1].set_title('|\u0394\u00B0| as a function of \u0394Profile Number')
ax0[1].grid(True)
ax0[1].legend(loc=1)
ax0[2].plot(PN_To_Angle, Delta_Angle_2, color='green', linestyle='-', label='\u0394\u00B0/\u0394PN')
ax0[2].set_xlabel('\u00B0')
ax0[2].set_ylabel('\u0394\u00B0')
ax0[2].set_title('\u0394\u00B0 as a function of \u0394Profile Number')
ax0[2].grid(True)
ax0[2].legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'Fig0.pdf', format='pdf')
plt.show()

