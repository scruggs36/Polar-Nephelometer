'''
Austen K. Scruggs
03/01/2019
Description: Calculates the phase functions for coated spheres
'''

import PyMieScatt as PMS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

save_directory = '/home/austen/Documents/'
# calculations for water coated PSL
# all parameters are in nanometers
d_Total = np.arange(903.0, 914.0, 1.0)
d_Core = 903.0
wav = 663.0
B9 = 1.4432
wav_c = 142.1
m_Core = (np.sqrt(1 + ((B9 * wav ** 2)/(wav ** 2 - wav_c ** 2)))) + 0.0003j
m_Shell = 1.5000 + 0.2j # water 1.331 + 0.0111j
# create a subplot
f0, ax0 = plt.subplots(figsize=(12, 7))
DF = pd.DataFrame()
# looping through different shell thicknesses, note the shell thickness cannot be zero!
for counter, element in enumerate(d_Total):
    if counter == 0:
        d_Shell = element - d_Core
        rads, SL, SR, SU = PMS.CoreShellScatteringFunction(mCore=m_Core, mShell=m_Shell, wavelength=wav, dCore=d_Core, dShell=element, minAngle=0, angularResolution=0.5, maxAngle=180, normed=False)
        theta = np.array(rads) * 180.0/pi
        DF['Theta'] = theta
        print(SU)
        DF['SU Core Diameter: ' + str(d_Core) + ' nm Coating: ' + str(d_Shell) + ' nm'] = SU
        ax0.semilogy(theta, SU, ls='-', label='dCore = ' + str(d_Core) + ', dShell = ' + str(d_Shell))
    if counter > 0:
        d_Shell = element - d_Core
        rads, SL, SR, SU = PMS.CoreShellScatteringFunction(mCore=m_Core, mShell=m_Shell, wavelength=wav, dCore=d_Core, dShell=element, minAngle=0, angularResolution=0.5, maxAngle=180, normed=False)
        theta = np.array(rads) * 180.0 / pi
        print(SU)
        DF['SU Core Diameter: ' + str(d_Core) + ' nm Coating: ' + str(d_Shell) + ' nm'] = SU
        ax0.semilogy(theta, SU, ls='-', label='dCore = ' + str(d_Core) + ', dShell = ' + str(d_Shell))

DF.to_csv(save_directory + 'coated_sphere_MT_0.1.txt', sep=',')

ax0.set_xlabel('\u0398')
ax0.set_ylabel('SU')
ax0.set_title('Phase Functions for Coated Polystyrene Latex Spheres at a Core Diameter of 903nm')
ax0.grid(True)
ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(save_directory + 'PF_Coated2.pdf', format='pdf')
plt.savefig(save_directory + 'PF_Coated2.png', format='png')
plt.show()

