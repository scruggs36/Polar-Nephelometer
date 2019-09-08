'''
Austen K. Sruggs
date created:  01-23-2019
Description: script solves for local maxima and minima in
Mie theory and measured scattering diagrams and finds their
indicesThen plots the Mie theory angle of the local maxima
and minima as a function of the profile number, does a pixel to angle calibration
and then attempts to apply a lens correction to the scattering anlge axis by using
the rayleigh scattering data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import gridspec
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.signal import savgol_filter, argrelmax, argrelmin
from matplotlib.ticker import MultipleLocator

# import N2 Rayleigh scattering data
Save_Directory = '/home/austen/Documents/'
Fig_Directory = '/home/austen/Documents/' # save figures directory
'''
# import mie data and format into intensities, angles, and profile numbers into separate arrays
Mie_600_SD_Directory = '/home/austen/Documents/01-23-2019_Analysis/PSL600nm_MieTheory_2.txt'
Mie_600_SD_Data = pd.read_csv(Mie_600_SD_Directory, delimiter=',', header=0)
Mie_600_Intensity_Matheson = np.asarray(Mie_600_SD_Data['SU Matheson 1957'])
Mie_600_Intensity_Ma = np.asarray(Mie_600_SD_Data['SU Ma 2003'])
Mie_600_Intensity_Greenslade = np.asarray(Mie_600_SD_Data['SU Greenslade 2017'])
Mie_600_Angles_Matheson = np.asarray(Mie_600_SD_Data['Theta Matheson 1957'])
Mie_600_Angles_Ma = np.asarray(Mie_600_SD_Data['Theta Ma 2003'])
Mie_600_Angles_Greenslade = np.asarray(Mie_600_SD_Data['Theta Greenslade 2017'])

# import mie data and format into intensities, angles, and profile numbers into separate arrays
Mie_800_SD_Directory = '/home/austen/Documents/01-23-2019_Analysis/PSL800nm_MieTheory_2.txt'
Mie_800_SD_Data = pd.read_csv(Mie_800_SD_Directory, delimiter=',', header=0)
Mie_800_Intensity_Matheson = np.asarray(Mie_800_SD_Data['SU Matheson 1957'])
Mie_800_Intensity_Ma = np.asarray(Mie_800_SD_Data['SU Ma 2003'])
Mie_800_Intensity_Greenslade = np.asarray(Mie_800_SD_Data['SU Greenslade 2017'])
Mie_800_Angles_Matheson = np.asarray(Mie_800_SD_Data['Theta Matheson 1957'])
Mie_800_Angles_Ma = np.asarray(Mie_800_SD_Data['Theta Ma 2003'])
Mie_800_Angles_Greenslade = np.asarray(Mie_800_SD_Data['Theta Greenslade 2017'])
'''
# import mie data and format into intensities, angles, and profile numbers into separate arrays
Mie_900_SD_Directory = '/home/austen/Documents/01-23-2019_Analysis/PSL900nm_MieTheory_2.txt'
Mie_900_SD_Data = pd.read_csv(Mie_900_SD_Directory, delimiter=',', header=0)
Mie_900_Intensity_Matheson = np.asarray(Mie_900_SD_Data['SU Matheson 1957'])
Mie_900_Intensity_Ma = np.asarray(Mie_900_SD_Data['SU Ma 2003'])
Mie_900_Intensity_Greenslade = np.asarray(Mie_900_SD_Data['SU Greenslade 2017'])
Mie_900_Angles_Matheson = np.asarray(Mie_900_SD_Data['Theta Matheson 1957'])
Mie_900_Angles_Ma = np.asarray(Mie_900_SD_Data['Theta Matheson 1957'])
Mie_900_Angles_Greenslade = np.asarray(Mie_900_SD_Data['Theta Greenslade 2017'])
'''
# import experiment data for 600nm PSL
Exp_600_SD_Directory = '/home/austen/Documents/04-16-2019 Analysis/Phase Functions 2/SD_Particle_600nmPSL.txt'
Exp_600_SD_Data = pd.read_csv(Exp_600_SD_Directory, delimiter=',', header=0)
Ray_600_Int = Exp_600_SD_Data['N2 Intensity']
Ray_600_PN = np.asarray(Exp_600_SD_Data['N2 Columns'])
#PSL_600_Intensity = np.asarray(Exp_600_SD_Data['Sample Intensity'] - Exp_600_SD_Data['N2 Intensity'])
PSL_600_Intensity = np.asarray(Exp_600_SD_Data['Sample Intensity Corrected'])
PSL_600_Intensity = PSL_600_Intensity[~np.isnan(PSL_600_Intensity)]
PSL_600_PN = np.asarray(Exp_600_SD_Data['Sample Columns']) # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!
PSL_600_PN = PSL_600_PN[~np.isnan(PSL_600_PN)]
print(PSL_600_Intensity)
# import experiment data for 800nm PSL
Exp_800_SD_Directory = '/home/austen/Documents/04-16-2019 Analysis/Phase Functions 2/SD_Particle_800nmPSL.txt'
Exp_800_SD_Data = pd.read_csv(Exp_800_SD_Directory, delimiter=',', header=0)
Ray_800_Int = Exp_800_SD_Data['N2 Intensity']
Ray_800_PN = np.asarray(Exp_800_SD_Data['N2 Columns'])
#PSL_800_Intensity = np.asarray(Exp_800_SD_Data['Sample Intensity'] - Exp_800_SD_Data['N2 Intensity'])
PSL_800_Intensity = np.asarray(Exp_800_SD_Data['Sample Intensity Corrected'])
PSL_800_Intensity = PSL_800_Intensity[~np.isnan(PSL_800_Intensity)]
PSL_800_PN = np.asarray(Exp_800_SD_Data['Sample Columns']) # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!
PSL_800_PN = PSL_800_PN[~np.isnan(PSL_800_PN)]
'''
# import experiment data for 900nm PSL
Exp_900_SD_Directory = '/home/austen/Documents/04-16-2019 Analysis/Phase Functions 2/SD_Particle_900nmPSL.txt'
Exp_900_SD_Data = pd.read_csv(Exp_900_SD_Directory, delimiter=',', header=0)
Ray_900_Int = Exp_900_SD_Data['N2 Intensity']
Ray_900_PN = np.asarray(Exp_900_SD_Data['N2 Columns'])
#PSL_900_Intensity = np.asarray(Exp_900_SD_Data['Sample Intensity'] - Exp_900_SD_Data['N2 Intensity'])
PSL_900_Intensity = np.asarray(Exp_900_SD_Data['Sample Intensity Corrected'])
PSL_900_Intensity = PSL_900_Intensity[~np.isnan(PSL_900_Intensity)]
PSL_900_PN = Exp_900_SD_Data['Sample Columns'] # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!
PSL_900_PN = PSL_900_PN[~np.isnan(PSL_900_PN)]
# This is setting all the angle data to come from a specific group
Mie_600_Angles = Mie_600_Angles_Greenslade
Mie_800_Angles = Mie_800_Angles_Greenslade
Mie_900_Angles = Mie_900_Angles_Greenslade

# this is the Mie data coming from a single group that your comparing everything to
Mie_600_Intensity_Var = Mie_600_Intensity_Greenslade
Mie_800_Intensity_Var = Mie_800_Intensity_Greenslade
Mie_900_Intensity_Var = Mie_900_Intensity_Greenslade

# this makes the mie data the same array length as the experimental data
Mie_600_Pchip_Matheson = pchip_interpolate(Mie_600_Angles, Mie_600_Intensity_Matheson, PSL_600_PN, der=0, axis=0)
Mie_800_Pchip_Matheson = pchip_interpolate(Mie_800_Angles, Mie_800_Intensity_Matheson, PSL_800_PN, der=0, axis=0)
Mie_900_Pchip_Matheson = pchip_interpolate(Mie_900_Angles, Mie_900_Intensity_Matheson, PSL_900_PN, der=0, axis=0)

Mie_600_Pchip_Ma = pchip_interpolate(Mie_600_Angles, Mie_600_Intensity_Ma, PSL_600_PN, der=0, axis=0)
Mie_800_Pchip_Ma = pchip_interpolate(Mie_800_Angles, Mie_800_Intensity_Ma, PSL_800_PN, der=0, axis=0)
Mie_900_Pchip_Ma = pchip_interpolate(Mie_900_Angles, Mie_900_Intensity_Ma, PSL_900_PN, der=0, axis=0)

Mie_600_Pchip_Greenslade = pchip_interpolate(Mie_600_Angles, Mie_600_Intensity_Greenslade, PSL_600_PN, der=0, axis=0)
Mie_800_Pchip_Greenslade = pchip_interpolate(Mie_800_Angles, Mie_800_Intensity_Greenslade, PSL_800_PN, der=0, axis=0)
Mie_900_Pchip_Greenslade = pchip_interpolate(Mie_900_Angles, Mie_900_Intensity_Greenslade, PSL_900_PN, der=0, axis=0)

# smooth experimental scattering diagrams by savitzky golay to eliminate noise spikes!
PSL_600_Savgol = savgol_filter(PSL_600_Intensity, window_length=151, polyorder=2, deriv=0)
PSL_800_Savgol = savgol_filter(PSL_800_Intensity, window_length=151, polyorder=2, deriv=0)
PSL_900_Savgol = savgol_filter(PSL_900_Intensity, window_length=151, polyorder=2, deriv=0)

PSL_600_Savgol_Pchip = pchip_interpolate(PSL_600_PN, PSL_600_Savgol, PSL_600_PN, der=0, axis=0)
PSL_800_Savgol_Pchip = pchip_interpolate(PSL_800_PN, PSL_800_Savgol, PSL_800_PN, der=0, axis=0)
PSL_900_Savgol_Pchip = pchip_interpolate(PSL_900_PN, PSL_900_Savgol, PSL_900_PN, der=0, axis=0)


# find all local maxima and minima in the 600nm PSL Mie scattering diagram
print('600nm PSL Features(Index) and PN:')
Mie_600_Max = np.argmax(Mie_600_Intensity_Var)
print('Mie maximum: ', Mie_600_Max)
Mie_600_Local_Max = np.asarray(argrelmax(Mie_600_Intensity_Var, axis=0)).flatten()
print('Mie local max indices: ', Mie_600_Local_Max)
Mie_600_Local_Min = np.asarray(argrelmin(Mie_600_Intensity_Var, axis=0)).flatten()
print('Mie local min indices: ', Mie_600_Local_Min)
Mie_600_Local_Features = sorted(list(set(np.concatenate((Mie_600_Max, Mie_600_Local_Max, Mie_600_Local_Min), axis=None).ravel().tolist())))
del Mie_600_Local_Features[0]
#print('Mie local features: ', Mie_Local_Features)
Mie_600_Featured_Angles = [Mie_600_Angles[element] for element in Mie_600_Local_Features]
print('Mie featured angles: ', Mie_600_Featured_Angles)


# find all local maxima and minima for 600nm PSL in measured scattering diagram
Exp_600_Max = np.argmax(PSL_600_Savgol_Pchip)
print('Exp maximum: ', Exp_600_Max)
Exp_600_Local_Max = np.asarray(argrelmax(PSL_600_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local max indices: ', Exp_600_Local_Max)
Exp_600_Local_Min = np.asarray(argrelmin(PSL_600_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local min indices: ', Exp_600_Local_Min)
# note that Exp_Local_Features parses over an index that corresponds to the length of the SD array (0 ~ 790)
Exp_600_Local_Features = sorted(list(set(np.concatenate((Exp_600_Max, Exp_600_Local_Max, Exp_600_Local_Min), axis=None).ravel().tolist())))
print('All exp local features indexes: \n', Exp_600_Local_Features)
print('All exp local features length: ', len(Exp_600_Local_Features))
#drop = [0, 4, 5, 6, 7, 11, 12, 13, 14]
drop = [0]
for index in sorted(drop, reverse=True):
    del Exp_600_Local_Features[index]
#Exp_Local_Features.append(630)
#Exp_Local_Features.append(680)
Exp_600_Local_Features = sorted(Exp_600_Local_Features)
print('Kept local features: ', Exp_600_Local_Features)
print('Length del local features: ', len(Exp_600_Local_Features))
Exp_600_Local_PN = [PSL_600_PN[x] for x in Exp_600_Local_Features]
print('All exp local features pn: \n', Exp_600_Local_PN)
Features_600_PN = []
for element in Exp_600_Local_Features:
    Features_600_PN.append(PSL_900_PN[element])
# note Features PN corresponds to the actual index of the CCD, (200 ~ 1000)
print('Profile Numbers @ exp local features: ', Features_600_PN)

# pull Mie 600nm PSL intensities from local max and minima, create arrays
Mie_600_Intensities_at_Features = [Mie_600_Intensity_Var[element] for element in Mie_600_Local_Features]
# element in Exp_Local_Features or Features_PN
Exp_600_Intensities_at_Features = [PSL_600_Savgol[element] for element in Exp_600_Local_Features]
Mie_600_to_Exp_Intenisty_Ratios_at_Features = np.divide(np.array(Mie_600_Intensities_at_Features), np.array(Exp_600_Intensities_at_Features))
Ratio_600_Avg = np.average(Mie_600_to_Exp_Intenisty_Ratios_at_Features)

#'''
# plot 600 imported data
f0, ax0 = plt.subplots(1, 2, figsize=(10, 4))
ax0[0].plot(Mie_600_Angles, Mie_600_Intensity_Matheson, color='orange', ls='-', label='Matheson 1957 600nm PSL')
ax0[0].plot(Mie_600_Angles, Mie_600_Intensity_Ma, color='purple', ls='-', label='Ma 2003 600nm PSL')
ax0[0].plot(Mie_600_Angles, Mie_600_Intensity_Greenslade, color='black', ls='-', label='Greenslade 2017 600nm PSL')
ax0[0].set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax0[0].set_xlabel('Angles (\u00B0)')
ax0[0].set_ylabel('Intensity')
ax0[0].legend(loc=1)
ax0[0].set_yscale('log')
ax0[0].grid(True)
ax0[1].plot(PSL_600_PN, PSL_600_Intensity, color='blue', ls='-', label='Raw 600nm PSL')
ax0[1].plot(PSL_600_PN, PSL_600_Savgol, color='lawngreen', ls='-', label='Savgol 600nm PSL')
ax0[1].plot(PSL_600_PN, PSL_600_Savgol_Pchip, color='red', ls='-', label='Savgol + Pchip 600nm PSL')
ax0[1].plot(Exp_600_Local_PN, Exp_600_Intensities_at_Features, marker='*', ms='6', ls=' ', color='black', label='Local Max & Min')
ax0[1].set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax0[1].set_xlabel('Profile Number')
ax0[1].set_ylabel('Intensity')
ax0[1].set_yscale('log')
ax0[1].legend(loc=1)
ax0[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + '600nm_PSL.pdf', format='pdf')
plt.show()
#'''

# find all local maxima and minima in the 800nm PSL Mie scattering diagram
print('800nm PSL Features(Index) and PN:')
Mie_800_Max = np.argmax(Mie_800_Intensity_Var)
print('Mie maximum: ', Mie_800_Max)
Mie_800_Local_Max = np.asarray(argrelmax(Mie_800_Intensity_Var, axis=0)).flatten()
print('Mie local max indices: ', Mie_800_Local_Max)
Mie_800_Local_Min = np.asarray(argrelmin(Mie_800_Intensity_Var, axis=0)).flatten()
print('Mie local min indices: ', Mie_800_Local_Min)
Mie_800_Local_Features = sorted(list(set(np.concatenate((Mie_800_Max, Mie_800_Local_Max, Mie_800_Local_Min), axis=None).ravel().tolist())))
del Mie_800_Local_Features[0]
#print('Mie local features: ', Mie_Local_Features)
Mie_800_Featured_Angles = [Mie_800_Angles[element] for element in Mie_800_Local_Features]
print('Mie featured angles: ', Mie_800_Featured_Angles)

# find all local maxima and minima for 800nm PSL in measured scattering diagram
Exp_800_Max = np.argmax(PSL_800_Savgol_Pchip)
print('Exp maximum: ', Exp_800_Max)
Exp_800_Local_Max = np.asarray(argrelmax(PSL_800_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local max indices: ', Exp_800_Local_Max)
Exp_800_Local_Min = np.asarray(argrelmin(PSL_800_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local min indices: ', Exp_800_Local_Min)
# note that Exp_Local_Features parses over an index that corresponds to the length of the SD array (0 ~ 790)
Exp_800_Local_Features = sorted(list(set(np.concatenate((Exp_800_Max, Exp_800_Local_Max, Exp_800_Local_Min), axis=None).ravel().tolist())))
print('All exp local features indexes: \n', Exp_800_Local_Features)
print('All exp local features length: ', len(Exp_800_Local_Features))
drop = [0]
for index in sorted(drop, reverse=True):
    del Exp_800_Local_Features[index]
Exp_800_Local_Features.append(240)
Exp_800_Local_Features.append(343)
Exp_800_Local_Features = sorted(Exp_800_Local_Features)
Exp_800_Local_PN = [PSL_900_PN[x] for x in Exp_800_Local_Features]
print('All exp local features pn: \n', Exp_800_Local_PN)
print('Kept local features: ', Exp_800_Local_Features)
print('Length del local features: ', len(Exp_800_Local_Features))
Features_800_PN = []
for element in Exp_800_Local_Features:
    Features_800_PN.append(PSL_800_PN[element])
# note Features PN corresponds to the actual index of the CCD, (200 ~ 1000)
print('Profile Numbers @ exp local features: ', Features_800_PN)

# pull Mie 800nm PSL intensities from local max and minima, create arrays
Mie_800_Intensities_at_Features = [Mie_800_Intensity_Var[element] for element in Mie_800_Local_Features]
# element in Exp_Local_Features or Features_PN
Exp_800_Intensities_at_Features = [PSL_800_Savgol[element] for element in Exp_800_Local_Features]
Mie_800_to_Exp_Intenisty_Ratios_at_Features = np.divide(np.array(Mie_800_Intensities_at_Features), np.array(Exp_800_Intensities_at_Features))
Ratio_800_Avg = np.average(Mie_800_to_Exp_Intenisty_Ratios_at_Features)

#'''
# plot 800 imported data
f1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
ax1[0].plot(Mie_800_Angles, Mie_800_Intensity_Matheson, color='orange', ls='-', label='Matheson 1957 800nm PSL')
ax1[0].plot(Mie_800_Angles, Mie_800_Intensity_Ma, color='purple', ls='-', label='Ma 2003 800nm PSL')
ax1[0].plot(Mie_800_Angles, Mie_800_Intensity_Greenslade, color='black', ls='-', label='Greenslade 2017 800nm PSL')
ax1[0].set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax1[0].set_xlabel('Angles (\u00B0)')
ax1[0].set_ylabel('Intensity')
ax1[0].legend(loc=1)
ax1[0].set_yscale('log')
ax1[0].grid(True)
ax1[1].plot(PSL_800_PN, PSL_800_Intensity, color='blue', ls='-', label='Raw 800nm PSL')
ax1[1].plot(PSL_800_PN, PSL_800_Savgol, color='lawngreen', ls='-', label='Savgol 800nm PSL')
ax1[1].plot(PSL_800_PN, PSL_800_Savgol_Pchip, color='red', ls='-', label='Savgol + Pchip 800nm PSL')
ax1[1].plot(Exp_800_Local_PN, Exp_800_Intensities_at_Features, marker='*', ms=6, ls='', color='black', label='Local Max & Min')
ax1[1].set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax1[1].set_xlabel('Profile Number')
ax1[1].set_ylabel('Intensity')
ax1[1].set_yscale('log')
ax1[1].legend(loc=1)
ax1[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + '800nm_PSL.pdf', format='pdf')
plt.show()
#'''

# find all local maxima and minima in the 900nm PSL Mie scattering diagram
print('900nm PSL Features(Index) and PN:')
Mie_900_Max = np.argmax(Mie_900_Intensity_Var)
print('Mie maximum: ', Mie_900_Max)
Mie_900_Local_Max = np.asarray(argrelmax(Mie_900_Intensity_Var, axis=0)).flatten()
print('Mie local max indices: ', Mie_900_Local_Max)
Mie_900_Local_Min = np.asarray(argrelmin(Mie_900_Intensity_Var, axis=0)).flatten()
print('Mie local min indices: ', Mie_900_Local_Min)
Mie_900_Local_Features = sorted(list(set(np.concatenate((Mie_900_Max, Mie_900_Local_Max, Mie_900_Local_Min), axis=None).ravel().tolist())))
del Mie_900_Local_Features[0]
#print('Mie local features: ', Mie_Local_Features)
Mie_900_Featured_Angles = [Mie_900_Angles[element] for element in Mie_900_Local_Features]
print('Mie featured angles: ', Mie_900_Featured_Angles)

# find all local maxima and minima for 900nm PSL in measured scattering diagram
Exp_900_Max = np.argmax(PSL_900_Savgol_Pchip)
print('Exp maximum: ', Exp_900_Max)
Exp_900_Local_Max = np.asarray(argrelmax(PSL_900_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local max indices: ', Exp_900_Local_Max)
Exp_900_Local_Min = np.asarray(argrelmin(PSL_900_Savgol_Pchip, order=50, axis=0)).flatten()
print('Exp local min indices: ', Exp_900_Local_Min)
# note that Exp_Local_Features parses over an index that corresponds to the length of the SD array (0 ~ 790)
Exp_900_Local_Features = sorted(list(set(np.concatenate((Exp_900_Max, Exp_900_Local_Max, Exp_900_Local_Min), axis=None).ravel().tolist())))
print('All exp local features indexes: \n', Exp_900_Local_Features)
print('All exp local features length: ', len(Exp_900_Local_Features))
drop = [0]
for index in sorted(drop, reverse=True):
    del Exp_900_Local_Features[index]
Exp_900_Local_Features = sorted(Exp_900_Local_Features)
# had to add these in when the last local minima is too shallow
Exp_900_Local_Features.append(636)
Exp_900_Local_Features.append(696)
Exp_900_Local_PN = [PSL_900_PN[x] for x in Exp_900_Local_Features]
print('All exp local features pn: \n', Exp_900_Local_PN)
print('Kept local features: ', Exp_900_Local_Features)
print('Length del local features: ', len(Exp_900_Local_Features))
Features_900_PN = []
for element in Exp_900_Local_Features:
    Features_900_PN.append(PSL_900_PN[element])
# note Features PN corresponds to the actual index of the CCD, (200 ~ 1000)
print('Profile Numbers @ exp local features: ', Features_900_PN)
# pull Mie 900nm PSL intensities from local max and minima, create arrays
Mie_900_Intensities_at_Features = [Mie_900_Intensity_Var[element] for element in Mie_900_Local_Features]
# element in Exp_Local_Features or Features_PN
Exp_900_Intensities_at_Features = [PSL_900_Savgol[element] for element in Exp_900_Local_Features]
Mie_900_to_Exp_Intenisty_Ratios_at_Features = np.divide(np.array(Mie_900_Intensities_at_Features), np.array(Exp_900_Intensities_at_Features))
Ratio_900_Avg = np.average(Mie_900_to_Exp_Intenisty_Ratios_at_Features)

#'''
# plot 900 imported data
f2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
ax2[0].plot(Mie_900_Angles, Mie_900_Intensity_Matheson, color='orange', ls='-', label='Matheson 1957 900nm PSL')
ax2[0].plot(Mie_900_Angles, Mie_900_Intensity_Ma, color='purple', ls='-', label='Ma 2003 900nm PSL')
ax2[0].plot(Mie_900_Angles, Mie_900_Intensity_Greenslade, color='black', ls='-', label='Greenslade 2017 900nm PSL')
ax2[0].set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax2[0].set_xlabel('Angles (\u00B0)')
ax2[0].set_ylabel('Intensity')
ax2[0].legend(loc=1)
ax2[0].set_yscale('log')
ax2[0].grid(True)
ax2[1].plot(PSL_900_PN, PSL_900_Intensity, color='blue', ls='-', label='Raw 900nm PSL')
ax2[1].plot(PSL_900_PN, PSL_900_Savgol, color='lawngreen', ls='-', label='Savgol 900nm PSL')
ax2[1].plot(PSL_900_PN, PSL_900_Savgol_Pchip, color='red', ls='-', label='Savgol + Pchip 900nm PSL')
ax2[1].plot(Exp_900_Local_PN, Exp_900_Intensities_at_Features, marker='*', ls='', ms=6, color='black', label='Local Max & Min')
ax2[1].set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax2[1].set_xlabel('Profile Number')
ax2[1].set_ylabel('Intensity')
ax2[1].set_yscale('log')
ax2[1].legend(loc=1)
ax2[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + '900nm_PSL.pdf', format='pdf')
plt.show()
#'''

# create a 2d array of Mie theta and Exp PN at local features, combining all the 600, 800, and 900 psl data
All_PN = np.concatenate((Features_600_PN, Features_800_PN, Features_900_PN), axis=None).ravel().tolist()
All_Theta = np.concatenate((Mie_600_Featured_Angles, Mie_800_Featured_Angles, Mie_900_Featured_Angles), axis=None).ravel().tolist()
print('All_PN: ', All_PN)
print('All_Theta: ', All_Theta)

# OLS on all PSL data
All_PN_W_Const = sm.add_constant(All_PN) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model0 = sm.OLS(All_Theta, All_PN_W_Const)
results0 = model0.fit()
print(results0.summary())

# plotting all the local max & min on one plot and conducting an OLS on all features
f3, ax3 = plt.subplots(figsize=(12, 7))
ax3.plot(Features_600_PN, Mie_600_Featured_Angles, marker='o', ms='4', ls='', color='red', label='600nm PSL Local Max & Min')
ax3.plot(Features_800_PN, Mie_800_Featured_Angles, marker='^', ms='4', ls='', color='blue', label='800nm PSL Local Max & Min')
ax3.plot(Features_900_PN, Mie_900_Featured_Angles, marker='x', ms='4', ls='', color='green', label='900nm PSL Local Max & Min')
ax3.plot(All_PN, results0.fittedvalues, color='black', ls='-', label='OLS: y = ' + str('{:.4f}'.format(results0.params[1])) + 'x + ' + str('{:.4f}'.format(results0.params[0])))
ax3.grid(True)
ax3.set_title('Linear Regression to All Local Features in the PSL Data')
ax3.set_xlabel('PN')
ax3.set_ylabel('\u03b8 (\u00b0)')
ax3.legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'All_PSL_Calibration.pdf', format='pdf')
plt.show()

# do a linear fit to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
Exp_X_Vals_OLS = sm.add_constant(Features_900_PN) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model1 = sm.OLS(Mie_900_Featured_Angles, Exp_X_Vals_OLS)
results1 = model1.fit()
print(results1.summary())


#'''
# plot imported data
f4 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax4a = f4.add_subplot(gs[0, 0])
ax4a.plot(Mie_900_Angles, Mie_900_Intensity_Var, 'r-', label='900nm PSL')
ax4a.plot(Mie_900_Featured_Angles, Mie_900_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min')
ax4a.set_title('Mie Theory Calculated Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax4a.set_xlabel('Angles (\u00B0)')
ax4a.set_ylabel('Intensity')
ax4a.legend(loc=1)
ax4a.set_yscale('log')
ax4a.grid(True)
ax4b = f4.add_subplot(gs[0, 1])
ax4b.plot(PSL_900_PN, PSL_900_Intensity, 'b-', label='900nm PSL')
ax4b.plot(PSL_900_PN, PSL_900_Savgol, 'y-', label='900nm PSL Smoothed')
ax4b.plot(Features_900_PN, Exp_900_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax4b.set_title('Measured Scattering Diagram\n Circularly Polarized 663nm Radiation')
ax4b.set_xlabel('Profile Number')
ax4b.set_ylabel('Intensity')
ax4b.set_yscale('log')
ax4b.legend(loc=1)
ax4b.grid(True)
ax4c = f4.add_subplot(gs[1, 0])
ax4c.plot(Features_900_PN, Mie_900_Featured_Angles, marker='o', ls='', color='green', label='Angles vs Profile Numbers')
ax4c.plot(Features_900_PN, results1.fittedvalues, color='black', linestyle='-', label='OLS: y = ' + str('{:.4f}'.format(results1.params[1])) + 'x + ' + str('{:.4f}'.format(results1.params[0])))
ax4c.set_title('Scattering Angle as a Function of Profile Number')
ax4c.set_xlabel('Profile Number')
ax4c.set_ylabel('Scattering Angle (\u00B0)')
ax4c.legend(loc=2)
ax4c.grid(True)
ax4d = f4.add_subplot(gs[1, 1])
ax4d.plot(Mie_900_Featured_Angles, Mie_900_to_Exp_Intenisty_Ratios_at_Features, marker='^', color='yellow', label='Mie:Exp Intensity Ratio vs. Angle')
ax4d.set_title('Mie:Measured Intensity Ratio as a \n Function of Local Max/Min Angle')
ax4d.set_xlabel('Scattering Angle (\u00B0)')
ax4d.set_ylabel('Intensity Ratio')
ax4d.legend(loc=1)
ax4d.grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + '900nm_Calibration.pdf', format='pdf')
plt.show()
#'''

# Apply correction by adding the delta angle correction to the angle axis of the PSL scattering data

# Increase the resolution of the Mie data such that it is the same number of data points
# as the experimental data

Mie_600_Spline_Func = interp1d(Mie_600_Angles, Mie_600_Intensity_Var, kind='cubic')
Mie_600_Spline_Angles = np.linspace(0, 180, len(PSL_600_PN), endpoint=False)
Mie_600_Spline_Intensity = Mie_600_Spline_Func(Mie_600_Spline_Angles)

Mie_800_Spline_Func = interp1d(Mie_800_Angles, Mie_800_Intensity_Var, kind='cubic')
Mie_800_Spline_Angles = np.linspace(0, 180, len(PSL_800_PN), endpoint=False)
Mie_800_Spline_Intensity = Mie_800_Spline_Func(Mie_800_Spline_Angles)

Mie_900_Spline_Func = interp1d(Mie_900_Angles, Mie_900_Intensity_Var, kind='cubic')
Mie_900_Spline_Angles = np.linspace(0, 180, len(PSL_900_PN), endpoint=False)
Mie_900_Spline_Intensity = Mie_900_Spline_Func(Mie_900_Spline_Angles)

# First normalize the intensities and plot them one ontop of the other
Mie_600_Spline_Int_Norm = Mie_600_Spline_Intensity / np.linalg.norm(Mie_600_Spline_Intensity)
Mie_800_Spline_Int_Norm = Mie_800_Spline_Intensity / np.linalg.norm(Mie_800_Spline_Intensity)
Mie_900_Spline_Int_Norm = Mie_900_Spline_Intensity / np.linalg.norm(Mie_900_Spline_Intensity)

PSL_600_Savgol_Int_Norm = PSL_600_Savgol / np.linalg.norm(PSL_600_Savgol)
PSL_800_Savgol_Int_Norm = PSL_800_Savgol / np.linalg.norm(PSL_800_Savgol)
PSL_900_Savgol_Int_Norm = PSL_900_Savgol / np.linalg.norm(PSL_900_Savgol)

# parameters from linear calibration conducted like manfred et al.
slope = results0.params[1]
intercept = results0.params[0]
#slope = 0.2112
#intercept = -47.972

# convert profile numbers to angles with the OLS from the Manfred appraoach
PSL_600_Profiles_to_Angles = [(slope * x) + intercept for x in PSL_600_PN]
PSL_600_Profiles_to_Angles = np.array(PSL_600_Profiles_to_Angles)
print(type(PSL_600_Profiles_to_Angles), PSL_600_Profiles_to_Angles.shape)

PSL_800_Profiles_to_Angles = [(slope * x) + intercept for x in PSL_800_PN]
PSL_800_Profiles_to_Angles = np.array(PSL_800_Profiles_to_Angles)
print(type(PSL_800_Profiles_to_Angles), PSL_800_Profiles_to_Angles.shape)

PSL_900_Profiles_to_Angles = [(slope * x) + intercept for x in PSL_900_PN]
PSL_900_Profiles_to_Angles = np.array(PSL_900_Profiles_to_Angles)
print(type(PSL_900_Profiles_to_Angles), PSL_900_Profiles_to_Angles.shape)


# make a plot of the 900 spline data and the experimental data overlayed
f5, ax5 = plt.subplots(figsize=(12, 6))
#pt0, = ax5.plot(Mie_900_Spline_Angles, Mie_900_Spline_Intensity, color='red', linestyle='-', label='Mie 900nm PSL Spline vs. Theta')
pt = ax5.plot(PSL_900_Profiles_to_Angles, PSL_900_Savgol * Ratio_900_Avg, color='purple', linestyle='-', lw=4, label='Meas. 900nm PSL Savgol vs. Theta')
#pt2, = ax5.plot(PSL_900_Profiles_to_Angles, PSL_900_Savgol, color='green', linestyle='-', label='Exp 900nm PSL Savgol vs. Theta')
#ax5.set_xlabel('Angles (\u00B0)', color='red')
ax5.set_xlabel('Angles (\u00B0)', fontsize=20)
ax5.set_ylabel('Intensity', fontsize=20)
#ax5.set_title('900nm Polystyrene Latex Sphere Raw Phase Function to Calibrated Phase Function')
#ax5.tick_params(axis='x', labelcolor='red')
ax5.tick_params(axis='x')
ax5.minorticks_on()
ax5.grid(True, which='both')
ax5.set_yscale('log')
#ax5.legend(loc=1)
#ax5a = ax5.twiny()
#pt3, = ax5a.plot(PSL_900_PN, PSL_900_Savgol, color='blue', linestyle='-', label='Exp 900nm PSL Savgol vs. PN')
#ax5a.set_xlabel('Profile Numbers', color='blue')
#ax5a.set_ylabel('Intensity')
#ax5a.tick_params(axis='x', labelcolor='blue')
#pt = [pt0, pt1, pt2, pt3]
ax5.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
#ax5.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + '900nm_PSL_Calibrated.pdf', format='pdf')
plt.show()

C0 = pd.Series(Mie_900_Spline_Angles, name='Spline Mie Theta')
C1 = pd.Series(Mie_900_Spline_Intensity, name='Spline Mie Intensity')
C2 = pd.Series(PSL_900_PN, name='PN')
C3 = pd.Series(PSL_900_Profiles_to_Angles, name='PN to Angle')
C4 = pd.Series(PSL_900_Savgol, name='Exp Smoothed Intensity')
C5 = pd.Series(Features_900_PN, name='Cal Exp Profiles Local Max & Min')
C6 = pd.Series(Mie_900_Featured_Angles, name='Cal Mie Angles Local Max & Min')
C7 = pd.Series([(x * results0.fittedvalues[1]) + results0.fittedvalues[0] for x in Features_900_PN], name='Cal Fit Angles')
C8 = pd.Series(results0.fittedvalues[1], name='Cal Fit Slope')
C9 = pd.Series(results0.fittedvalues[0], name='Cal Fit Intercept')
All_Data = pd.concat([C0, C1, C2, C3, C4, C5, C6, C7, C8, C9], axis=1)
All_Data.to_csv(Save_Directory + '/Calibrated_Data_PSL900nm.txt')

# plot calibrated PSL 600nm data against theory
f6, ax6 = plt.subplots(figsize=(12, 6))
pt0, = ax6.plot(Mie_600_Spline_Angles, Mie_600_Spline_Intensity, color='red', linestyle='-', label='Mie 600nm PSL Spline vs. Theta')
pt1, = ax6.plot(PSL_600_Profiles_to_Angles, PSL_600_Savgol * Ratio_900_Avg * 0.30, color='purple', linestyle='-', label='Exp 600nm PSL Savgol Scaled vs. Theta')
pt2, = ax6.plot(PSL_600_Profiles_to_Angles, PSL_600_Savgol, color='green', linestyle='-', label='Exp 600nm PSL Savgol vs. Theta')
ax6.set_xlabel('Angles (\u00B0)', color='red')
ax6.set_ylabel('Intensity')
ax6.set_title('600nm Polystyrene Latex Sphere Raw Phase Function to Calibrated Phase Function')
ax6.tick_params(axis='x', labelcolor='red')
ax6.minorticks_on()
ax6.grid(True, which='both')
ax6.legend(loc=1)
ax6.set_yscale('log')
ax6a = ax6.twiny()
pt3, = ax6a.plot(PSL_600_PN, PSL_600_Savgol, color='blue', linestyle='-', label='Exp. Smoothed Intensity vs. PN')
ax6a.set_xlabel('Profile Numbers', color='blue')
ax6a.set_ylabel('Intensity')
ax6a.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2, pt3]
ax6.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax6.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + '600nm_PSL_Calibrated.pdf', format='pdf')
plt.show()


# plot calibrated PSL 800nm data against theory
f7, ax7 = plt.subplots(figsize=(12, 6))
pt0, = ax7.plot(Mie_800_Spline_Angles, Mie_800_Spline_Intensity, color='red', linestyle='-', label='Mie 800nm PSL Spline vs. Theta')
pt1, = ax7.plot(PSL_800_Profiles_to_Angles, PSL_800_Savgol * Ratio_900_Avg * 0.36, color='purple', linestyle='-', label='Exp 800nm PSL Savgol Scaled vs. Theta')
pt2, = ax7.plot(PSL_800_Profiles_to_Angles, PSL_800_Savgol, color='green', linestyle='-', label='Exp 800nm PSL Savgol vs. Theta')
ax7.set_xlabel('Angles (\u00B0)', color='red')
ax7.set_ylabel('Intensity')
ax7.set_title('800nm Polystyrene Latex Sphere Raw Phase Function to Calibrated Phase Function')
ax7.tick_params(axis='x', labelcolor='red')
ax7.minorticks_on()
ax7.grid(True, which='both')
ax7.legend(loc=1)
ax7.set_yscale('log')
ax7a = ax7.twiny()
pt3, = ax7a.plot(PSL_800_PN, PSL_800_Savgol, color='blue', linestyle='-', label='Exp. Smoothed Intensity vs. PN')
ax7a.set_xlabel('Profile Numbers', color='blue')
ax7a.set_ylabel('Intensity')
ax7a.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2, pt3]
ax7.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax7.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + '800nm_PSL_Calibrated.pdf', format='pdf')
plt.show()
