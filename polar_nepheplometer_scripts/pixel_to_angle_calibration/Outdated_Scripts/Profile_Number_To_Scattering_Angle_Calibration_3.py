'''
Austen K. Sruggs
date created:  10-04-2018
Description: script solves for local maxima and minima in
Mie theory and measured scattering diagrams and finds their
indicesThen plots the Mie theory angle of the local maxima
and minima as a function of the profile number, does a pixel to angle calibration
and then attempts to apply a lens correction to the scattering anlge axis by using the rayleigh scattering data
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from matplotlib import gridspec
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter, argrelmax, argrelmin
from matplotlib.ticker import MultipleLocator

# import N2 Rayleigh scattering data
Save_Directory = '/home/austen/Documents/'
Fig_Directory = '/home/austen/Documents/' # save figures directory


# import mie data and format into intensities, angles, and profile numbers into separate arrays
Mie_SD_Directory = '/home/austen/Documents/PSL900nm_MieTheory.txt'
Mie_SD_Data = pd.read_csv(Mie_SD_Directory, delimiter=',', header=0)
Mie_Intensity = np.asarray(Mie_SD_Data['SU'])
Mie_Angles = np.asarray(Mie_SD_Data['Theta'])

# import experiment data
Exp_SD_Directory = '/home/austen/Documents/Good_Data/PSL_900nm_T5/SD_Offline.txt'
Exp_SD_Data = pd.read_csv(Exp_SD_Directory, delimiter=',', header=0)
Rayleigh_Int = Exp_SD_Data['Nitrogen Intensity']
Rayleigh_PN = np.asarray(Exp_SD_Data['Columns'])
Exp_Intensity = np.asarray(Exp_SD_Data['Sample Intensity'] - Exp_SD_Data['Nitrogen Intensity'])
Exp_PN = Exp_SD_Data['Columns'] # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!


# smooth experimental scattering diagrams by savitzky golay to eliminate noise spikes!
Exp_Smoothed_Intensity = savgol_filter(Exp_Intensity, window_length=151, polyorder=2, deriv=0)

#'''
# plot imported data
f0, ax0 = plt.subplots(1, 2, figsize=(10, 4))
ax0[0].plot(Mie_Angles, Mie_Intensity, 'r-', label='900nm PSL')
ax0[0].set_title('Mie Theory Calculated Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax0[0].set_xlabel('Angles (\u00B0)')
ax0[0].set_ylabel('Intensity')
ax0[0].legend(loc=1)
ax0[0].set_yscale('log')
ax0[0].grid(True)
ax0[1].plot(Exp_PN, Exp_Intensity, 'b-', label='900nm PSL')
ax0[1].plot(Exp_PN, Exp_Smoothed_Intensity, 'y-', label='900nm PSL Smoothed')
ax0[1].set_title('Measured Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax0[1].set_xlabel('Profile Number')
ax0[1].set_ylabel('Intensity')
ax0[1].set_yscale('log')
ax0[1].legend(loc=1)
ax0[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F0.pdf', format='pdf')
plt.show()
#'''


# find all local maxima and minima in Mie scattering diagram
Mie_Max = np.argmax(Mie_Intensity)
print('Mie maximum: ', Mie_Max)
Mie_Local_Max = np.asarray(argrelmax(Mie_Intensity, axis=0)).flatten()
print('Mie local max indices: ', Mie_Local_Max)
Mie_Local_Min = np.asarray(argrelmin(Mie_Intensity, axis=0)).flatten()
print('Mie local min indices: ', Mie_Local_Min)
Mie_Local_Features = sorted(list(set(np.concatenate((Mie_Max, Mie_Local_Max, Mie_Local_Min), axis=None).ravel().tolist())))
del Mie_Local_Features[0]
#print('Mie local features: ', Mie_Local_Features)
Mie_Featured_Angles = [Mie_Angles[element] for element in Mie_Local_Features]
print('Mie featured angles: ', Mie_Featured_Angles)

# find all local maxima and minima in measured scattering diagram
Exp_Max = np.argmax(Exp_Smoothed_Intensity)
print('Exp maximum: ', Exp_Max)
Exp_Local_Max = np.asarray(argrelmax(Exp_Smoothed_Intensity, axis=0)).flatten()
print('Exp local max indices: ', Exp_Local_Max)
Exp_Local_Min = np.asarray(argrelmin(Exp_Smoothed_Intensity, axis=0)).flatten()
print('Exp local min indices: ', Exp_Local_Min)
Exp_Local_Features = sorted(list(set(np.concatenate((Exp_Max, Exp_Local_Max, Exp_Local_Min), axis=None).ravel().tolist())))
drop = [0, 1, 2, 3, 4, 5, 7, 8]
for index in sorted(drop, reverse=True):
    del Exp_Local_Features[index]
print('Exp local features: ', Exp_Local_Features)

Features_PN = []
for element in Exp_Local_Features:
    Features_PN.append(Exp_PN[element])

# pull intensities from local max and minima, create arrays
Mie_Intensities_at_Features = [Mie_Intensity[element] for element in Mie_Local_Features]
Exp_Intensities_at_Features = [Exp_Smoothed_Intensity[element] for element in Exp_Local_Features]
Mie_to_Exp_Intenisty_Ratios_at_Features = np.divide(np.array(Mie_Intensities_at_Features), np.array(Exp_Intensities_at_Features))
Ratio_Avg = np.average(Mie_to_Exp_Intenisty_Ratios_at_Features)

# do a linear fit to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
Exp_X_Vals_OLS = sm.add_constant(np.add(Exp_Local_Features, Exp_PN[0])) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model0 = sm.OLS(Mie_Featured_Angles, Exp_X_Vals_OLS)
results0 = model0.fit()
print(results0.summary())

#'''
# plot imported data
f1 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax1a = f1.add_subplot(gs[0, 0])
ax1a.plot(Mie_Angles, Mie_Intensity, 'r-', label='900nm PSL')
ax1a.plot(Mie_Featured_Angles, Mie_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min')
ax1a.set_title('Mie Theory Calculated Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1a.set_xlabel('Angles (\u00B0)')
ax1a.set_ylabel('Intensity')
ax1a.legend(loc=1)
ax1a.set_yscale('log')
ax1a.grid(True)
ax1b = f1.add_subplot(gs[0, 1])
ax1b.plot(Exp_PN, Exp_Intensity, 'b-', label='900nm PSL')
ax1b.plot(Exp_PN, Exp_Smoothed_Intensity, 'y-', label='900nm PSL Smoothed')
ax1b.plot(np.add(Exp_Local_Features, Exp_PN[0]), Exp_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax1b.set_title('Measured Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1b.set_xlabel('Profile Number')
ax1b.set_ylabel('Intensity')
ax1b.set_yscale('log')
ax1b.legend(loc=1)
ax1b.grid(True)
ax1c = f1.add_subplot(gs[1, 0])
ax1c.plot(np.add(Exp_Local_Features, Exp_PN[0]), Mie_Featured_Angles, marker='X', color='black', label='Angles vs Profile Numbers')
ax1c.plot(np.add(Exp_Local_Features, Exp_PN[0]), results0.fittedvalues, color='green', linestyle='--', label='OLS: y = ' + str('{:.3f}'.format(results0.params[1])) + 'x + ' + str('{:.3f}'.format(results0.params[0])))
ax1c.set_title('Scattering Angle as a Function of Profile Number')
ax1c.set_xlabel('Profile Number')
ax1c.set_ylabel('Scattering Angle (\u00B0)')
ax1c.legend(loc=2)
ax1c.grid(True)
ax1d = f1.add_subplot(gs[1, 1])
ax1d.plot(Mie_Featured_Angles, Mie_to_Exp_Intenisty_Ratios_at_Features, marker='^', color='yellow', label='Mie:Exp Intensity Ratio vs. Angle')
ax1d.set_title('Mie:Measured Intensity Ratio as a \n Function of Local Max/Min Angle')
ax1d.set_xlabel('Scattering Angle (\u00B0)')
ax1d.set_ylabel('Intensity Ratio')
ax1d.legend(loc=1)
ax1d.grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F1.pdf', format='pdf')
plt.show()
#'''


# Apply correction by adding the delta angle correction to the angle axis of the PSL scattering data

# Increase the resolution of the Mie data such that it is the same number of data points
# as the experimental data
Mie_Spline_Func = interp1d(Mie_Angles, Mie_Intensity, kind='cubic')
Mie_Spline_Angles = np.linspace(0, 180, len(Exp_PN), endpoint=False)
Mie_Spline_Intensity = Mie_Spline_Func(Mie_Spline_Angles)

# First normalize the intensities and plot them one ontop of the other
Mie_Spline_Intensity_Normalized = Mie_Spline_Intensity / np.linalg.norm(Mie_Spline_Intensity)
Exp_Smoothed_Intensity_Normalized = Exp_Smoothed_Intensity / np.linalg.norm(Exp_Smoothed_Intensity)

# convert profile numbers to angles with the OLS from the Manfred appraoach
Exp_Profiles_to_Angles = [results0.params[1] * x + results0.params[0] for x in Exp_PN]
Exp_Profiles_to_Angles = np.array(Exp_Profiles_to_Angles)
print(type(Exp_Profiles_to_Angles), Exp_Profiles_to_Angles.shape)

# parameters from linear calibration conducted like manfred et al.
slope = results0.params[1]
intercept = results0.params[0]

# make a plot of the spline data and the experimental data overlayed
f3, ax3a = plt.subplots(figsize=(12, 6))
pt0, = ax3a.plot(Mie_Spline_Angles, Mie_Spline_Intensity, color='red', linestyle='-', label='Mie Spline Intensity vs. Theta')
pt1, = ax3a.plot(Exp_Profiles_to_Angles, Exp_Smoothed_Intensity * Ratio_Avg, color='purple', linestyle='-', label='Exp Smoothed Intensity Scaled vs. Theta')
pt2, = ax3a.plot(Exp_Profiles_to_Angles, Exp_Smoothed_Intensity, color='green', linestyle='-', label='Exp Smoothed Intensity vs. Theta')
ax3a.set_xlabel('Angles (\u00B0)', color='red')
ax3a.set_ylabel('Intensity')
ax3a.tick_params(axis='x', labelcolor='red')
ax3a.minorticks_on()
ax3a.grid(True, which='both')
ax3a.legend(loc=1)
ax3a.set_yscale('log')
ax3b = ax3a.twiny()
pt3, = ax3b.plot(Exp_PN, Exp_Smoothed_Intensity, color='blue', linestyle='-', label='Exp. Smoothed Intensity vs. PN')
ax3b.set_xlabel('Profile Numbers', color='blue')
ax3b.set_ylabel('Intensity')
ax3b.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2, pt3]
ax3a.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, bbox_to_anchor=[1.35, 1], fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax3a.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F3.pdf', format='pdf')
plt.show()

All_Data = pd.DataFrame()
All_Data['Spline Mie Theta'] = Mie_Spline_Angles
All_Data['Spline Mie Intensity'] = Mie_Spline_Intensity
All_Data['PN'] = Exp_PN
All_Data['PN to Angle'] = Exp_Profiles_to_Angles
All_Data['Exp Smoothed Intensity'] = Exp_Smoothed_Intensity
All_Data.to_csv(Save_Directory + '/Calibrated_Data_PSL900nm.txt')





