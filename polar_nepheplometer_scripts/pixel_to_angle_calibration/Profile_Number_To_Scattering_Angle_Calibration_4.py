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
from scipy.interpolate import interp1d, pchip_interpolate
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

# import mie data and format into intensities, angles, and profile numbers into separate arrays
MieVal_SD_Directory = '/home/austen/Documents/PSL600nm_MieTheory.txt'
MieVal_SD_Data = pd.read_csv(MieVal_SD_Directory, delimiter=',', header=0)
MieVal_Intensity = np.asarray(MieVal_SD_Data['SU'])
MieVal_Angles = np.asarray(MieVal_SD_Data['Theta'])

# import experiment data for 900nm PSL
Exp_SD_Directory = '/home/austen/Documents/Good_Data/PSL_900nm_T6/6s/SD_Offline.txt'
Exp_SD_Data = pd.read_csv(Exp_SD_Directory, delimiter=',', header=0)
Rayleigh_Int = Exp_SD_Data['Nitrogen Intensity']
Rayleigh_PN = np.asarray(Exp_SD_Data['Columns'])
Exp_Intensity = np.asarray(Exp_SD_Data['Sample Intensity'] - Exp_SD_Data['Nitrogen Intensity'])
Exp_PN = Exp_SD_Data['Columns'] # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!

# import experiment data for 600nm PSL
Val_SD_Directory = '/home/austen/Documents/Good_Data/PSL_600nm_T1/6s/SD_Offline_6s.txt'
Val_SD_Data = pd.read_csv(Val_SD_Directory, delimiter=',', header=0)
Ray_Val_Int = Val_SD_Data['Nitrogen Intensity']
Ray_Val_PN = np.asarray(Val_SD_Data['Columns'])
Val_Intensity = np.asarray(Val_SD_Data['Sample Intensity'] - Val_SD_Data['Nitrogen Intensity'])
Val_PN = np.asarray(Val_SD_Data['Columns']) # the actual profile number needs to be added into the labview code, it is in the Python Offline Analysis!

# smooth experimental scattering diagrams by savitzky golay to eliminate noise spikes!
Mie_Pchip = pchip_interpolate(Mie_Angles, Mie_Intensity, Exp_PN, der=0, axis=0)

Exp_Smoothed_Intensity = savgol_filter(Exp_Intensity, window_length=151, polyorder=2, deriv=0)
Exp_Pchip = pchip_interpolate(Exp_PN, Exp_Smoothed_Intensity, Exp_PN, der=0, axis=0)
Val_Smoothed_Intensity = savgol_filter(Val_Intensity, window_length=151, polyorder=2, deriv=0)
Val_Pchip = pchip_interpolate(Val_PN, Val_Smoothed_Intensity, Exp_PN, der=0, axis=0)

#'''
# plot imported data
f0, ax0 = plt.subplots(1, 2, figsize=(10, 4))
ax0[0].plot(Mie_Angles, Mie_Intensity, color='r', ls='-', label='900nm PSL')
ax0[0].set_title('Mie Theory Calculated Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax0[0].set_xlabel('Angles (\u00B0)')
ax0[0].set_ylabel('Intensity')
ax0[0].legend(loc=1)
ax0[0].set_yscale('log')
ax0[0].grid(True)
ax0[1].plot(Exp_PN, Exp_Intensity, color='blue', ls='-', label='900nm PSL')
ax0[1].plot(Exp_PN, Exp_Smoothed_Intensity, color='yellow', ls='-', label='900nm PSL Smoothed')
ax0[1].plot(Exp_PN, Exp_Pchip, color='orange', ls='-', label='900nm PSL Pchip')
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

#'''
# plot imported data
f1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
ax1[0].plot(MieVal_Angles, MieVal_Intensity, color='r', ls='-', label='600nm PSL')
ax1[0].set_title('Mie Theory Calculated Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1[0].set_xlabel('Angles (\u00B0)')
ax1[0].set_ylabel('Intensity')
ax1[0].legend(loc=1)
ax1[0].set_yscale('log')
ax1[0].grid(True)
ax1[1].plot(Val_PN, Val_Intensity, color='blue', ls='-', label='600nm PSL')
ax1[1].plot(Val_PN, Val_Smoothed_Intensity, color='yellow', ls='-', label='600nm PSL Smoothed')
ax1[1].plot(Exp_PN, Val_Pchip, color='orange', ls='-', label='600nm PSL Pchip')
ax1[1].set_title('Measured Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1[1].set_xlabel('Profile Number')
ax1[1].set_ylabel('Intensity')
ax1[1].set_yscale('log')
ax1[1].legend(loc=1)
ax1[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F1.pdf', format='pdf')
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
Exp_Max = np.argmax(Exp_Pchip)
print('Exp maximum: ', Exp_Max)
Exp_Local_Max = np.asarray(argrelmax(Exp_Pchip, axis=0)).flatten()
print('Exp local max indices: ', Exp_Local_Max)
Exp_Local_Min = np.asarray(argrelmin(Exp_Pchip, axis=0)).flatten()
print('Exp local min indices: ', Exp_Local_Min)
# note that Exp_Local_Features parses over an index that corresponds to the length of the SD array (0 ~ 790)
Exp_Local_Features = sorted(list(set(np.concatenate((Exp_Max, Exp_Local_Max, Exp_Local_Min), axis=None).ravel().tolist())))
print('All exp local features indexes: \n', Exp_Local_Features)
Exp_Local_PN = [Exp_PN[x] for x in Exp_Local_Features]
print('All exp local features pn: \n', Exp_Local_PN)
print('All exp local features length: ', len(Exp_Local_Features))
drop = [0, 12, 11, 10, 9, 8, 7]
for index in sorted(drop, reverse=True):
    del Exp_Local_Features[index]
#Exp_Local_Features.append(630)
#Exp_Local_Features.append(680)
Exp_Local_Features = sorted(Exp_Local_Features)


print('Kept local features: ', Exp_Local_Features)
print('Length del local features: ', len(Exp_Local_Features))

Features_PN = []
for element in Exp_Local_Features:
    Features_PN.append(Exp_PN[element])

# note Features PN corresponds to the actual index of the CCD, (200 ~ 1000)
print('Profile Numbers @ exp local features: ', Features_PN)

# pull intensities from local max and minima, create arrays
Mie_Intensities_at_Features = [Mie_Intensity[element] for element in Mie_Local_Features]
# element in Exp_Local_Features or Features_PN
Exp_Intensities_at_Features = [Exp_Smoothed_Intensity[element] for element in Exp_Local_Features]
Mie_to_Exp_Intenisty_Ratios_at_Features = np.divide(np.array(Mie_Intensities_at_Features), np.array(Exp_Intensities_at_Features))
Ratio_Avg = np.average(Mie_to_Exp_Intenisty_Ratios_at_Features)

# do a linear fit to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
Exp_X_Vals_OLS = sm.add_constant(Features_PN) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model0 = sm.OLS(Mie_Featured_Angles, Exp_X_Vals_OLS)
results0 = model0.fit()
print(results0.summary())

#'''
# plot imported data
f2 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax2a = f2.add_subplot(gs[0, 0])
ax2a.plot(Mie_Angles, Mie_Intensity, 'r-', label='900nm PSL')
ax2a.plot(Mie_Featured_Angles, Mie_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min')
ax2a.set_title('Mie Theory Calculated Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax2a.set_xlabel('Angles (\u00B0)')
ax2a.set_ylabel('Intensity')
ax2a.legend(loc=1)
ax2a.set_yscale('log')
ax2a.grid(True)
ax2b = f2.add_subplot(gs[0, 1])
ax2b.plot(Exp_PN, Exp_Intensity, 'b-', label='900nm PSL')
ax2b.plot(Exp_PN, Exp_Smoothed_Intensity, 'y-', label='900nm PSL Smoothed')
ax2b.plot(Features_PN, Exp_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax2b.set_title('Measured Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax2b.set_xlabel('Profile Number')
ax2b.set_ylabel('Intensity')
ax2b.set_yscale('log')
ax2b.legend(loc=1)
ax2b.grid(True)
ax2c = f2.add_subplot(gs[1, 0])
ax2c.plot(Features_PN, Mie_Featured_Angles, marker='X', color='black', label='Angles vs Profile Numbers')
ax2c.plot(Features_PN, results0.fittedvalues, color='green', linestyle='--', label='OLS: y = ' + str('{:.4f}'.format(results0.params[1])) + 'x + ' + str('{:.4f}'.format(results0.params[0])))
ax2c.set_title('Scattering Angle as a Function of Profile Number')
ax2c.set_xlabel('Profile Number')
ax2c.set_ylabel('Scattering Angle (\u00B0)')
ax2c.legend(loc=2)
ax2c.grid(True)
ax2d = f2.add_subplot(gs[1, 1])
ax2d.plot(Mie_Featured_Angles, Mie_to_Exp_Intenisty_Ratios_at_Features, marker='^', color='yellow', label='Mie:Exp Intensity Ratio vs. Angle')
ax2d.set_title('Mie:Measured Intensity Ratio as a \n Function of Local Max/Min Angle')
ax2d.set_xlabel('Scattering Angle (\u00B0)')
ax2d.set_ylabel('Intensity Ratio')
ax2d.legend(loc=1)
ax2d.grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F2.pdf', format='pdf')
plt.show()
#'''

# Apply correction by adding the delta angle correction to the angle axis of the PSL scattering data

# Increase the resolution of the Mie data such that it is the same number of data points
# as the experimental data
Mie_Spline_Func = interp1d(Mie_Angles, Mie_Intensity, kind='cubic')
Mie_Spline_Angles = np.linspace(0, 180, len(Exp_PN), endpoint=False)
Mie_Spline_Intensity = Mie_Spline_Func(Mie_Spline_Angles)

MieVal_Spline_Func = interp1d(MieVal_Angles, MieVal_Intensity, kind='cubic')
MieVal_Spline_Angles = np.linspace(0, 180, len(Val_PN), endpoint=False)
MieVal_Spline_Intensity = MieVal_Spline_Func(MieVal_Spline_Angles)

# First normalize the intensities and plot them one ontop of the other
Mie_Spline_Intensity_Normalized = Mie_Spline_Intensity / np.linalg.norm(Mie_Spline_Intensity)
Exp_Smoothed_Intensity_Normalized = Exp_Smoothed_Intensity / np.linalg.norm(Exp_Smoothed_Intensity)

# convert profile numbers to angles with the OLS from the Manfred appraoach
Exp_Profiles_to_Angles = [results0.params[1] * x + results0.params[0] for x in Exp_PN]
Exp_Profiles_to_Angles = np.array(Exp_Profiles_to_Angles)
print(type(Exp_Profiles_to_Angles), Exp_Profiles_to_Angles.shape)

Val_Profiles_to_Angles = [results0.params[1] * x + results0.params[0] for x in Val_PN]
Val_Profiles_to_Angles = np.array(Val_Profiles_to_Angles)
print(type(Val_Profiles_to_Angles), Val_Profiles_to_Angles.shape)

# parameters from linear calibration conducted like manfred et al.
#slope = results0.params[1]
#intercept = results0.params[0]
slope = 0.2112
intercept = -47.972

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

C0 = pd.Series(Mie_Spline_Angles, name='Spline Mie Theta')
C1 = pd.Series(Mie_Spline_Intensity, name='Spline Mie Intensity')
C2 = pd.Series(Exp_PN, name='PN')
C3 = pd.Series(Exp_Profiles_to_Angles, name='PN to Angle')
C4 = pd.Series(Exp_Smoothed_Intensity, name='Exp Smoothed Intensity')
C5 = pd.Series(Features_PN, name='Cal Exp Profiles Local Max & Min')
C6 = pd.Series(Mie_Featured_Angles, name='Cal Mie Angles Local Max & Min')
C7 = pd.Series([(x * results0.fittedvalues[1]) + results0.fittedvalues[0] for x in Features_PN], name='Cal Fit Angles')
C8 = pd.Series(results0.fittedvalues[1], name='Cal Fit Slope')
C9 = pd.Series(results0.fittedvalues[0], name='Cal Fit Intercept')
All_Data = pd.concat([C0, C1, C2, C3, C4, C5, C6, C7, C8, C9], axis=1)
All_Data.to_csv(Save_Directory + '/Calibrated_Data_PSL900nm.txt')

# plot calibrated PSL 600nm data against theory
f4, ax4a = plt.subplots(figsize=(12, 6))
pt0, = ax4a.plot(MieVal_Spline_Angles, MieVal_Spline_Intensity, color='red', linestyle='-', label='Mie Spline Intensity vs. Theta')
pt1, = ax4a.plot(Val_Profiles_to_Angles, Val_Smoothed_Intensity * Ratio_Avg * 0.08, color='purple', linestyle='-', label='Exp Smoothed Intensity Scaled vs. Theta')
pt2, = ax4a.plot(Val_Profiles_to_Angles, Val_Smoothed_Intensity, color='green', linestyle='-', label='Exp Smoothed Intensity vs. Theta')
ax4a.set_xlabel('Angles (\u00B0)', color='red')
ax4a.set_ylabel('Intensity')
ax4a.tick_params(axis='x', labelcolor='red')
ax4a.minorticks_on()
ax4a.grid(True, which='both')
ax4a.legend(loc=1)
ax4a.set_yscale('log')
ax4b = ax4a.twiny()
pt3, = ax4b.plot(Val_PN, Val_Smoothed_Intensity, color='blue', linestyle='-', label='Exp. Smoothed Intensity vs. PN')
ax4b.set_xlabel('Profile Numbers', color='blue')
ax4b.set_ylabel('Intensity')
ax4b.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2, pt3]
ax4a.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, bbox_to_anchor=[1.35, 1], fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax4a.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F4.pdf', format='pdf')
plt.show()

print(Exp_Profiles_to_Angles == Val_Profiles_to_Angles)
