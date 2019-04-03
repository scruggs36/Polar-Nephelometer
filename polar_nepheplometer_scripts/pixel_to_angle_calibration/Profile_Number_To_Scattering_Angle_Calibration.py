'''
Austen K. Sruggs
date created:  08-23-2018
Description: script solves for local maxima and minima in
Mie theory and measured scattering diagrams and finds their
indicesThen plots the Mie theory angle of the local maxima
and minima as a function of the profile number
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

# import data and format into intensities, angles, and profile numbers into separate arrays
Mie_SD_Directory = '/home/austen/Documents/Distribution_Analysis/Theory_Summary2.txt'
Exp_SD_Directory = '/home/austen/Documents/09-28-2018-Data/PSL_900nm/SD_avg_5s.txt'
Fig_Directory = '/home/austen/Documents/' # save figures directory
Mie_SD_Data = pd.read_csv(Mie_SD_Directory, delimiter=',', header=0)
Exp_SD_Data = pd.read_csv(Exp_SD_Directory, delimiter=',', header=0)
Mie_Intensity = np.asarray(Mie_SD_Data['S11'])
Mie_Angles = np.asarray(Mie_SD_Data['Theta'])
Exp_Intensity = np.asarray(Exp_SD_Data['SD Particle'])
Exp_Profile_Numbers = np.asarray(Exp_SD_Data['Profile Number'])

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
ax0[1].plot(Exp_Profile_Numbers, Exp_Intensity, 'b-', label='900nm PSL')
ax0[1].plot(Exp_Profile_Numbers, Exp_Smoothed_Intensity, 'y-', label='900nm PSL Smoothed')
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
print('Mie local features: ', Mie_Local_Features)

# find all local maxima and minima in measured scattering diagram
Exp_Max = np.argmax(Exp_Smoothed_Intensity)
print('Exp maximum: ', Exp_Max)
Exp_Local_Max = np.asarray(argrelmax(Exp_Smoothed_Intensity, axis=0)).flatten()
print('Exp local max indices: ', Exp_Local_Max)
Exp_Local_Min = np.asarray(argrelmin(Exp_Smoothed_Intensity, axis=0)).flatten()
print('Exp local min indices: ', Exp_Local_Min)
Exp_Local_Features = sorted(list(set(np.concatenate((Exp_Max, Exp_Local_Max, Exp_Local_Min), axis=None).ravel().tolist())))
drop = [0]
for index in sorted(drop, reverse=True):
    del Exp_Local_Features[index]
print('Exp local features: ', Exp_Local_Features)

# pull intensities from local max and minima, create arrays
Mie_Intensities_at_Features = [Mie_Intensity[element] for element in Mie_Local_Features]
Exp_Intensities_at_Features = [Exp_Smoothed_Intensity[element] for element in Exp_Local_Features]

# do a linear fit to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
Exp_X_Vals_OLS = sm.add_constant(Exp_Local_Features)
model0 = sm.OLS(Mie_Local_Features, Exp_X_Vals_OLS)
results0 = model0.fit()
print(results0.summary())

#'''
# plot imported data
f1 = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2)
ax1a = f1.add_subplot(gs[0, 0])
ax1a.plot(Mie_Angles, Mie_Intensity, 'r-', label='900nm PSL')
ax1a.plot(Mie_Local_Features, Mie_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min')
ax1a.set_title('Mie Theory Calculated Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1a.set_xlabel('Angles (\u00B0)')
ax1a.set_ylabel('Intensity')
ax1a.legend(loc=1)
ax1a.set_yscale('log')
ax1a.grid(True)
ax1b = f1.add_subplot(gs[0, 1])
ax1b.plot(Exp_Profile_Numbers, Exp_Intensity, 'b-', label='900nm PSL')
ax1b.plot(Exp_Profile_Numbers, Exp_Smoothed_Intensity, 'y-', label='900nm PSL Smoothed')
ax1b.plot(Exp_Local_Features, Exp_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min')
ax1b.set_title('Measured Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1b.set_xlabel('Profile Number')
ax1b.set_ylabel('Intensity')
ax1b.set_yscale('log')
ax1b.legend(loc=1)
ax1b.grid(True)
ax1c = f1.add_subplot(gs[1, :])
ax1c.plot(Exp_Local_Features, Mie_Local_Features, marker='X', color='black', label='Angles vs Profile Numbers')
ax1c.plot(Exp_Local_Features, results0.fittedvalues, color='green', linestyle='--', label='OLS: y = ' + str('{:.3f}'.format(results0.params[1])) + 'x + ' + str('{:.3f}'.format(results0.params[0])))
ax1c.set_title('Scattering Angle as a Function of Profile Number')
ax1c.set_xlabel('Profile Number')
ax1c.set_ylabel('Scattering Angle (\u00B0)')
ax1c.legend(loc=2)
ax1c.grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F1.pdf', format='pdf')
plt.show()
#'''

'''
now, we want to fit the experiment SD intensities to the Mie inensities,
we will take the same approach as the Manfred et al. 2018 paper and use the intensity
values at the maxima and minima and find a good fit
'''
Mie_Local_Max_Intensities = Mie_Intensity[Mie_Local_Max]
Exp_Local_Max2 = np.delete(Exp_Local_Max, [0, 3, 5])
Exp_Local_Max3 = [results0.params[1] * x + results0.params[0] for x in Exp_Local_Max2]
Exp_Local_Max_Intensities = [Exp_Smoothed_Intensity[element] for element in Exp_Local_Max2]
print('EXP MAX PROFILE NUMBERS: ', Exp_Local_Max2)
print('EXP MAX THETA: ', Exp_Local_Max3)
print('EXP MAX INTENSITIES: ', Exp_Local_Max_Intensities)
print('MIE LOCAL MAX INTENSITIES: ', Mie_Local_Max_Intensities)
Mie_to_Exp_Intensity_Ratios = np.true_divide(Mie_Local_Max_Intensities, Exp_Local_Max_Intensities)
print('MIE/EXP INTENSITY RATIOS: ', Mie_to_Exp_Intensity_Ratios)

Mie_X_Vals_OLS = sm.add_constant(Mie_Local_Max)
model1 = sm.OLS(Mie_to_Exp_Intensity_Ratios, Mie_X_Vals_OLS)
results1 = model1.fit()
print(results1.summary())

f3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(Mie_Local_Max, np.true_divide(Mie_Local_Max_Intensities, Exp_Local_Max_Intensities), color='black', marker='o', label='Data')
ax3.plot(Mie_Local_Max, results1.fittedvalues, color='green', linestyle='--', label='OLS: y = ' + str('{:.3e}'.format(results1.params[1])) + 'x + ' + str('{:.3e}'.format(results1.params[0])))
ax3.set_title('Mie Scattering Intensity : Experiment Scattering Intensity Ratio\nas a Function of Scattering Angle')
ax3.set_xlabel('Scattering Angle (\u00b0)')
ax3.set_ylabel('Intensity Ratio (Mie/Experimental)')
ax3.grid(True)
ax3.legend(loc=2)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F3.pdf', format='pdf')
plt.show()

'''
Now lets check the ratio between local maxima in the Mie theory, and then again in the experiment,
then compare these ratios and see if they match up!
'''
Exp_Int_A = Exp_Local_Max_Intensities[0]
Exp_Int_B = Exp_Local_Max_Intensities[1]
Exp_Int_C = Exp_Local_Max_Intensities[2]
Mie_Int_A = Mie_Local_Max_Intensities[0]
Mie_Int_B = Mie_Local_Max_Intensities[1]
Mie_Int_C = Mie_Local_Max_Intensities[2]

# order A/B, A/C, B/C
Mie_Int_Ratio_Array = [Mie_Int_A/Mie_Int_B, Mie_Int_A/Mie_Int_C, Mie_Int_B/Mie_Int_C]
#print(Mie_Int_Ratio_Array)
Exp_Int_Ratio_Array = [Exp_Int_A/Exp_Int_B, Exp_Int_A/Exp_Int_C, Exp_Int_B/Exp_Int_C]
#print(Exp_Int_Ratio_Array)
Inter_Comparison_Ratio_Array = [Mie_Int_Ratio_Array[0]/Exp_Int_Ratio_Array[0], Mie_Int_Ratio_Array[1]/Exp_Int_Ratio_Array[1], Mie_Int_Ratio_Array[2]/Exp_Int_Ratio_Array[2]]
#print(Inter_Comparison_Ratio_Array)
Inter_Comparison_Maxes_Array = [Mie_Int_A/Exp_Int_A, Mie_Int_B/Exp_Int_B, Mie_Int_C/Exp_Int_C]
print(Inter_Comparison_Maxes_Array)


'''
Now we are going to try to write a intensity matching script
to use all the data
'''

# Increase the resolution of the Mie data such that it is the same number of data points
# as the experimental data
Mie_Spline_Func = interp1d(Mie_Angles, Mie_Intensity, kind='cubic')
Mie_Spline_Angles = np.linspace(0, 180, len(Exp_Profile_Numbers), endpoint=False)
Mie_Spline_Intensity = Mie_Spline_Func(Mie_Spline_Angles)

# First normalize the intensities and plot them one ontop of the other
Mie_Spline_Intensity_Normalized = Mie_Spline_Intensity / np.linalg.norm(Mie_Spline_Intensity)
Exp_Smoothed_Intensity_Normalized = Exp_Smoothed_Intensity / np.linalg.norm(Exp_Smoothed_Intensity)
Exp_Profiles_to_Angles = [results0.params[1] * x + results0.params[0] for x in Exp_Profile_Numbers]
#print(Exp_Profiles_to_Angles)
# make a plot of the spline data and the experimental data overlayed
f2, ax2a = plt.subplots(figsize=(12, 6))
pt0, = ax2a.plot(Mie_Spline_Angles, Mie_Spline_Intensity, color='red', linestyle='-', label='Mie Spline Intensity')
pt1, = ax2a.plot(Exp_Profiles_to_Angles, Exp_Smoothed_Intensity, color='green', linestyle='-', label='Exp Smoothed Intensity \n Independent Variable: Theta')
ax2a.set_xlabel('Angles (\u00B0)', color='red')
ax2a.set_ylabel('Intensity')
ax2a.tick_params(axis='x', labelcolor='red')
ax2a.minorticks_on()
ax2a.grid(True, which='both')
ax2a.legend(loc=1)
ax2a.set_yscale('log')
ax2b = ax2a.twiny()
pt2, = ax2b.plot(Exp_Profile_Numbers, Exp_Smoothed_Intensity, color='blue', linestyle='-', label='Exp. Smoothed Intensity \n Independent Variable: Profile Numbers')
ax2b.set_xlabel('Profile Numbers', color='blue')
ax2b.set_ylabel('Intensity')
ax2b.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt2]
ax2a.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, bbox_to_anchor=[1.35, 1], fontsize='small')
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F2.pdf', format='pdf')
plt.show()

# okay, we are going to refind the local max and mins so we can correct the profile number to angle calibration
# find all local maxima and minima in Mie scattering diagram
Mie_Max2 = np.argmax(Mie_Spline_Intensity_Normalized)
print('Mie maximum: ', Mie_Max2)
Mie_Local_Max2 = np.asarray(argrelmax(Mie_Spline_Intensity_Normalized, axis=0)).flatten()
print('Mie local max indices: ', Mie_Local_Max2)
Mie_Local_Min2 = np.asarray(argrelmin(Mie_Spline_Intensity_Normalized, axis=0)).flatten()
print('Mie local min indices: ', Mie_Local_Min2)
Mie_Local_Features2 = sorted(list(set(np.concatenate((Mie_Max2, Mie_Local_Max2, Mie_Local_Min2), axis=None).ravel().tolist())))
del Mie_Local_Features2[0]
print('Mie local features: ', Mie_Local_Features2)

# find all local maxima and minima in measured scattering diagram
Exp_Max2 = np.argmax(Exp_Smoothed_Intensity_Normalized)
print('Exp maximum: ', Exp_Max2)
Exp_Local_Max2 = np.asarray(argrelmax(Exp_Smoothed_Intensity_Normalized, axis=0)).flatten()
print('Exp local max indices: ', Exp_Local_Max2)
Exp_Local_Min2 = np.asarray(argrelmin(Exp_Smoothed_Intensity_Normalized, axis=0)).flatten()
print('Exp local min indices: ', Exp_Local_Min2)
Exp_Local_Features2 = sorted(list(set(np.concatenate((Exp_Max2, Exp_Local_Max2, Exp_Local_Min2), axis=None).ravel().tolist())))
drop2 = [5, 6, 9, 10]
for index in sorted(drop2, reverse=True):
    del Exp_Local_Features2[index]
del Exp_Local_Features2[0]
print('Exp local features: ', Exp_Local_Features2)

# Now we need to find the delta angle for each feature!
Mie_Local_Features_Angles = [Mie_Spline_Angles[element] for element in Mie_Local_Features2]
Exp_Local_Features_Angles = [Exp_Profiles_to_Angles[element] for element in Exp_Local_Features2]
Delta_Angles = np.asarray(Mie_Local_Features_Angles) - np.asarray(Exp_Local_Features_Angles)

f6, ax6 = plt.subplots(figsize=(6, 6))
ax6.plot(Exp_Local_Features2, Delta_Angles, marker='.', linestyle=' ', color='black', label='\u0394 \u00b0 vs. Profile Number')
ax6.set_title('Change in the Calibrated Scattering Angle Needed In Order \n to Match Mie Theory as a Function of Profile Number ')
ax6.set_xlabel('Profile Number')
ax6.set_ylabel('\u0394 \u00b0')
ax6.grid(True)
plt.legend(loc=2)
plt.savefig(Fig_Directory + 'F6.pdf', format='pdf')
plt.show()
'''
Now we are going to attempt to make a correlation scatter plot and
draw a linear regression through the data
'''
Mie_Spline_Intensity_Edited = []
Exp_Smoothed_Intensity_Edited = []

# had to implement these for loops to make sure the data is the same size afer rejecting some points
for element in Mie_Spline_Intensity:
    if element > 50:
        Mie_Spline_Intensity_Edited.append(element)

for element in Exp_Smoothed_Intensity:
    if element > 639.2:
       Exp_Smoothed_Intensity_Edited.append(element)

print(len(Mie_Spline_Intensity_Edited))
print(len(Exp_Smoothed_Intensity_Edited))

# now to perform another ordinary least squares regression
b = 136
a = 106

Mie_Spline_X_Vals_OLS2 = sm.add_constant(Mie_Spline_Intensity_Edited[a:b])
model2 = sm.OLS(Exp_Smoothed_Intensity_Edited[a:b], Mie_Spline_X_Vals_OLS2)
results2 = model2.fit()
print(results2.summary())

Mie_Spline_X_Vals_OLS3 = sm.add_constant(Mie_Spline_Intensity_Edited)
model3 = sm.OLS(Exp_Smoothed_Intensity_Edited, Mie_Spline_X_Vals_OLS3)
results3 = model3.fit()
print(results3.summary())


f4, ax4 = plt.subplots(1, 2, figsize=(10, 6))
ax4[0].scatter(Mie_Spline_Intensity, Exp_Smoothed_Intensity, marker='.', color='black', label='Measured vs. Mie')
ax4[0].set_xlabel('Mie Intensity')
ax4[0].set_ylabel('Measured Intensity')
ax4[0].set_title('Measured Intensity vs Mie Intensity')
ax4[0].legend(loc=2)
ax4[0].grid(True)
ax4[1].scatter(Mie_Spline_Intensity_Edited[a:b], Exp_Smoothed_Intensity_Edited[a:b], marker='.', color='black', label='Measured vs. Mie')
ax4[1].plot(Mie_Spline_Intensity_Edited[a:b], [results2.params[1] * element + results2.params[0] for element in (Mie_Spline_Intensity_Edited[a:b])], color='green', linestyle='--', label='OLS: y = ' + str('{:.3f}'.format(results2.params[1])) + 'x + ' + str('{:.3f}'.format(results2.params[0])))
ax4[1].plot(Mie_Spline_Intensity_Edited[a:b], Mie_Spline_Intensity_Edited[a:b], linestyle='--', color='red', label='1:1')
ax4[1].set_xlabel('Mie Intensity')
ax4[1].set_ylabel('Measured Intensity')
ax4[1].set_title('Measured Intensity vs Mie Intensity\nLowest Intensities Removed')
ax4[1].legend(loc=2)
ax4[1].grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F4.pdf', format='pdf')
plt.show()

# make a plot of the spline data and the experimental data overlayed
f5, ax5a = plt.subplots(figsize=(12, 6))
p0, = ax5a.plot(Mie_Spline_Angles, Mie_Spline_Intensity, color='red', linestyle='-', label='Mie Spline Intensity')
p1, = ax5a.plot(Exp_Profiles_to_Angles, Exp_Smoothed_Intensity, color='green', linestyle='-', label='Exp Smoothed Intensity Unscaled \n Independent Variable: Theta')
#ax5a.plot(Exp_Profiles_to_Angles, (Exp_Smoothed_Intensity_Normalized * Inter_Comparison_Maxes_Array[0]), color='green', linestyle='-', label='Exp Smoothed Intensity Scaled & Normalized\nIndependent Variable: Theta (Mie/Exp Ratio #1 & 2)')
#ax5a.plot(Exp_Profiles_to_Angles, (Exp_Smoothed_Intensity_Normalized * Inter_Comparison_Maxes_Array[2]), color='purple', linestyle='-', label='Exp Smoothed Intensity Scaled & Normalized\nIndependent Variable: Theta (Mie/Exp Ratio #3)')
p2, = ax5a.plot(Exp_Profiles_to_Angles, ((Exp_Smoothed_Intensity) / results3.params[1]), color='purple', linestyle='-', label='Exp Smoothed Intensity \n Independent Variable: Theta \n (linear fit Exp vs. Mie)')
p3, = ax5a.plot(Exp_Profiles_to_Angles, ((Exp_Smoothed_Intensity) / results2.params[1]), color='orange', linestyle='-', label='Exp Smoothed Intensity \n Independent Variable: Theta \n (linear fit Exp vs. Mie Weight Low Vals.)')
ax5a.set_xlabel('Angles (\u00B0)', color='red')
ax5a.set_ylabel('Intensity')
ax5a.tick_params(axis='x', labelcolor='red')
ax5a.minorticks_on()
ax5a.grid(True, which='both')
ax5a.set_yscale('log')
ax5b = ax5a.twiny()
p4, = ax5b.plot(Exp_Profile_Numbers, Exp_Smoothed_Intensity, color='blue', linestyle='-', label='Exp. Smoothed Intensity \n Independent Variable: Profile Numbers')
ax5b.set_xlabel('Profile Numbers', color='blue')
ax5b.set_ylabel('Intensity')
ax5b.tick_params(axis='x', labelcolor='blue')
p = [p0, p1, p2, p3, p4]
ax5a.legend(p, [p_.get_label() for p_ in p], loc=1, bbox_to_anchor=[1.35, 1], fontsize='small')
plt.tight_layout()
plt.savefig(Fig_Directory + 'F5.pdf', format='pdf')
plt.show()