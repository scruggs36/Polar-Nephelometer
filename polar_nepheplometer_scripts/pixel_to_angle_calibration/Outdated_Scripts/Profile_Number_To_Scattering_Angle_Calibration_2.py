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
Rayleigh_Directory = '/home/austen/Documents/SD_avg_15s_N2_Offline.txt'
Save_Directory = '/home/austen/Documents/'
Rayleigh_Data = pd.read_csv(Rayleigh_Directory, delimiter=',', header=0)
Rayleigh_Int = Rayleigh_Data['Intensity']
Rayleigh_PN = np.asarray(Rayleigh_Data['Columns'])

# import data and format into intensities, angles, and profile numbers into separate arrays
Mie_SD_Directory = '/home/austen/Documents/Distribution_Analysis/Theory_Summary2.txt'
Exp_SD_Directory = '/home/austen/Documents/09-28-2018-Data/PSL_900nm/SD_avg_15s.txt'
Fig_Directory = '/home/austen/Documents/' # save figures directory
Mie_SD_Data = pd.read_csv(Mie_SD_Directory, delimiter=',', header=0)
Exp_SD_Data = pd.read_csv(Exp_SD_Directory, delimiter=',', header=0)
Mie_Intensity = np.asarray(Mie_SD_Data['S11'])
Mie_Angles = np.asarray(Mie_SD_Data['Theta'])
Exp_Intensity = np.asarray(Exp_SD_Data['SD Particle'])
# the actual profile number needs to be added into the labview code!
Exp_Profile_Numbers = np.arange(Rayleigh_Data['Columns'][0], Rayleigh_Data['Columns'][0] + len(Exp_SD_Data['Profile Number']), 1)
print(len(Exp_Profile_Numbers), len(Exp_SD_Data['Profile Number']))

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
Exp_X_Vals_OLS = sm.add_constant(np.add(Exp_Local_Features, Rayleigh_PN[0])) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
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
ax1b.plot(np.add(Exp_Local_Features, Rayleigh_PN[0]), Exp_Intensities_at_Features, marker='X', color='black', linestyle='None', label='Local Max & Min') # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
ax1b.set_title('Measured Scattering Diagram\n Vertically Polarized 663nm Radiation')
ax1b.set_xlabel('Profile Number')
ax1b.set_ylabel('Intensity')
ax1b.set_yscale('log')
ax1b.legend(loc=1)
ax1b.grid(True)
ax1c = f1.add_subplot(gs[1, :])
ax1c.plot(np.add(Exp_Local_Features, Rayleigh_PN[0]), Mie_Local_Features, marker='X', color='black', label='Angles vs Profile Numbers')
ax1c.plot(np.add(Exp_Local_Features, Rayleigh_PN[0]), results0.fittedvalues, color='green', linestyle='--', label='OLS: y = ' + str('{:.3f}'.format(results0.params[1])) + 'x + ' + str('{:.3f}'.format(results0.params[0])))
ax1c.set_title('Scattering Angle as a Function of Profile Number')
ax1c.set_xlabel('Profile Number')
ax1c.set_ylabel('Scattering Angle (\u00B0)')
ax1c.legend(loc=2)
ax1c.grid(True)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F1.pdf', format='pdf')
plt.show()
#'''


# Apply correction by adding the delta angle correction to the angle axis of the PSL scattering data

# Increase the resolution of the Mie data such that it is the same number of data points
# as the experimental data
Mie_Spline_Func = interp1d(Mie_Angles, Mie_Intensity, kind='cubic')
Mie_Spline_Angles = np.linspace(0, 180, len(Exp_Profile_Numbers), endpoint=False)
Mie_Spline_Intensity = Mie_Spline_Func(Mie_Spline_Angles)

# First normalize the intensities and plot them one ontop of the other
Mie_Spline_Intensity_Normalized = Mie_Spline_Intensity / np.linalg.norm(Mie_Spline_Intensity)
Exp_Smoothed_Intensity_Normalized = Exp_Smoothed_Intensity / np.linalg.norm(Exp_Smoothed_Intensity)

# convert profile numbers to angles with the OLS from the Manfred appraoach
Exp_Profiles_to_Angles = [results0.params[1] * x + results0.params[0] for x in Exp_Profile_Numbers]
Exp_Profiles_to_Angles = np.array(Exp_Profiles_to_Angles)
print(type(Exp_Profiles_to_Angles), Exp_Profiles_to_Angles.shape)

# parameters from linear calibration conducted like manfred et al.
slope = results0.params[1]
intercept = results0.params[0]

# fit N2 Rayleigh scattering data to a quadratic
PN_To_Angle = Rayleigh_PN * slope + intercept

# These values control the slicing of the N2 Data plotted below
T1 = 50
T2 = 700

# The below fits the N2 data
coeffs = np.polynomial.polynomial.polyfit(Rayleigh_PN[T1:T2], Rayleigh_Int[T1:T2], deg=2)
fit_R_PN_Int = np.polynomial.polynomial.polyval(Rayleigh_PN, coeffs)
fit_Exp_PN_Int = np.polynomial.polynomial.polyval(Exp_Profile_Numbers, coeffs)
fit_R_PN_Int_Min = fit_R_PN_Int[np.argmin(fit_R_PN_Int)]
fit_Exp_PN_Int_Min = fit_Exp_PN_Int[np.argmin(fit_Exp_PN_Int)]


# fit Rayleigh scattering, the Delta Angle vs. Angle Function, note that Delta Angle 2 is the lens correction applied after the linear calibration

I_v_Imin = (fit_R_PN_Int/fit_R_PN_Int_Min)# * slope - slope
Change_In_Theta = ((fit_R_PN_Int/fit_R_PN_Int_Min) * slope)
Change_In_Theta2 = ((fit_Exp_PN_Int/fit_Exp_PN_Int_Min) * slope)
PN_Sign_Flip = Rayleigh_PN[np.argmin(fit_R_PN_Int)]
PN_Grouping = []
for counter, element in enumerate(Rayleigh_PN):
    if element > PN_Sign_Flip:
        PN_Grouping.append(I_v_Imin[counter] * -1)
    else:
        PN_Grouping.append(I_v_Imin[counter])

# convert from list to numpy array
PN_Grouping = np.array(PN_Grouping)

# Theta grouping
Theta_Changing = []
for counter, element in enumerate(Rayleigh_PN):
    if element > PN_Sign_Flip:
        Theta_Changing.append(Change_In_Theta[counter] * -1)
    else:
        Theta_Changing.append(Change_In_Theta[counter])

Theta_Changing2 = []
for counter, element in enumerate(Exp_Profile_Numbers):
    if element > PN_Sign_Flip:
        Theta_Changing2.append(Change_In_Theta2[counter] * -1)
    else:
        Theta_Changing2.append(Change_In_Theta2[counter])

# convert from list to numpy array
Theta_Changing = np.array(Theta_Changing)

# convert from list to numpy array
Theta_Changing2 = np.array(Theta_Changing2)
# Make pretty plots
f2, ax2 = plt.subplots(2, 2, figsize=(12, 6))
# Nitrogen Data, slices, and fits
ax2[0, 0].plot(Rayleigh_PN, Rayleigh_Int, color='blue', linestyle='-', label='N2 15s Exposure Full')
ax2[0, 0].plot(Rayleigh_PN[T1:T2], Rayleigh_Int[T1:T2], color='orange', linestyle='-', label='N2 15s Exposure Truncated')
ax2[0, 0].plot(Rayleigh_PN, fit_R_PN_Int, color='red', linestyle='--', label='Polynomial Fit')
ax2[0, 0].set_xlabel('Profile Number (PN)')
ax2[0, 0].set_ylabel('Intensity')
ax2[0, 0].set_title('Nitrogen Rayleigh Scattering Data')
ax2[0, 0].legend(loc=1)
ax2[0, 0].grid(True)
# The fit throughout the entire ROI of the images, this plot shows how the vertical transects get grouped together across the ROI of the CCD
# This plot is supposed to show how the intensity changes as a function of the vertical profile grouping
ax2[0, 1].plot(Rayleigh_PN, I_v_Imin, color='purple', linestyle='-', label='\u0394\u00B0/\u0394PN')
ax2[0, 1].set_xlabel('Profile Number (PN)')
ax2[0, 1].set_ylabel('$I/I_{min}$')
ax2[0, 1].set_title('The Symmetric Grouping of Profiles Across The ROI: \n \u0394\PN as a function of PN')
ax2[0, 1].grid(True)
ax2[0, 1].legend(loc=1)
# This plot is supposed to show that the profile grouping goes in the opposite directions from the middle, although we forced this plot
ax2[1, 0].plot(Rayleigh_PN, PN_Grouping, color='green', linestyle='-', label='\u0394\u00B0/\u0394PN')
ax2[1, 0].set_xlabel('Profile Number (PN)')
ax2[1, 0].set_ylabel('\u0394PN')
ax2[1, 0].set_title('\u0394PN as a function of Profile Number')
ax2[1, 0].grid(True)
ax2[1, 0].legend(loc=1)
# Converting to Angle
#ax2[1, 1].plot(Rayleigh_PN, Theta_Changing + (slope * Rayleigh_PN) + intercept, color='orange', linestyle='-', label='\u0394\u0398 vs. PN')
ax2[1, 1].plot(Rayleigh_PN, Theta_Changing, color='black', linestyle='-', label='\u0394\u0398 vs. PN')
ax2[1, 1].set_xlabel('Profile Number (PN)')
ax2[1, 1].set_ylabel('\u0394\u0398')
ax2[1, 1].set_title('\u0394\u0398 as a function of Profile Number')
ax2[1, 1].grid(True)
ax2[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Save_Directory + 'F2.pdf', format='pdf')
plt.show()


# then apply the lens correction, lens correction 1 uses the quadratic whose data was inverted after the minimum, lens
# correction 2 uses the fit of qudadratic data with half its data inverted, we may have to add the profile number recording into the Labview program

Lens_Corrected_Angles = []
for counter, element in enumerate(Theta_Changing2):
    val = (slope * Exp_Profile_Numbers[counter]) + intercept + element
    Lens_Corrected_Angles.append(val)
    print(val)

Lens_Corrected_Angles = np.array(Lens_Corrected_Angles)



# make a plot of the spline data and the experimental data overlayed
f3, ax3a = plt.subplots(figsize=(12, 6))
pt0, = ax3a.plot(Mie_Spline_Angles, Mie_Spline_Intensity, color='red', linestyle='-', label='Mie Spline Intensity')
pt1, = ax3a.plot(Exp_Profiles_to_Angles, Exp_Smoothed_Intensity, color='green', linestyle='-', label='Exp Smoothed Intensity \n Independent Variable: Theta')
#pt2, = ax3a.plot(Exp_Profiles_to_Angles_LC1_Angles, Exp_Smoothed_Intensity, color='purple', linestyle='-', label='Exp Smoothed Intensity \n Independent Variable: Theta \n Lens Corrected Mod. Quad.')
pt3, = ax3a.plot(Lens_Corrected_Angles, Exp_Smoothed_Intensity, color='purple', linestyle='-', label='Exp Smoothed Intensity \n Independent Variable: Theta \n Lens Corrected Fit')
ax3a.set_xlabel('Angles (\u00B0)', color='red')
ax3a.set_ylabel('Intensity')
ax3a.tick_params(axis='x', labelcolor='red')
ax3a.minorticks_on()
ax3a.grid(True, which='both')
ax3a.legend(loc=1)
ax3a.set_yscale('log')
ax3b = ax3a.twiny()
pt4, = ax3b.plot(Exp_Profile_Numbers, Exp_Smoothed_Intensity, color='blue', linestyle='-', label='Exp. Smoothed Intensity \n Independent Variable: Profile Numbers')
ax3b.set_xlabel('Profile Numbers', color='blue')
ax3b.set_ylabel('Intensity')
ax3b.tick_params(axis='x', labelcolor='blue')
pt = [pt0, pt1, pt3, pt4]
ax3a.legend(pt, [pt_.get_label() for pt_ in pt], loc=1, bbox_to_anchor=[1.35, 1], fontsize='small')
# this little bit set_major_locator(plt.MaxNLocator(10)) sets the x axis minor ticks to 5 degree increments
ax3a.xaxis.set_major_locator(plt.MaxNLocator(10))
#f2.suptitle('Overlayed Mie Intensity Normalized Spline Fit Scattering Diagram\n and Experiment Normalized and Smoothed Scattering Diagram', y=1.03)
plt.tight_layout()
plt.savefig(Fig_Directory + 'F3.pdf', format='pdf')
plt.show()





