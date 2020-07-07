'''
Austen K. Scruggs
06/10/2020
Description: Calculating the optical transfer function to correct measured Rayleigh scattering
phase functions
'''

import numpy as np
import pandas as pd
import PyMieScatt as PMS
import matplotlib.pyplot as plt
from matplotlib import ticker
from math import pi, cos
from scipy.interpolate import pchip_interpolate
from scipy.optimize import curve_fit

save_directory = '/home/sm3/Desktop/Recent/'

def optical_transfer_function(x, a, b, c, d):
    # gaussian doesnt fit well
    #return a * np.exp(- (x - b) ** 2 / (2 * c ** 2)) + d
    # cosine squared doesn't fit well
    #return (a * np.cos(x * (pi/180.0))) + b
    # lorentzian
    return a * ((b ** 2) / (b ** 2 + (x - c) ** 2)) + d


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
    #print('coefficients: ', [isotropic_coeff, anisotropic_coeff])
    return sigma_angle_anisotropic, sigma_angle_isotropic


# other constants
wav = 663
theta_array = np.arange(0, 180.5, 0.5)
chamber_length = 182.88 #cm
chamber_volume = pi * (0.5/2)**2 * 182.88
N = 2.54743E19
# wavelength for some reason on penndorf given in microns, it has to be converted to cm in the calculation
wav_micron = .550
wav_centimeters = wav_micron / 10000.0
wav_meters = wav_centimeters * 10 **-2
frequency = 1 / wav_meters
wavenumbers = (((6.626E-34 * 3.0E8)/(wav_meters))/1.62E-19) * 8065.54
print('wavenumbers: ', wavenumbers)



# CO2 parameters, bondlength in nm
m_CO2 = 1.00044697
bondlength_CO2 = .1163
d_CO2 = 2 * bondlength_CO2
# carbon dioxide optical parameters
co2_kings = ((1.1364 + 25.3) * 10**-12) * frequency**2
ns_co2 = (1.1427E6 * ((5799.25/(128908.9**2 - frequency**2)) + (120.05/(89223.8**2 - frequency**2)) + (5.3334/(75037.5**2 - frequency**2)) + (4.3244/(67837.7**2 - frequency**2)) + (0.1218145E-4/(2418.136 - frequency**2))) + 1)
pn_co2 = .0805
#print('n co2: ', ns_co2)



# N2 parameters, bondlength in nm
m_N2 = 1.00029739
bondlength_N2 = .10976
d_N2 = bondlength_N2
ns_n2 = 1.00029839**2
ns_minus_1_squared_n2 = (1.00029839**2 - 1)**2
#print(ns_minus_1_squared_n2)
pn_n2 = .0305
# D. Spelsberg et al 1994
a_N2 = 11.74



# Air parameters
m_Air = 1.00027618
bondlength_O2 = .121
d_Air = (.78 * d_N2) + (.22 * bondlength_O2) / 2
ns_air = 1.00027773
ns_minus_1_squared_air = (1.00027773**2 - 1)**2
#print(ns_minus_1_squared_air)
pn_air = .035
R_dist = 1

# called functions
CO2_xsections_p = np.array([Angular_Scattering_Cross_Section(theta=i, wav=wav_centimeters, ns=ns_co2, Ns=N, pn=pn_co2)[0] for i in theta_array])
N2_xsections_p = np.array([Angular_Scattering_Cross_Section(theta=i, wav=wav_centimeters, ns=ns_n2, Ns=N, pn=pn_n2)[0] for i in theta_array])
Air_xsections_p = np.array([Angular_Scattering_Cross_Section(theta=i, wav=wav_centimeters, ns=ns_air, Ns=N, pn=pn_air)[0] for i in theta_array])
CO2_xsections_s = np.repeat(CO2_xsections_p[0], len(CO2_xsections_p))
N2_xsections_s = np.repeat(N2_xsections_p[0], len(N2_xsections_p))
Air_xsections_s = np.repeat(Air_xsections_p[0], len(Air_xsections_p))
CO2_xsections_u  = 0.5 * np.add(CO2_xsections_p, CO2_xsections_s)
N2_xsections_u  = 0.5 * np.add(N2_xsections_p, N2_xsections_s)
Air_xsections_u  = 0.5 * np.add(Air_xsections_p, Air_xsections_s)

# make figure
f0, ax0 = plt.subplots(1, 3, figsize=(24, 6))
ax0[0].plot(theta_array, N2_xsections_p, color='red', ls='-', label='N2 $dC_{sca}/d(\u03B8)$' + ' \u2225')
ax0[0].plot(theta_array, N2_xsections_s, color='green', ls='-', label='N2 $dC_{sca}/d(\u03B8)$' + ' ⊥')
ax0[0].plot(theta_array, N2_xsections_u, color='blue', ls='-', label='N2 $dC_{sca}/d(\u03B8)$' +' \u27f3')
ax0[0].set_xlabel('\u03B8')
ax0[0].set_ylabel('$dC_{sca}/d(\u03B8)$')
ax0[0].set_title('$N_2$ Differential Scattering Cross Section\n as a Function of Scattering Angle')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].plot(theta_array, CO2_xsections_p, color='red', ls='-', label='CO2 $dC_{sca}/d(\u03B8)$' + ' \u2225')
ax0[1].plot(theta_array, CO2_xsections_s, color='green', ls='-', label='CO2 $dC_{sca}/d(\u03B8)$' + ' ⊥')
ax0[1].plot(theta_array, CO2_xsections_u, color='blue', ls='-', label='CO2 $dC_{sca}/d(\u03B8)$' + ' \u27f3')
ax0[1].set_xlabel('\u03B8')
ax0[1].set_ylabel('$dC_{sca}/d(\u03B8)$')
ax0[1].set_title('$CO_2$ Differential Scattering Cross Section\n as a Function of Scattering Angle')
ax0[1].grid(True)
ax0[1].legend(loc=1)
ax0[2].plot(theta_array, Air_xsections_p, color='red', ls='-', label='Air $dC_{sca}/d(\u03B8)$' + ' \u2225')
ax0[2].plot(theta_array, Air_xsections_s, color='green', ls='-', label='Air $dC_{sca}/d(\u03B8)$' + ' ⊥')
ax0[2].plot(theta_array, Air_xsections_u, color='blue', ls='-', label='Air $dC_{sca}/d(\u03B8)$' + ' \u27f3')
ax0[2].set_xlabel('\u03B8')
ax0[2].set_ylabel('$dC_{sca}/d(\u03B8)$')
ax0[2].set_title('Air Differential Scattering Cross Section\n as a Function of Scattering Angle')
ax0[2].grid(True)
ax0[2].legend(loc=1)
plt.savefig(save_directory + 'Rayleigh_Theoretical_Diff_Xsection_Scattering.png', format='png')
plt.savefig(save_directory + 'Rayleigh_Theoretical_Diff_Xsection_Scattering.pdf', format='pdf')
plt.tight_layout()
plt.show()

# using CO2 measurements find the optical transfer functions G_p, G_s, and G_u G = I_theory / I measured
directory_CO2_p= '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0.5R/SD_Rayleigh_LCVR@0.5R+QWP@F0.txt'
directory_CO2_s= '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0R/SD_Rayleigh_RR_Exact_Dolgos_QWP_S0.txt'
directory_CO2_u= '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0.25R/SD_Rayleigh_LCVR@0.25R+QWP@F0.txt'
data_CO2_p = pd.read_csv(directory_CO2_p, sep=',', header=0)
data_CO2_s = pd.read_csv(directory_CO2_s, sep=',', header=0)
data_CO2_u = pd.read_csv(directory_CO2_u, sep=',', header=0)
CO2_p_pf = np.array(data_CO2_p['CO2 Intensity gfit'])
CO2_s_pf = np.array(data_CO2_s['CO2 Intensity gfit'])
CO2_u_pf = np.array(data_CO2_u['CO2 Intensity gfit'])
theta_CO2_p = np.array(data_CO2_p['CO2 Theta'])
theta_CO2_s = np.array(data_CO2_s['CO2 Theta'])
theta_CO2_u = np.array(data_CO2_u['CO2 Theta'])
theta_CO2_p = np.array(theta_CO2_p[~np.isnan(CO2_p_pf)])
theta_CO2_s = np.array(theta_CO2_s[~np.isnan(CO2_s_pf)])
theta_CO2_u = np.array(theta_CO2_u[~np.isnan(CO2_u_pf)])
# raw measurement
CO2_p_pf = np.array(CO2_p_pf[~np.isnan(CO2_p_pf)])
CO2_s_pf = np.array(CO2_s_pf[~np.isnan(CO2_s_pf)])
CO2_u_pf = np.array(CO2_u_pf[~np.isnan(CO2_u_pf)])
# take natural logarithm of measurement
#CO2_p_pf = np.log(np.array(CO2_p_pf[~np.isnan(CO2_p_pf)]))
#CO2_s_pf = np.log(np.array(CO2_s_pf[~np.isnan(CO2_s_pf)]))
#CO2_u_pf = np.log(np.array(CO2_u_pf[~np.isnan(CO2_u_pf)]))


# pchip differential cross-section data
CO2_p_diffxsca = np.array(pchip_interpolate(xi=theta_array, yi=np.array(CO2_xsections_p), x=theta_CO2_p))
CO2_s_diffxsca = np.array(pchip_interpolate(xi=theta_array, yi=np.array(CO2_xsections_s), x=theta_CO2_s))
CO2_u_diffxsca = np.array(pchip_interpolate(xi=theta_array, yi=np.array(CO2_xsections_u), x=theta_CO2_u))
CO2_p_diffbsca = CO2_p_diffxsca * N
CO2_s_diffbsca = CO2_s_diffxsca * N
CO2_u_diffbsca = CO2_u_diffxsca * N

# optical transfer function this is in units of Mm^-1/DN, if we are right on the differential scattering cross section for CO2, the phase function y axis should be in Mm^-1 as well!
G_p = np.divide(CO2_p_diffbsca, CO2_p_pf)
G_s = np.divide(CO2_s_diffbsca, CO2_s_pf)
G_u = np.divide(CO2_u_diffbsca, CO2_u_pf)

# fit optical transfer function parallel (p)
popt_G_p, pcov_G_p = curve_fit(optical_transfer_function, theta_CO2_p, G_p, p0=[1E-9, 1, 90, .2E-9])
# fit optical transfer function for perp. (s)
#popt_G_s, pcov_G_s = curve_fit(optical_transfer_function, theta_CO2_s, G_s)

# poly fit for p polarization: extremely high order polynomial for it to fit...
#G_p_coeffs = np.polyfit(theta_CO2_p, G_p, deg=18)
#G_p_fit = np.poly1d(G_p_coeffs)

# fit optical transfer function for perpendicular (S), poly fit for s polarization: fits a 4th order polynomial pretty well
G_s_coeffs = np.polyfit(theta_CO2_s, G_s, deg=4)
G_s_fit = np.poly1d(G_s_coeffs)

# fit optical transfer function for circular
popt_G_u, pcov_G_u = curve_fit(optical_transfer_function, theta_CO2_u, G_u, p0=[1E-9, 1, 90, .2E-9])

# correcting measured data
CO2_p_corrected = optical_transfer_function(theta_CO2_p, *popt_G_p) * CO2_p_pf
CO2_s_corrected = G_s_fit(theta_CO2_s) * CO2_s_pf
CO2_u_corrected = optical_transfer_function(theta_CO2_u, *popt_G_u) * CO2_u_pf
what_if_1 = G_s_fit(theta_CO2_p) * CO2_p_pf
what_if_2 = optical_transfer_function(theta_CO2_s, *popt_G_p) * CO2_s_pf

# Residuals
R_p = ((CO2_p_pf/np.sum(CO2_p_pf)) - (CO2_p_diffbsca/np.sum(CO2_p_diffbsca)))
R_s = ((CO2_s_pf/np.sum(CO2_s_pf)) - (CO2_s_diffbsca/np.sum(CO2_s_diffbsca)))
R_u = ((CO2_u_pf/np.sum(CO2_u_pf)) - (CO2_u_diffbsca/np.sum(CO2_u_diffbsca)))
percent_R_p = (R_p / (CO2_p_diffbsca/np.sum(CO2_p_diffbsca))) * 100
percent_R_s = (R_s / (CO2_s_diffbsca/np.sum(CO2_s_diffbsca))) * 100
percent_R_u = (R_u / (CO2_u_diffbsca/np.sum(CO2_u_diffbsca))) * 100
# create figure
f1, ax1 = plt.subplots(1, 2, figsize=(18, 6))
ax1[0].plot(theta_CO2_p, CO2_p_corrected, color='red', ls='-', label='\u2225 CO2 PF Corrected')
#ax1[0].plot(theta_CO2_p, what_if_1, color='black', ls='-', label='⊥ Correction on \u2225 CO2 PF')
ax1[0].plot(theta_CO2_p, CO2_p_diffbsca, color='orange', ls='-', label='\u2225 CO2 PF Theory')
ax1[0].plot(theta_CO2_s, CO2_s_corrected, color='green', ls='-', label='⊥ CO2 PF Corrected')
#ax1[0].plot(theta_CO2_s, what_if_2, color='purple', ls='-', label='\u2225 Correction on ⊥ CO2 PF')
ax1[0].plot(theta_CO2_s, CO2_s_diffbsca, color='lawngreen', ls='-', label='⊥ CO2 PF Theory')
ax1[0].plot(theta_CO2_u, CO2_u_corrected, color='blue', ls='-', label='\u27f3 CO2 PF Corrected')
ax1[0].plot(theta_CO2_u, CO2_u_diffbsca, color='cyan', ls='-', label='\u27f3 CO2 PF Theory')
ax1[0].set_xlabel('\u03B8')
ax1[0].set_ylabel('$db_{sca}/d\u03B8$')
ax1[0].set_title('Corrected $CO_2$ Rayleigh Scattering Phase Functions')
ax1[0].grid(True)
ax1[0].legend(loc=1)
ax1[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1[1].plot(theta_CO2_p, G_p, color='red', ls='-', label='\u2225 Optical Transfer Function (Gp)')
ax1[1].plot(theta_CO2_p, optical_transfer_function(theta_CO2_p, *popt_G_p), color='orange', ls='-', label='Gp fit: Lorentzian\n coefficients: a=' + str('{:.3e}'.format(popt_G_p[0])) + ' b=' + str('{:.3e}'.format(popt_G_p[1])) + '\nc=' + str('{:.3e}'.format(popt_G_p[2])) + ' d=' + str('{:.3e}'.format(popt_G_p[3])))
#ax1[1].plot(theta_CO2_p, G_p_fit(theta_CO2_p), color='orange', ls='-', label='Parallel Optical Transfer Function Fit')
ax1[1].plot(theta_CO2_s, G_s, color='green', ls='-', label='⊥ Optical Transfer Function (Gs)')
#ax1[1].plot(theta_CO2_s, optical_transfer_function(theta_CO2_s, *popt_G_s), color='gold', ls='-', label='G fit: a=' + str('{:.3e}'.format(popt_G_s[0])) + ' b=' + str('{:.3e}'.format(popt_G_s[1])))
ax1[1].plot(theta_CO2_s, G_s_fit(theta_CO2_s), color='lawngreen', ls='-', label='Gs fit: Quartic Polynomial\n coefficients: a=' + str('{:.3e}'.format(G_s_coeffs[0])) + ' b=' + str('{:.3e}'.format(G_s_coeffs[1])) + '\nc=' + str('{:.3e}'.format(G_s_coeffs[2])) + ' d=' + str('{:.3e}'.format(G_s_coeffs[3])) + ' e=' + str('{:.3e}'.format(G_s_coeffs[4])))
ax1[1].plot(theta_CO2_u, G_u, color='blue', ls='-', label='\u27f3 Optical Transfer Function (Gu)')
ax1[1].plot(theta_CO2_u, optical_transfer_function(theta_CO2_u, *popt_G_u), color='cyan', ls='-', label='Gu fit: Lorentzian\n coefficients: a=' + str('{:.3e}'.format(popt_G_u[0])) + ' b=' + str('{:.3e}'.format(popt_G_u[1])) + '\nc=' + str('{:.3e}'.format(popt_G_u[2])) + ' d=' + str('{:.3e}'.format(popt_G_u[3])))
ax1[1].set_xlabel('\u03B8')
ax1[1].set_ylabel('$Mm^{-1}$/DN')
ax1[1].set_title('$CO_2$ Optical Transfer Function')
ax1[1].grid(True)
ax1[1].legend(loc=1)
plt.suptitle('Correction Result and Correction as a Function of Scattering Angle')
plt.savefig(save_directory + 'optical_transfer_function_result.png', format='png')
plt.savefig(save_directory + 'optical_transfer_function_result.pdf', format='pdf')
plt.tight_layout()
plt.show()

f2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(theta_CO2_p, CO2_p_pf, color='red', ls='-', label='\u2225 CO2 PF Original')
ax2.plot(theta_CO2_s, CO2_s_pf, color='green', ls='-', label='⊥ CO2 PF Original')
ax2.plot(theta_CO2_u, CO2_u_pf, color='blue', ls='-', label='\u27f3 CO2 PF Original')
ax2.set_xlabel('\u03B8')
ax2.set_ylabel('DN')
ax2.set_title('Original $CO_2$ Rayleigh Scattering Phase Functions')
ax2.grid(True)
ax2.legend(loc=1)
plt.savefig(save_directory + 'CO2_Original.png', format='png')
plt.savefig(save_directory + 'CO2_Original.pdf', format='pdf')
plt.tight_layout()
plt.show()


f3 = plt.figure( figsize=(30, 12))
gs = f3.add_gridspec(2, 3)
axa = f3.add_subplot(gs[0, 0])
axb = f3.add_subplot(gs[0, 1])
axc = f3.add_subplot(gs[0, 2])
axd = f3.add_subplot(gs[1, :])
axa.plot(theta_CO2_p, CO2_p_pf/np.sum(CO2_p_pf), color='black', ls='-', label='\u2225 CO2 PF Raw Meas. Normalized')
axa.plot(theta_CO2_p, CO2_p_corrected/np.sum(CO2_p_corrected), color='red', ls='-', label='\u2225 CO2 PF Corr. Meas. Normalized')
axa.plot(theta_CO2_p, CO2_p_diffbsca/np.sum(CO2_p_diffbsca), color='orange', ls='-', label='\u2225 CO2 PF Theory Normalized')
axa.set_xlabel('\u03B8')
axa.set_ylabel('Normalized Intensity')
axa.set_title('Parallel Polarization')
axa.grid(True)
axa.legend(loc=1)
axb.plot(theta_CO2_s, CO2_s_pf/np.sum(CO2_s_pf), color='black', ls='-', label='⊥ CO2 PF Raw Meas. Normalized')
axb.plot(theta_CO2_s, CO2_s_corrected/np.sum(CO2_s_corrected), color='green', ls='-', label='⊥ CO2 PF Corr. Meas. Normalized')
axb.plot(theta_CO2_s, CO2_s_diffbsca/np.sum(CO2_s_diffbsca), color='lawngreen', ls='-', label='⊥ CO2 PF Theory Normalized')
axb.set_xlabel('\u03B8')
axb.set_ylabel('Normalized Intensity')
axb.set_title('Perpendicular Polarization')
axb.grid(True)
axb.legend(loc=1)
axc.plot(theta_CO2_u, CO2_u_pf/np.sum(CO2_u_pf), color='black', ls='-', label='\u27f3 CO2 PF Raw Meas. Normalized')
axc.plot(theta_CO2_u, CO2_u_corrected/np.sum(CO2_u_corrected), color='blue', ls='-', label='\u27f3 CO2 PF Corr. Meas. Normalized')
axc.plot(theta_CO2_u, CO2_u_diffbsca/np.sum(CO2_u_diffbsca), color='cyan', ls='-', label='\u27f3 CO2 PF Theory Normalized')
axc.set_xlabel('\u03B8')
axc.set_ylabel('Normalized Intensity')
axc.set_title('Perpendicular Polarization')
axc.grid(True)
axc.legend(loc=1)
axd.plot(theta_CO2_p, percent_R_p, color='red', label='\u2225 Residuals')
axd.plot(theta_CO2_s, percent_R_s, color='green', label='⊥ Residuals')
axd.plot(theta_CO2_u, percent_R_u, color='blue', label='\u27f3 Residuals')
axd.set_xlabel('\u03B8')
axd.set_ylabel('Residuals')
axd.set_title('Percent Error as a Function of Scattering Angle')
axd.grid(True)
axd.legend(loc=1)
plt.suptitle('Normalized Measured and Theoretical $CO_2$ Rayleigh Scattering Phase Functions')
#plt.tight_layout()
plt.savefig(save_directory + 'CO2_Meas+Theory_Normalized.png', format='png')
plt.savefig(save_directory + 'CO2_Meas+Theory_Normalized.pdf', format='pdf')
plt.show()

# Things to do:
# 4. Apply the fit correction to PSL data
# 5. Compare the corrected PSL data to Mie theory


# Import and play with PSL data
PSL_p_directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-09/2020-03-09_Analysis/PSL/900 CAL/0R/SD_Particle.txt'
#PSL_s_directory =
#PSL_u_directory =

PSL_p_df = pd.read_csv(PSL_p_directory, sep=',', header=0)
PSL_p_pf = PSL_p_df['Sample Intensity gfit']
PSL_p_theta = PSL_p_df['Sample Theta']



