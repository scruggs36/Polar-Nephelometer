'''
Austen K. Scruggs
10-23-2019
Description: Looped plotting rayleigh scattering data at different polarizations, updated
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import pchip_interpolate

# Rayleigh theory
# 0 retardance flat case perpendicular SR
sr_path = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0R/SD_Rayleigh_Straigh_From_Laser_NoRetarders.txt'
sr_path_retarded = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0R/SD_Rayleigh_RR_Exact_Dolgos_QWP_S0.txt'
# 0.5 retardance parallel case SL
sl_path = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-03-08/Analysis/Rayleigh/plot_directory/0.5R/SD_Rayleigh_LCVR@0.5R+QWP@F0.txt'
# theory directory
theory_path = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-02-08/Rayleigh_Analysis' + '/Rayleigh Theory/Rayleigh_PF.txt'
save_directory = '/home/austen/Desktop/Recent/Rayleigh Validation/'

# Cosine Squared Fit Function
def Rayleigh_Circular_Polarized(theta, a, b):
    rads = (np.asarray(theta) * np.pi) / 180.0
    f = a * (1 + b * (np.cos(rads)**2))
    return f


def Rayleigh_Linear_Polarized(theta, a, b):
    rads = (np.asarray(theta) * np.pi) / 180.0
    f = a * (np.cos(rads)**2) + b
    return f


def Rayleigh_Residuals(x, theta, measurement):
    rads = (np.asarray(theta) * np.pi) / 180.0
    residuals = measurement - (x[0] * (np.cos(rads)**2) + x[1])
    return residuals



# SL Measurements
sl_rayleigh_meas = pd.read_csv(sl_path, sep=',', header=0)
# drops rows with any nan value present
#SL_rayleigh_meas = SL_rayleigh_meas.dropna()
sl_gas_meas = np.array(sl_rayleigh_meas['CO2 Intensity gfit corr'])
sl_gas_meas = np.array(sl_gas_meas[~np.isnan(sl_gas_meas)])
sl_gas_theta = np.array(sl_rayleigh_meas['CO2 Theta'])
sl_gas_theta = np.array(sl_gas_theta[~np.isnan(sl_gas_theta)])

# SR measurement
sr_rayleigh_meas = pd.read_csv(sr_path, sep=',', header=0)
#SR_rayleigh_meas = SR_rayleigh_meas.dropna()
sr_gas_meas = np.array(sr_rayleigh_meas['CO2 Intensity gfit corr'])
sr_gas_meas = np.array(sr_gas_meas[~np.isnan(sr_gas_meas)])
sr_gas_theta = np.array(sr_rayleigh_meas['CO2 Theta'])
sr_gas_theta = np.array(sr_gas_theta[~np.isnan(sr_gas_theta)])

# SR measuerement retarded
sr_rayleigh_meas_retarded = pd.read_csv(sr_path_retarded, sep=',', header=0)
#SR_rayleigh_meas = SR_rayleigh_meas.dropna()
sr_gas_meas_retarded = np.array(sr_rayleigh_meas_retarded['CO2 Intensity gfit corr'])
sr_gas_theta_retarded = np.array(sr_rayleigh_meas_retarded['CO2 Theta'])
sr_gas_theta_retarded = np.array(sr_gas_theta_retarded[~np.isnan(sr_gas_meas_retarded)])
sr_gas_meas_retarded = np.array(sr_gas_meas_retarded[~np.isnan(sr_gas_meas_retarded)])


'''
# conversion profile numbers to angle
slope = 0.2095
intercept = -3.1433
SL_theta = [(slope * x) + intercept for x in sl_gas_columns]
SR_theta = [(slope * x) + intercept for x in sr_gas_columns]
SR_theta_retarded = [(slope * x) + intercept for x in sr_gas_columns_retarded]
'''

rayleigh_theory_df = pd.read_csv(theory_path, sep=',', header=0)
SL_gas_theory = rayleigh_theory_df['CO2 anisotropic']
SL_gas_theory_theta = rayleigh_theory_df['Theta']
SL_gas_theory_pchip = pchip_interpolate(SL_gas_theory_theta, SL_gas_theory, sl_gas_theta)

print(len(sl_gas_theta))
print(len(sl_gas_meas))
print(len(sr_gas_theta))
print(len(sr_gas_meas))
print(len(sr_gas_theta_retarded))
print(len(sr_gas_meas_retarded))

popt_SL_gas, pcov_SL_gas = curve_fit(Rayleigh_Linear_Polarized, sl_gas_theta, sl_gas_meas, p0=[np.amax(sl_gas_meas)/np.amin(sl_gas_meas), np.amin(sl_gas_meas)])
SL_gas_residuals = sl_gas_meas - Rayleigh_Linear_Polarized(sl_gas_theta, *popt_SL_gas)
SL_gas_error = (SL_gas_residuals / Rayleigh_Linear_Polarized(sl_gas_theta, *popt_SL_gas)) * 100

'''
popt_SR_gas, pcov_SR_gas = curve_fit(Rayleigh_Circular_Polarized, SR_theta, SR_gas_meas, p0=[np.amin(SR_gas_meas), np.amax(SR_gas_meas)/np.amin(SR_gas_meas)])
SR_gas_residuals = SR_gas_meas - Rayleigh_Circular_Polarized(SR_theta, *popt_SR_gas)
SR_gas_error = (SR_gas_residuals / Rayleigh_Circular_Polarized(SR_theta, *popt_SR_gas)) * 100

guess = np.array([np.amax(SL_gas_meas)/np.amin(SL_gas_meas), np.amin(SL_gas_meas)])
SL_ls_result = least_squares(Rayleigh_Residuals, guess, method='lm', args=(SL_theta, SL_gas_meas))
#SL_ls_percent_error = (SL_ls_result.fun / (SL_gas_meas)) * 100
SL_ls_percent_error = (SL_ls_result.fun / (Rayleigh_Linear_Polarized(SL_theta, SL_ls_result.x[0], SL_ls_result.x[1]))) * 100
# figure font parameters

# parameters in function to play with
# sony ccd specs came from webpage https://www.ximea.com/en/products/cameras-filtered-by-sensor-sizes/sony-icx285-cooled-mono-scientific-grade-camera
power = .8
wav_nm = 663
quantum_efficiency = 57.5
#gain_dB = 6 an expression of the dynamic range of the detector
# FWC = 24000, full well capacity
e_ADU = 1.5
image_exposure_time = 300
watt_guess = np.array([1.00E-8, 0.000005])
conc = 2.54E19
ls_watt_result = least_squares(Power_v_Theta, watt_guess, method='lm', args=(SL_gas_meas, SL_gas_theory_pchip, power, conc))
'''


# figure font parameters
q = 20
r = 14
s = 14

# measurement figure
f, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(sl_gas_theta, sl_gas_meas, 'r-', label='SL Meas.')
#ax[0].plot(sl_gas_theta, (np.amin(sl_gas_meas)/np.amin(SL_gas_theory_pchip))*SL_gas_theory_pchip, color='blue', ls='-', label='SL Theory')
ax[0].plot(sl_gas_theta, Rayleigh_Linear_Polarized(sl_gas_theta, *popt_SL_gas), color='blue', ls='-', label='SL Theory Fit')
ax[0].set_xlabel('\u03b8', fontsize=r)
ax[0].set_ylabel('Intensity', fontsize=r)
ax[0].set_title('$CO_2$ Gas Angular Scattering SL', fontsize=q)
ax[0].grid(True)
ax[0].legend(loc=1, fontsize=s)
ax[1].plot(sr_gas_theta, sr_gas_meas, color='black', ls='-', label='SR Meas.')
ax[1].plot(sr_gas_theta_retarded, sr_gas_meas_retarded, color='green', ls='-', label='SR Meas. + QWP')
ax[1].plot(sr_gas_theta, (6.0/7.0)*sr_gas_meas[425]*np.ones(len(sr_gas_theta)), color='red', ls='-', label='SR Theory')
ax[1].set_xlabel('\u03b8', fontsize=r)
ax[1].set_ylabel('Intensity', fontsize=r)
ax[1].set_title('Rayleigh Gas Angular Scattering SR', fontsize=q)
ax[1].grid(True, which='both')
ax[1].legend(loc=1, fontsize=s)
plt.tight_layout()
plt.savefig(save_directory + 'CO2_lamda0&05_NEW.png', format='png')
plt.savefig(save_directory + 'CO2_lamda0&05_NEW.pdf', format='pdf')
plt.show()

'''
# SL curve_fit figure
f0, ax0 = plt.subplots(2, 1, figsize=(20, 10))
ax0[0].plot(SL_theta, SL_gas_meas, label='$CO_2$ \u03bb = 0.5\n', ls='-', color='black')
#ax0[0].plot(SL_theta, Rayleigh_Linear_Polarized(SL_theta, *popt_SL_gas), ls='--', color='red', label='Fit: ' + str('{:.3f}'.format(popt_SL_gas[0])) + '*$cos^2(\u03b8)$) + '  + str('{:.3f}'.format(popt_SL_gas[1])))
ax0[0].set_title('$CO_2$ Angular Scattering (\u03bb = 0.5)', fontsize=q)
ax0[0].set_ylabel('Intensity', fontsize=r)
ax0[0].set_xlabel('\u03b8', fontsize=r)
ax0[0].grid(True)
ax0[0].legend(loc=1, fontsize=s)
#ax0[1].plot(SL_theta, SL_gas_error, color='black', ls='-', label='residuals')
ax0[1].set_title('$CO_2$ Percent Error as a Function of \u03b8 at Retardance \u03bb = 0.5', fontsize=q)
ax0[1].set_ylabel('Intensity', fontsize=r)
ax0[1].set_xlabel('\u03b8', fontsize=r)
ax0[1].grid(True)
ax0[1].legend(loc=1, fontsize=s)
plt.tight_layout()
plt.savefig(save_directory + 'CO2_lamda0.5_RAYFIG.png', format='png')
plt.savefig(save_directory + 'CO2_lamda0.5_RAYFIG.pdf', format='pdf')
plt.show()

# SR curve_fit figure
f1, ax1 = plt.subplots(2, 1, figsize=(20, 10))
ax1[0].plot(SR_theta, SR_gas_meas, label='$CO_2$ \u03bb = 0.0\n', ls='-', color='black')
#ax1[0].plot(SR_theta, Rayleigh_Circular_Polarized(SR_theta, *popt_SR_gas), ls='--', color='red', label='Fit: ' + str('{:.3f}'.format(popt_SR_gas[0])) + '(1 + ' + str('{:.3f}'.format(popt_SR_gas[1])) + '$cos^2(\u03b8)$)\n')
ax1[0].set_title('$CO_2$ Angular Scattering (\u03bb = 0.0)', fontsize=q)
ax1[0].set_ylabel('Intensity', fontsize=r)
ax1[0].set_xlabel('\u03b8', fontsize=r)
ax1[0].grid(True)
ax1[0].legend(loc=1, fontsize=s)
#ax1[1].plot(SR_theta, SR_gas_error, color='black', ls='-', label='residuals')
ax1[1].set_title('$CO_2$ Percent Error as a Function of \u03b8 at Retardance \u03bb = 0.0', fontsize=q)
ax1[1].set_ylabel('Intensity', fontsize=r)
ax1[1].set_xlabel('\u03b8', fontsize=r)
ax1[1].grid(True)
ax1[1].legend(loc=1, fontsize=s)
plt.tight_layout()
plt.savefig(save_directory + 'CO2_lamda0_RAYFIG.png', format='png')
plt.savefig(save_directory + 'CO2_lamda0_RAYFIG.pdf', format='pdf')
plt.show()


# SL nonlinear least squares minimization
f2, ax2 = plt.subplots(2, 1, figsize=(20, 10))
ax2[0].plot(SL_theta, SL_gas_meas, label='$CO_2$ \u03bb = 0.5\n', ls='-', color='black')
ax2[0].plot(SL_theta, Rayleigh_Linear_Polarized(SL_theta, SL_ls_result.x[0], SL_ls_result.x[1]), ls='--', color='red', label='Fit: ' + str('{:.3f}'.format(SL_ls_result.x[0])) + '*$cos^2(\u03b8)$) + '  + str('{:.3f}'.format(SL_ls_result.x[1])))
ax2[0].set_title('$CO_2$ Angular Scattering (\u03bb = 0.5)', fontsize=q)
ax2[0].set_ylabel('Intensity', fontsize=r)
ax2[0].set_xlabel('\u03b8', fontsize=r)
ax2[0].grid(True)
ax2[0].legend(loc=1, fontsize=s)
ax2[1].plot(SL_theta, SL_ls_percent_error, color='black', ls='-', label='residuals')
ax2[1].set_title('$CO_2$ Residual as a Function of \u03b8 at Retardance \u03bb = 0.5', fontsize=q)
ax2[1].set_ylabel('Intensity', fontsize=r)
ax2[1].set_xlabel('\u03b8', fontsize=r)
ax2[1].grid(True)
ax2[1].legend(loc=1, fontsize=s)
plt.tight_layout()
plt.savefig(save_directory + 'CO2_lamda0.5_nlls.png', format='png')
plt.savefig(save_directory + 'CO2_lamda0.5_nlls.pdf', format='pdf')
plt.show()

'''


