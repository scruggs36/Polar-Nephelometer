'''
Austen K. Scruggs
01-20-2020
Description: Writing a Function for a nonlinear least squares to find the size distribution, the real refractive index,
and all other parameters
'''

import numpy as np
import pandas as pd
import PyMieScatt as ps
import matplotlib.pyplot as plt
from scipy.integrate import simps
from math import sqrt, log, pi
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate, interp1d
from sklearn.linear_model import LinearRegression


# Gaussian distribution function
def Gaussian(x, mu, sigma):
   return 1/(sigma * sqrt(2*pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# log normal distribution function, we might want it normalized, check if the equation is right...
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))


# function to obtain residuals from a NLLS
def Residuals_DLP(x, w_n, SL_M, SL_Theta, SR_M, SR_Theta, SU_M, SU_Theta):
    # pre allocation
    SL_2darray = []
    SR_2darray = []
    SU_2darray = []
    # Data SL
    SL_M = np.array(SL_M)
    #SL_M = SL_M[~np.isnan(SL_M)]
    #SL_Theta = SL_Theta[~np.isnan(SL_M)]
    # Data SR
    SR_M = np.array(SR_M)
    #SR_M = SR_M[~np.isnan(SR_M)]
    #SR_Theta = SR_Theta[~np.isnan(SR_M)]
    # Data SU
    SU_M = np.array(SU_M)
    #SU_M = SR_M[~np.isnan(SU_M)]
    #SU_Theta = SR_Theta[~np.isnan(SU_M)]
    # Pchip measurement
    SR_M_Pchip = pchip_interpolate(xi=SR_Theta, yi=SR_M, x=SL_Theta, der=0, axis=0)
    SU_M_Pchip = pchip_interpolate(xi=SU_Theta, yi=SU_M, x=SL_Theta, der=0, axis=0)
    # DLP measurement, no need for pchip it should already be the same length as SL
    DLP = (-1.0 * (SL_M - SR_M_Pchip)) / (SL_M + SR_M_Pchip)
    # size distribution and weights calculation
    sizes = np.arange(850.0, 950.0, 5.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SL_2darray.append(SL)
        SR_2darray.append(SR)
        SU_2darray.append(SU)
    SL_average = np.average(SL_2darray, axis=0, weights=weights)
    SR_average = np.average(SR_2darray, axis=0, weights=weights)
    SU_average = np.average(SU_2darray, axis=0, weights=weights)
    SL_average_pchip = pchip_interpolate(xi=theta_mie, yi=SL_average, x=SL_Theta, der=0, axis=0)
    SR_average_pchip = pchip_interpolate(xi=theta_mie, yi=SR_average, x=SL_Theta, der=0, axis=0)
    SU_average_pchip = pchip_interpolate(xi=theta_mie, yi=SU_average, x=SL_Theta, der=0, axis=0)
    DLP_average_pchip = (-1.0 * (SL_average_pchip - SR_average_pchip)) / (SL_average_pchip + SR_average_pchip)
    #print(DLP.shape)
    #print(DLP_average_pchip.shape)
    # compute the sum of mean squared errors relaitve to the theory
    sl_error = ((x[3] * SL_M - SL_average_pchip)**2)/SL_average_pchip
    sr_error = ((x[4] * SR_M_Pchip - SR_average_pchip)**2)/SR_average_pchip
    su_error = ((x[5] * SU_M_Pchip - SU_average_pchip)**2)/SU_average_pchip
    dlp_error = ((x[6] * DLP - DLP_average_pchip)**2)/DLP_average_pchip
    residuals = np.sum(np.sum([sl_error, sr_error, su_error, dlp_error], axis=1), axis=None)
    print('summed residuals: ', residuals,'Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'SL Scalar: ', x[3], 'SR Scalar: ', x[4], 'SU Scalar: ', x[5], 'DLP Scalar', x[6], 'Weights: ', weights)
    return residuals


def Asymmetry_Parameter(PF, theta):
    y = np.cos((pi * theta)/180.0)*(PF/np.linalg.norm(PF))*np.sin((pi * theta)/180.0)
    g = 0.5 * simps(y, theta)
    return g


'''
the function above remove nans from the data
Explanation:
The inner function, numpy.isnan returns a boolean/logical array which has the value True everywhere that x is not-a-number.
As we want the opposite, we use the logical-not operator, ~ to get an array with Trues everywhere that x is a valid number.
Lastly we use this logical array to index into the original array x, to retrieve just the non-NaN values.
'''

# this is to tes the nonlinear least square under levenberg Marquad Method
#exp_directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-21/2020-01-21_Analysis/Measurement/SD_Particle.txt'
sl_exp_dir = '/home/austen/Desktop/Recent/Retrieval/SD_Particle_900nm_0R_2019-11-21.txt'
sr_exp_dir = '/home/austen/Desktop/Recent/Retrieval/SD_Particle_900nm_0.5R_2019-11-21_1s.txt'
su_exp_dir = '/home/austen/Desktop/Recent/Retrieval/Calibrated_Data_PSL900nm.txt'
save_directory = '/home/austen/Desktop/Recent/Retrieval/'
sl_data = pd.read_csv(sl_exp_dir, sep=',', header=0)
sr_data = pd.read_csv(sr_exp_dir, sep=',', header=0)
su_data = pd.read_csv(su_exp_dir, sep=',', header=0)
sl_measurement = sl_data['Sample Intensity gfit']
sl_measurement = np.array(sl_measurement[~np.isnan(sl_measurement)])
sr_measurement = sr_data['Sample Intensity gfit']
sr_measurement = np.array(sr_measurement[~np.isnan(sr_measurement)])
su_measurement = su_data['Exp Smoothed Intensity']
su_measurement = np.array(su_measurement[~np.isnan(su_measurement)])
slp = .2049
inter = -2.7594
sl_theta = sl_data['Sample Columns']
sl_theta = np.array(sl_theta[~np.isnan(sl_theta)])
sl_theta = np.array([(slp * x) + inter for x in sl_theta])
sr_theta = sr_data['Sample Columns']
sr_theta = np.array(sr_theta[~np.isnan(sr_theta)])
sr_theta = np.array([(slp * x) + inter for x in sr_theta])
su_theta = su_data['PN to Angle']
su_theta = np.array(su_theta[~np.isnan(su_measurement)])


'''
# you should already have calibrated theta by the time you get here... we are looking to take all calibration conversion
# to theta info out this script
#slope = 0.2052
#intercept = 0.7795
#theta = np.array([(slope * x) + intercept for x in su_columns])
'''

# the only input parameter other than the calibrated data
wavelength = 663.0
# cut off
cut_off = 200
# NLLS result, bounds = ([array containing all parameter lower bounds],[array containing all parameter upper bounds])
result_DLP = least_squares(Residuals_DLP, x0=[903.0, 4.1, 1.598, 0.12, 0.12, 0.12, 0.12], method='trf', args=(wavelength, sl_measurement[cut_off:-cut_off], sl_theta[cut_off:-cut_off], sr_measurement[cut_off:-cut_off], sr_theta[cut_off:-cut_off], su_measurement[cut_off:-cut_off], su_theta[cut_off:-cut_off]), bounds=([850.0, 1.00, 1.00, .001, .001, .001, .001],[950.0, 100.0, 2.00, 1.000, 1.000, 1.000, 1.000]))

# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_DLP.x[0])
print('\u03c3: ', result_DLP.x[1])
print('m: ', result_DLP.x[2])
print('sl scalar: ', result_DLP.x[3])
print('sr scalar: ', result_DLP.x[4])
print('su scalar: ', result_DLP.x[5])
print('dlp scalar: ', result_DLP.x[6])
print('iterations: ', result_DLP.nfev)
print('status: ', result_DLP.status)

# compute weights using the found distribution parameters
sizes = np.arange(850.0, 950.0, 5.0)
weights = [Gaussian(element, result_DLP.x[0], result_DLP.x[1]) for element in sizes]

# calculate g with recovered parameters, parameters is an an array of arrays of qext, qsca, qabs, g, qpr, qback, qratio at each size
print('-------asymmetry parameter-------')
parameters = ps.MieQ_withDiameterRange(m=result_DLP.x[2], wavelength=wavelength, nMedium=1.0, diameterRange=(850.0, 950.0), nd=20)
#print(parameters)
g_average = np.average(parameters[4], axis=0, weights=weights)
print('g(theory): ', g_average)

# recovering NLLS best mie theory match
SL_2d = []
SR_2d = []
SU_2d = []

# size range input into the function needs to be here to calculate the best weights outside NLLS
size_result = np.arange(850.0, 950.0, 5.0)
weights_result = [Gaussian(element, result_DLP.x[0], result_DLP.x[1]) for element in size_result]
for element in size_result:
    theta_mie, SL, SR, SU = ps.ScatteringFunction(result_DLP.x[2], wavelength, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    SL_2d.append(SL)
    SR_2d.append(SR)
    SU_2d.append(SU)
SL_mie_NLLS = np.average(SL_2d, axis=0, weights=weights_result)
SR_mie_NLLS = np.average(SR_2d, axis=0, weights=weights_result)
SU_mie_NLLS = np.average(SU_2d, axis=0, weights=weights_result)
SL_mie_NLLS_pchip = pchip_interpolate(xi=theta_mie, yi=SL_mie_NLLS, x=sl_theta, der=0, axis=0)
SR_mie_NLLS_pchip = pchip_interpolate(xi=theta_mie, yi=SR_mie_NLLS, x=sl_theta, der=0, axis=0)
SU_mie_NLLS_pchip = pchip_interpolate(xi=theta_mie, yi=SU_mie_NLLS, x=sl_theta, der=0, axis=0)
DLP_mie_NLLS_pchip = (-1.0 * (SL_mie_NLLS_pchip - SR_mie_NLLS_pchip)) / (SL_mie_NLLS_pchip + SR_mie_NLLS_pchip)


# find g
# we use this in the event we wanna integrate with quad which isn't built yet
# we also use this to integrate with simpson method which is built
sl_measurement_cspline = interp1d(sl_theta, sl_measurement, kind='cubic', fill_value='extrapolate')
sr_measurement_cspline = interp1d(sr_theta, sr_measurement, kind='cubic', fill_value='extrapolate')
su_measurement_cspline = interp1d(su_theta, su_measurement, kind='cubic', fill_value='extrapolate')
slope = 0.2049 #0.2052
theta_full = np.arange(0.0, 180.0 + slope, slope)
sl_extrapolated = sl_measurement_cspline(theta_full)
sr_extrapolated = sr_measurement_cspline(theta_full)
su_extrapolated = su_measurement_cspline(theta_full)
g = Asymmetry_Parameter(su_extrapolated, theta_full)
print('g(measurement): ', g)

# pchip measurements because this happens in the residual function but not in the script!
sr_meas_pchip = pchip_interpolate(xi=sr_theta, yi=sr_measurement, x=sl_theta, der=0, axis=0)
su_meas_pchip = pchip_interpolate(xi=su_theta, yi=su_measurement, x=sl_theta, der=0, axis=0)
dlp_meas_pchip = (-1.0 * (sl_measurement - sr_meas_pchip)) / (sl_measurement + sr_meas_pchip)

# calculating residuals
sl_residuals = sl_measurement - SL_mie_NLLS_pchip
sr_residuals = sr_meas_pchip - SR_mie_NLLS_pchip
su_residuals = su_meas_pchip - SU_mie_NLLS_pchip
dlp_residuals = dlp_meas_pchip - DLP_mie_NLLS_pchip

# plotting the results
# font sizes for figures
f_title = 24
f_axes = 18
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
# making figure of NLLS fits comparing data and theory
f0, ax0 = plt.subplots(2, 2, figsize=(16, 12))
ax0[0, 0].semilogy(sl_theta, SL_mie_NLLS_pchip, color='red', ls='-', label='NLLS: ' + 'm: ' + str('{:.3f}'.format(result_DLP.x[2])) + ' \u03bc: ' + str('{:.3f}'.format(result_DLP.x[0])) + '\n\u03c3: ' + str('{:.3f}'.format(result_DLP.x[1])))
ax0[0, 0].semilogy(sl_theta,  result_DLP.x[3] * sl_measurement, color='black', ls='-', label='SL Measurement')
ax0[0, 0].semilogy(sl_theta[cut_off:-cut_off],  result_DLP.x[3] * sl_measurement[cut_off:-cut_off], color='gray', ls='-', lw=7.0, alpha=0.5, label='Used Points')
ax0[0, 0].set_title('Non-Linear Least Squares Fit: SL', fontsize=f_title)
ax0[0, 0].set_xlabel('\u03b8', fontsize=f_axes)
ax0[0, 0].set_ylabel('Intensity', fontsize=f_axes)
ax0[0, 0].grid(True, which='both')
ax0[0, 0].legend(loc=1, fontsize=f_axes)
ax0[0, 1].semilogy(sl_theta, SR_mie_NLLS_pchip, color='green', ls='-', label='NLLS: ' + 'm: ' + str('{:.3f}'.format(result_DLP.x[2])) + ' \u03bc: ' + str('{:.3f}'.format(result_DLP.x[0])) + '\n\u03c3: ' + str('{:.3f}'.format(result_DLP.x[1])))
ax0[0, 1].semilogy(sl_theta,  result_DLP.x[4] * sr_meas_pchip, color='black', ls='-', label='SRMeasurement')
ax0[0, 1].semilogy(sl_theta[cut_off:-cut_off],  result_DLP.x[4] * sr_meas_pchip[cut_off:-cut_off], color='gray', ls='-', lw=7.0, alpha=0.5, label='Used Points')
ax0[0, 1].set_title('Non-Linear Least Squares Fit: SR', fontsize=f_title)
ax0[0, 1].set_xlabel('\u03b8', fontsize=f_axes)
ax0[0, 1].set_ylabel('Intensity', fontsize=f_axes)
ax0[0, 1].grid(True, which='both')
ax0[0, 1].legend(loc=1, fontsize=f_axes)
ax0[1, 0].semilogy(sl_theta, SU_mie_NLLS_pchip, color='blue', ls='-', label='NLLS: ' + 'm: ' + str('{:.3f}'.format(result_DLP.x[2])) + ' \u03bc: ' + str('{:.3f}'.format(result_DLP.x[0])) + '\n\u03c3: ' + str('{:.3f}'.format(result_DLP.x[1])))
ax0[1, 0].semilogy(sl_theta,  result_DLP.x[5] * su_meas_pchip, color='black', ls='-', label='SU Measurement')
ax0[1, 0].semilogy(sl_theta[cut_off:-cut_off],  result_DLP.x[5] * su_meas_pchip[cut_off:-cut_off], color='gray', ls='-', lw=7.0, alpha=0.5, label='Used Points')
ax0[1, 0].set_title('Non-Linear Least Squares Fit: SU', fontsize=f_title)
ax0[1, 0].set_xlabel('\u03b8', fontsize=f_axes)
ax0[1, 0].set_ylabel('Intensity', fontsize=f_axes)
ax0[1, 0].grid(True, which='both')
ax0[1, 0].legend(loc=1, fontsize=f_axes)
ax0[1, 1].semilogy(sl_theta, DLP_mie_NLLS_pchip, color='purple', ls='-', label='NLLS: ' + 'm: ' + str('{:.3f}'.format(result_DLP.x[2])) + ' \u03bc: ' + str('{:.3f}'.format(result_DLP.x[0])) + '\n\u03c3: ' + str('{:.3f}'.format(result_DLP.x[1])))
ax0[1, 1].semilogy(sl_theta,  result_DLP.x[6] * dlp_meas_pchip, color='black', ls='-', label='DLP Measurement')
ax0[1, 1].semilogy(sl_theta[cut_off:-cut_off],  result_DLP.x[6] * dlp_meas_pchip[cut_off:-cut_off], color='gray', ls='-', lw=7.0, alpha=0.5, label='Used Points')
ax0[1, 1].set_title('Non-Linear Least Squares Fit: DLP', fontsize=f_title)
ax0[1, 1].set_xlabel('\u03b8', fontsize=f_axes)
ax0[1, 1].set_ylabel('Intensity', fontsize=f_axes)
ax0[1, 1].grid(True, which='both')
ax0[1, 1].legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(save_directory + 'NL_Least_Squares_Result_DLP.png', format='png')
plt.savefig(save_directory + 'NL_Least_Squares_Result_DLP.pdf', format='pdf')
plt.show()

# plotting residuals
f1, ax1 = plt.subplots(figsize=(16, 12))
ax1.plot(sl_theta, sl_residuals, color='red', ls='-', label='SL Residuals')
ax1.plot(sl_theta[cut_off:-cut_off], sl_residuals[cut_off:-cut_off], color='red', ls='-', lw=7.0, alpha=0.5, label='SL Points Used')
ax1.plot(sl_theta, sr_residuals, color='green', ls='-', label='SR Residuals')
ax1.plot(sl_theta[cut_off:-cut_off], sr_residuals[cut_off:-cut_off], color='green', ls='-', lw=7.0, alpha=0.5, label='SR Points Used')
ax1.plot(sl_theta, su_residuals, color='blue', ls='-', label='SU Residuals')
ax1.plot(sl_theta[cut_off:-cut_off], su_residuals[cut_off:-cut_off], color='blue', ls='-', lw=7.0, alpha=0.5, label='SU Points Used')
ax1.plot(sl_theta, dlp_residuals, color='purple', ls='-', label='DLP Residuals')
ax1.plot(sl_theta[cut_off:-cut_off], dlp_residuals[cut_off:-cut_off], color='purple', ls='-', lw=7.0, alpha=0.5, label='DLP Points Used')
ax1.set_title('Non-Linear Least Squares Fit Residuals', fontsize=f_title)
ax1.set_xlabel('\u03b8', fontsize=f_axes)
ax1.set_ylabel('Intensity', fontsize=f_axes)
ax1.grid(True, which='both')
ax1.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(save_directory + 'NL_Least_Squares_Result_DLP_Residuals.png', format='png')
plt.savefig(save_directory + 'NL_Least_Squares_Result_DLP_Residuals.pdf', format='pdf')
plt.show()

# fitting intensities
SL_X = result_DLP.x[3] * sl_measurement[cut_off:-cut_off]
SL_X = SL_X.reshape(-1, 1)
SL_Y = SL_mie_NLLS_pchip[cut_off:-cut_off]
SL_Y = SL_Y.reshape(-1, 1)
SR_X = result_DLP.x[4] * sr_meas_pchip[cut_off:-cut_off]
SR_X = SR_X.reshape(-1, 1)
SR_Y = SR_mie_NLLS_pchip[cut_off:-cut_off]
SR_Y = SR_Y.reshape(-1, 1)
SU_X = result_DLP.x[5] * su_meas_pchip[cut_off:-cut_off]
SU_X = SU_X.reshape(-1, 1)
SU_Y = SU_mie_NLLS_pchip[cut_off:-cut_off]
SU_Y = SU_Y.reshape(-1, 1)
reg_SL = LinearRegression().fit(SL_X, SL_Y)
reg_SR = LinearRegression().fit(SR_X, SR_Y)
reg_SU = LinearRegression().fit(SU_X, SU_Y)
print(reg_SL.coef_)
print(reg_SL.intercept_)
# plot fit linear regressions of theory v measured intensities
f2, ax2 = plt.subplots(1, 3, figsize=(20, 6))
ax2[0].plot(SL_X, SL_Y, marker='o', color='red', label='SL: Meas. v. Theory')
ax2[0].plot(SL_X, (reg_SL.coef_[0][0] * SL_X) + reg_SL.intercept_[0], color='black', label='SL: Meas. v. Theory Fit\n' + 'y = ' + str('{:.3f}'.format(reg_SL.coef_[0][0])) + 'x + ' + str('{:.3f}'.format(reg_SL.intercept_[0])))
ax2[0].set_title('Theory and Measured\n SL Intensities Compared', fontsize=f_title)
ax2[0].set_xlabel('Measured Intensity', fontsize=f_axes)
ax2[0].set_ylabel('Theoretical Intensity', fontsize=f_axes)
ax2[0].grid(True, which='both')
ax2[0].legend(loc=1, fontsize=f_axes)
ax2[1].plot(SR_X, SR_Y, marker='o', color='green', label='SR: Meas. v. Theory')
ax2[1].plot(SR_X, (reg_SR.coef_[0][0] * SR_X) + reg_SR.intercept_[0], color='black', ls='-', label='SR: Meas. v. Theory Fit\n' + 'y = ' + str('{:.3f}'.format(reg_SR.coef_[0][0])) + 'x + ' + str('{:.3f}'.format(reg_SR.intercept_[0])))
ax2[1].set_title('Theory and Measured\n SR Intensities Compared', fontsize=f_title)
ax2[1].set_xlabel('Measured Intensity', fontsize=f_axes)
ax2[1].set_ylabel('Theoretical Intensity', fontsize=f_axes)
ax2[1].grid(True, which='both')
ax2[1].legend(loc=1, fontsize=f_axes)
ax2[2].plot(SU_X, SU_Y, marker='o', color='blue', label='SU: Meas. v. Theory')
ax2[2].plot(SU_X, (reg_SU.coef_[0][0] * SU_X) + reg_SU.intercept_[0], color='black', ls='-', label='SU: Meas. v. Theory Fit\n' + 'y = ' + str('{:.3f}'.format(reg_SU.coef_[0][0])) + 'x + ' + str('{:.3f}'.format(reg_SU.intercept_[0])))
ax2[2].set_title('Theory and Measured\n SU Intensities Compared', fontsize=f_title)
ax2[2].set_xlabel('Measured Intensity', fontsize=f_axes)
ax2[2].set_ylabel('Theoretical Intensity', fontsize=f_axes)
ax2[2].grid(True, which='both')
ax2[2].legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(save_directory + 'IntVInt.pdf', format='pdf')
plt.savefig(save_directory + 'IntVInt.png', format='png')
plt.show()

