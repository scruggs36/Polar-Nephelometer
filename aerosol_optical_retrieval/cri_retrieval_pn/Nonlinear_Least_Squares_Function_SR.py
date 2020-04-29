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

# cut off
cut_off = 165
# Gaussian distribution function
def Gaussian(x, mu, sigma):
   return 1/(sigma * sqrt(2*pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# log normal distribution function, we might want it normalized, check if the equation is right...
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))



# x -->  x[0] = m, x[1] = mu, x[2] = scalar
# Measurement --> SL_M
def Residuals_SL(x, w_n, SL_M, SL_Theta):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(850.0, 950.0, 5.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SL_2darray.append(SL)
    SL_average = np.average(SL_2darray, axis=0, weights=weights)
    SL_average_pchip = pchip_interpolate(xi=theta_mie, yi=SL_average, x=SL_Theta, der=0, axis=0)
    residuals = np.sum(((np.absolute(x[3] * SL_M[cut_off:len(SL_M)] - SL_average_pchip[cut_off:len(SL_M)]))/SL_average_pchip[cut_off:len(SL_M)])*100.0)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Scalar: ', x[3], 'Weights: ', weights)
    return residuals



def Residuals_SR(x, w_n, SR_M, SR_Theta):
    # pre allocation
    SR_2darray = []
    # Data
    SR_M = np.array(SR_M)
    SR_M = SR_M[~np.isnan(SR_M)]
    SR_Theta = SR_Theta[~np.isnan(SR_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(850.0, 950.0, 5.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SR_2darray.append(SR)
    SR_average = np.average(SR_2darray, axis=0, weights=weights)
    SR_average_pchip = pchip_interpolate(xi=theta_mie, yi=SR_average, x=SR_Theta, der=0, axis=0)
    residuals = np.sum(((np.absolute(x[3] * SR_M[cut_off:len(SR_M)] - SR_average_pchip[cut_off:len(SR_M)]))/SR_average_pchip[cut_off:len(SR_M)])*100.0)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Scalar: ', x[3], 'Weights: ', weights)
    return residuals


def Residuals_SU(x, w_n, SU_M, SU_Theta):
    # pre allocation
    SU_2darray = []
    # Data
    SU_M = np.array(SU_M)
    SU_M = SU_M[~np.isnan(SU_M)]
    SU_Theta = SU_Theta[~np.isnan(SU_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(850.0, 950.0, 5.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SU_2darray.append(SU)
    SU_average = np.average(SU_2darray, axis=0, weights=weights)
    SU_average_pchip = pchip_interpolate(xi=theta_mie, yi=SU_average, x=SU_Theta, der=0, axis=0)
    residuals = np.sum(((np.absolute(x[3] * SU_M[cut_off:len(SU_M)] - SU_average_pchip[cut_off:len(SU_M)]))/SU_average_pchip[cut_off:len(SU_M)])*100.0)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Scalar: ', x[3], 'Weights: ', weights)
    return residuals


def Residuals_DLP(x, w_n, SL_M, SR_M, SL_Theta, SR_Theta):
    # pre allocation
    SL_2darray = []
    SR_2darray = []
    # Data SL
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    # Data SR
    SR_M = np.array(SR_M)
    SR_M = SR_M[~np.isnan(SR_M)]
    SR_Theta = SR_Theta[~np.isnan(SR_M)]
    # Pchip measurement
    SR_M_Pchip = pchip_interpolate(xi=SR_Theta, yi=SR_M, x=SL_Theta, der=0, axis=0)
    # DLP measurement
    DLP = (SL_M - SR_M_Pchip) / (SL_M + SR_M_Pchip)
    # size distribution and weights calculation
    sizes = np.arange(850.0, 950.0, 5.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        SL_2darray.append(SL)
        SR_2darray.append(SR)
    SL_average = np.average(SL_2darray, axis=0, weights=weights)
    SR_average = np.average(SR_2darray, axis=0, weights=weights)
    SL_average_pchip = pchip_interpolate(xi=theta_mie, yi=SL_average, x=SL_Theta, der=0, axis=0)
    SR_average_pchip = pchip_interpolate(xi=theta_mie, yi=SR_average, x=SL_Theta, der=0, axis=0)
    DLP_average_pchip = (SL_average_pchip - SR_average_pchip) / (SL_average_pchip + SR_average_pchip)
    # compute the sum of the residuals
    residuals = np.sum((((np.absolute(x[3] * SL_M[cut_off:len(SL_M)] - SL_average_pchip[cut_off:len(SL_M)]))/SL_average_pchip[cut_off:len(SL_M)])*100.0) + (((np.absolute(x[4] * SR_M_Pchip[cut_off:len(SR_M_Pchip)] - SR_average_pchip[cut_off:len(SR_M_Pchip)]))/SR_average_pchip[cut_off:len(SR_M_Pchip)])*100.0) + (((np.absolute(x[5] * DLP[cut_off:len(DLP)] - DLP_average_pchip[cut_off:len(DLP)]))/DLP_average_pchip[cut_off:len(DLP)])*100.0))
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Scalar: ', x[3], 'Weights: ', weights)
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
exp_directory = '/home/austen/Desktop/Recent/Retrieval/Calibrated_Data_PSL900nm.txt'
save_directory = '/home/austen/Desktop/Recent/Retrieval/'
su_data = pd.read_csv(exp_directory, sep=',', header=0)
su_measurement = su_data['Exp Smoothed Intensity']
su_theta = su_data['PN to Angle']
#su_measurement = np.array(su_measurement[~np.isnan(su_measurement)])[100:825] / np.sum(np.array(su_measurement[~np.isnan(su_measurement)])[20:825])
su_measurement = np.array(su_measurement[~np.isnan(su_measurement)])#[100:825]
su_theta = np.array(su_theta[~np.isnan(su_measurement)])#[100:825]

'''
# you should already have calibrated theta by the time you get here... we are looking to take all calibration conversion
# to theta info out this script
#slope = 0.2052
#intercept = 0.7795
#theta = np.array([(slope * x) + intercept for x in su_columns])
'''

# the only input parameter other than the calibrated data
wavelength = 663.0

# NLLS result
result_SU = least_squares(Residuals_SU, x0=[903.0, 4.1, 1.598, 0.12], method='trf', args=(wavelength, su_measurement, su_theta), bounds=([850.0, 1.0, 1.00, .001],[950.0, 100.0, 2.00, 1.000]))

# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_SU.x[0])
print('\u03c3: ', result_SU.x[1])
print('m: ', result_SU.x[2])
print('scalar: ', result_SU.x[3])
print('iterations: ', result_SU.nfev)
print('status: ', result_SU.status)

# compute weights using the found distribution parameters
sizes = np.arange(850.0, 950.0, 5.0)
weights = [Gaussian(element, result_SU.x[0], result_SU.x[1]) for element in sizes]

# calculate g with recovered parameters
qext_array, qsca_array, qabs_array, g_array, qpr_array, qback_array, qratio_array = ps.MieQ_withDiameterRange(m=result_SU.x[2], wavelength=wavelength, nMedium=1.0, diameterRange=(850.0, 950.0), nd=20)
g_average = np.average(g_array, axis=1, weights=weights)
print('g(theory): ', g_average)

# recovering NLLS best mie theory match
SU_2d = []
size_result = np.arange(result_SU.x[0] - 3 * result_SU.x[1], result_SU.x[0] + 3 * result_SU.x[1], 1.0)
weights_result = [Gaussian(element, result_SU.x[0], result_SU.x[1]) for element in size_result]
for element in size_result:
    theta_mie, SL, SR, SU = ps.ScatteringFunction(result_SU.x[2], wavelength, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    SU_2d.append(SU)
SU_mie_NLLS = np.average(SU_2d, axis=0, weights=weights_result)
SU_mie_NLLS_pchip = pchip_interpolate(xi=theta_mie, yi=SU_mie_NLLS, x=su_theta, der=0, axis=0)


# find g
# we use this in the event we wanna integrate with quad which isn't built yet
# we also use this to integrate with simpson method which is built
su_measurement_cspline = interp1d(su_theta, su_measurement, kind='cubic', fill_value='extrapolate')
slope = 0.2052
theta_full = np.arange(0.0, 180.0 + slope, slope)
su_extrapolated = su_measurement_cspline(theta_full)
g = Asymmetry_Parameter(su_extrapolated, theta_full)
print('g(measurement): ', g)

# plotting the results
# font sizes for figures
f_title = 24
f_axes = 18
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']
# making figure
f0, ax0 = plt.subplots(figsize=(8, 6))
ax0.semilogy(su_theta, SU_mie_NLLS_pchip, color='red', ls='-', label='NLLS: ' + 'm: ' + str('{:.3f}'.format(result_SU.x[2])) + ' \u03bc: ' + str('{:.3f}'.format(result_SU.x[0])) + '\n\u03c3: ' + str('{:.3f}'.format(result_SU.x[1])) + ' scalar: '+ str('{:.3f}'.format(result_SU.x[3])))
ax0.semilogy(su_theta,  result_SU.x[3] * su_measurement, color='black', ls='-', label='Measurement')
ax0.semilogy(su_theta[cut_off:len(SU_mie_NLLS_pchip)],  result_SU.x[3] * su_measurement[cut_off:len(SU_mie_NLLS_pchip)], color='red', ls='-', lw=7.0, alpha=0.25, label='Used Points')
ax0.set_title('Non-Linear Least Squares Fit', fontsize=f_title)
ax0.set_xlabel('\u03b8', fontsize=f_axes)
ax0.set_ylabel('Intensity', fontsize=f_axes)
ax0.grid(True, which='both')
ax0.legend(loc=1, fontsize=f_axes)
plt.tight_layout()
plt.savefig(save_directory + 'NL_Least_Squares_Result_SU.png', format='png')
plt.savefig(save_directory + 'NL_Least_Squares_Result_SU.pdf', format='pdf')
plt.show()

