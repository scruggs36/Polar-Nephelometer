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
from math import sqrt, log, pi
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate


# Gaussian distribution function
def Gaussian(x, mu, sigma):
   return 1/(sigma * sqrt(2*pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# log normal distribution function, we might want it normalized, check if the equation is right...
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))



# x -->  x[0] = m, x[1] = mu, x[2] = scalar
# Measurement --> SL_M
def Residuals_SL(x, w_n, SL_M, SL_C,  slope, intercept):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_C = SL_C[~np.isnan(SL_M)]
    theta_cal = np.array([slope * element + intercept for element in SL_C])
    sizes = np.arange(800.0, 1000.0, 1.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='t')
        SL_2darray.append(SL)
    SL_average = np.average(SL_2darray, axis=0, weights=weights)
    SL_average_pchip = pchip_interpolate(xi=theta_mie, yi=SL_average, x=theta_cal, der=0, axis=0)
    residuals = SL_average_pchip - (x[3] * SL_M)
    return residuals


def Residuals_SR(x, w_n, SR_M, SR_C,  slope, intercept):
    # pre allocation
    SR_2darray = []
    # Data
    SR_M = np.array(SR_M)
    SR_M = SR_M[~np.isnan(SR_M)]
    SR_C = SR_C[~np.isnan(SR_M)]
    theta_cal = np.array([slope * element + intercept for element in SR_C])
    sizes = np.arange(800.0, 1000.0, 1.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='t')
        SR_2darray.append(SR)
    SR_average = np.average(SR_2darray, axis=0, weights=weights)
    SR_average_pchip = pchip_interpolate(xi=theta_mie, yi=SR_average, x=theta_cal, der=0, axis=0)
    residuals = SR_average_pchip - (x[3] * SR_M)
    return residuals


def Residuals_SU(x, w_n, SU_M, SU_C,  slope, intercept):
    # pre allocation
    SU_2darray = []
    # Data
    SU_M = np.array(SU_M)
    SU_M = SU_M[~np.isnan(SU_M)]
    SU_C = SU_C[~np.isnan(SU_M)]
    theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(850.0, 950.0, 2.0)
    weights = [Gaussian(element, x[0], x[1]) for element in sizes]
    for element in sizes:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='t')
        SU_2darray.append(SU)
    SU_average = np.average(SU_2darray, axis=0, weights=weights)
    SU_average_pchip = pchip_interpolate(xi=theta_mie, yi=SU_average, x=theta_cal, der=0, axis=0)
    residuals = SU_average_pchip - (x[3] * SU_M)
    return residuals

'''
def Linear_Combination_Residuals_SL_SR(x, w_n, M):
    # Data
    M = np.array(M)
    # pre allocate
    SL_2darray = []
    SR_2darray = []
    # size array
    size_array = np.arange(800, 1000, 2)
    # create gaussian distribution values in array
    gaus_dist = np.array([Gaussian(element, x[0], x[1]) for element in size_array])
    # create weighted average phase functions over all sizes in distribution
    for element in size_array:
        theta_mie, SL, SR, SU = ps.ScatteringFunction(x[2], w_n, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='None')
        SL_2darray.append(SL)
        SR_2darray.append(SR)
    SL_average = np.average(SL_2darray, axis=0, weights=gaus_dist)
    SR_average = np.average(SR_2darray, axis=0, weights=gaus_dist)
    residuals = (np.sqrt(x[4]*SL_average**2 + x[5]*SR_average**2) / x[6]) - x[3]*M
    return residuals
'''

'''
the function above remove nans from the data
Explanation:
The inner function, numpy.isnan returns a boolean/logical array which has the value True everywhere that x is not-a-number.
As we want the opposite, we use the logical-not operator, ~ to get an array with Trues everywhere that x is a valid number.
Lastly we use this logical array to index into the original array x, to retrieve just the non-NaN values.
'''

# this is to tes the nonlinear least square under levenberg Marquad Method
exp_directory = '/home/sm3/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-01-21/2020-01-21_Analysis/Measurement/SD_Particle.txt'
save_directory = '/home/sm3/Desktop/'
su_data = pd.read_csv(exp_directory, sep=',', header=0)
su_measurement = su_data['Sample Intensity']
su_columns = su_data['Sample Columns']
su_columns = np.array(su_columns[~np.isnan(su_measurement)])[100:825]
su_measurement = np.array(su_measurement[~np.isnan(su_measurement)])[100:825] / np.sum(np.array(su_measurement[~np.isnan(su_measurement)])[20:825])


slope = 0.2052
intercept = 0.7795
wavelength = 663
theta = np.array([(slope * x) + intercept for x in su_columns])

result_SU = least_squares(Residuals_SU, x0=[903.0, 2.00, 1.598, 0.005], method='lm', args=(wavelength, su_measurement, su_columns, slope, intercept))


print('m: ', result_SU.x[0])
print('\u03bc: ', result_SU.x[1])
#print('\u03c3: ', result_SL.x[1])
print('scalar: ', result_SU.x[2])
print('iterations: ', result_SU.nfev)
print('status: ', result_SU.status)


SU_2d = []
size_result = np.arange(result_SU.x[0] - 3 * result_SU.x[1], result_SU.x[0] + 3 * result_SU.x[1], 1.0)
weights_result = [Gaussian(element, result_SU.x[0], result_SU.x[1]) for element in size_result]
for element in size_result:
    theta_mie, SL, SR, SU = ps.ScatteringFunction(result_SU.x[2], wavelength, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='t')
    SU_2d.append(SL)
SU_avg = np.average(SU_2d, axis=0, weights=weights_result)
SU_average_pchip = pchip_interpolate(xi=theta_mie, yi=SU_avg, x=theta, der=0, axis=0)


f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].semilogy(theta, SU_average_pchip, color='red', ls='-', label='LVMQ: ' + 'm: ' + str('{:.3f}'.format(result_SU.x[2])) + ' \u03bc: ' + str('{:.3f}'.format(result_SU.x[0])) + ' \u03c3: ' + str('{:.3f}'.format(result_SU.x[1])))
ax0[0].semilogy(theta,  result_SU.x[3] * su_measurement, color='black', ls='-', label='Measurement')
ax0[0].set_title('Non-Linear Least Squares Fit')
ax0[0].set_xlabel('\u03b8')
ax0[0].set_ylabel('Intensity')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].plot(theta,  result_SU.fun, color='black', ls='-', label='Measurement')
ax0[1].set_title('Non-Linear Least Squares Residuals')
ax0[1].set_xlabel('\u03b8')
ax0[1].set_ylabel('Intensity')
ax0[1].grid(True)
ax0[1].legend(loc=1)
plt.tight_layout()
plt.savefig(save_directory + 'NL_Least_Squares_Result_SU.png', format='png')
plt.savefig(save_directory + 'NL_Least_Squares_Result_SU.pdf', format='pdf')
plt.show()


'''
result_SLSR = least_squares(Linear_Combination_Residuals_SL_SR, x0=[900.0, 10.0, 1.5, 0.005, 1.0, 1.0, 2.0], method='lm', args=(wavelength, sl_measurement, sl_columns, slope, intercept))
print('\u03bc: ', result_SLSR.x[0])
print('\u03c3: ', result_SLSR.x[1])
print('m: ', result_SLSR.x[2])
print('scalar: ', result_SLSR.x[3])
print('combination scalars: ' [result_SLSR.x[4], result_SLSR.x[5], result_SLSR.x[6]])

f1, ax1 = plt.subplots(1, 2, figsize=(12, 6))
ax1[0].semilogy(theta, result_SLSR, color='red', ls='-', label= 'NLLS LVMQ Fit Parameters: \n'+ '\u03bc: ' + str('{:.3f}'.format(result_SLSR.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SLSR.x[1]))+ 'm: ' + str('{:.3f}'.format(result_SLSR.x[2])) + '\n a: ' + str('{:.3f}'.format(result_SLSR.x[4])) + 'b: '+ str('{:.3f}'.format(result_SLSR.x[5])) + 'c: ' + str('{:.3f}'.format(result_SLSR.x[6])))
ax1[0].semilogy(theta,  result_SLSR.x[3] * sl_measurement, color='black', ls='-', label='Measurement')
ax1[0].set_title('Non-Linear Least Squares Fit')
ax1[0].set_xlabel('\u03b8')
ax1[0].set_ylabel('Intensity')
ax1[0].grid(True)
ax1[0].legend(loc=1)
ax1[1].plot(theta,  result_SLSR.fun, color='black', ls='-', label='Measurement')
ax1[1].set_title('Non-Linear Least Squares Residuals')
ax1[1].set_xlabel('\u03b8')
ax1[1].set_ylabel('Intensity')
ax1[1].grid(True)
ax1[1].legend(loc=1)
plt.tight_layout()
plt.savfig(save_directory + 'NL_Least_Squares_Result_LinComb.png', format='png')
plt.savfig(save_directory + 'NL_Least_Squares_Result_LinComb.pdf', format='pdf')
plt.show()
'''
