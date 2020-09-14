'''
Austen K. Scruggs
08/06/2020
Description: Phase function stitching using PSL data
'''

import pandas as pd
import numpy as np
import PyMieScatt as PMS
import matplotlib.pyplot as plt
from math import sqrt, pi, log
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import pchip_interpolate
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# functions
def Normalization(x):
    return x / np.nansum(x)


# gaussian distribution
def Gaussian_Normalized(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def Gaussian(x, mu, sigma, N):
    return N * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


# just keeping this function aside for when I evaluate the AS data
def LogNormal(size, mu, gsd, N):
    if size ==0.0:
        return 0.0
    else:
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))

def M(meas_pf, mie_pf):
    a = np.sum(np.square(np.subtract(meas_pf, mie_pf)))
    a_0 = np.square(np.subtract(meas_pf, mie_pf))
    b = np.sum(np.abs(np.subtract(meas_pf, mie_pf)))
    b_0 = np.abs(np.subtract(meas_pf, mie_pf))
    c = np.sum(np.divide(np.abs(np.subtract(meas_pf, mie_pf)), mie_pf))
    c_0 = np.divide(np.abs(np.subtract(meas_pf, mie_pf)), mie_pf)
    return c


def Residuals_SL(x, w_n, SL_M, SL_Theta):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(1.0, 1000.0, 2.0)
    counts = [Gaussian(element, x[0], x[1], 400) for element in sizes]
    theta_mie, SL, SR, SU = PMS.SF_SD(m=complex(x[2], x[3]), wavelength=w_n, dp=sizes, ndp=counts, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
    SL_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SL, x=SL_Theta, der=0, axis=0))
    residuals = M(SL_M, SL_pchip_norm)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3])
    return residuals


def Residuals_SLSRSU(x, w_n, SL_M, SR_M, SU_M, SL_Theta, SR_Theta, SU_Theta, bin_edges):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    #SL_M = SL_M[~np.isnan(SL_M)]
    #SL_Theta = SL_Theta[~np.isnan(SL_M)]
    SR_M = np.array(SR_M)
    #SR_M = SL_M[~np.isnan(SR_M)]
    #SR_Theta = SR_Theta[~np.isnan(SR_M)]
    SU_M = np.array(SU_M)
    #SU_M = SL_M[~np.isnan(SU_M)]
    #SU_Theta = SU_Theta[~np.isnan(SU_M)]
    # theta_cal = np.array([slope * element + intercept for element in SU_C])
    bin_counts = [LogNormal(size=i, mu=x[0], gsd=x[1], N=x[2]) for i in bin_edges]
    theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[3], x[4]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_mie = np.array(theta_mie) * (180.0 / pi)
    SL_pchip = pchip_interpolate(xi=theta_mie, yi=SL, x=SL_Theta, der=0, axis=0)
    SR_pchip = pchip_interpolate(xi=theta_mie, yi=SR, x=SR_Theta, der=0, axis=0)
    SU_pchip = pchip_interpolate(xi=theta_mie, yi=SU, x=SU_Theta, der=0, axis=0)
    SL_pchip_norm = SL_pchip / np.sum(SL_pchip)
    SR_pchip_norm = SR_pchip / np.sum(SR_pchip)
    SU_pchip_norm = SU_pchip / np.sum(SU_pchip)
    sl_diff = M(SL_M, SL_pchip_norm)
    sr_diff = M(SR_M, SR_pchip_norm)
    su_diff = M(SU_M, SU_pchip_norm)
    residuals = sl_diff + sr_diff + su_diff
    print('Guess mu: ', x[0], 'Geometric stdev: ', x[1], 'N: ', x[2], 'Guess n: ', x[3], 'Guess k: ', x[4], 'Summed Error: ', residuals)
    return residuals


# import data
save_directory = '/home/austen/Desktop/Recent/PSL_Temporary'
file_directory = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
df = pd.read_csv(file_directory, sep=',', header=0)

# eliminate extra column
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#print(df)

# multiindex dataframe
df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Date'], inplace=True)
#print(df)

# pandas dataframe.xs returns a cross-section of the data, so basically I am filtering out data that isn't PSL, size 900, and pol = SL
#xs_tuple = ('PSL', 900, 'SL')
df_900_SL = df.xs(('PSL', 900, 'SL', '2020-08-08')).reset_index()
df_900_SU = df.xs(('PSL', 900, 'SU', '2020-08-08')).reset_index()
df_900_SR = df.xs(('PSL', 900, 'SR', '2020-08-08')).reset_index()

df_700_SL = df.xs(('PSL', 700, 'SL', '2020-07-03')).reset_index()
df_700_SU = df.xs(('PSL', 700, 'SU', '2020-07-03')).reset_index()
df_700_SR = df.xs(('PSL', 700, 'SR', '2020-07-03')).reset_index()

df_600_SL = df.xs(('PSL', 600, 'SL', '2020-08-08')).reset_index()
df_600_SU = df.xs(('PSL', 600, 'SU', '2020-08-08')).reset_index()
df_600_SR = df.xs(('PSL', 600, 'SR', '2020-08-08')).reset_index()

# view some of the data, use .loc if multiindexed,
#print(df_900_SL.loc[:, 'Exposure Time (s)'], df_900_SL.loc[:, 'Laser Power (mW)'], df_900_SL.loc[:, 'Number of Averages'])
#print(df_900_SU.loc[:, 'Exposure Time (s)'], df_900_SU.loc[:, 'Laser Power (mW)'], df_900_SU.loc[:, 'Number of Averages'])
#print(df_900_SR.loc[:, 'Exposure Time (s)'], df_900_SR.loc[:, 'Laser Power (mW)'], df_900_SR.loc[:, 'Number of Averages'])

# if not just reset the index of the multiindexed data and use as normal
#print(df_900_SL)

# compute Mie theory for PSL

'''
PSLs:
Mean    Mean Uncertainty     Size Dist Sigma
600nm     9nm                     10.0nm
701nm     6nm                     9.0nm
800nm     14nm                    5.6nm
903nm     12nm                    4.1nm
'''

m_AS = 1.525
m_PSL = 1.58514608
wavelength_red = 663
col_transects = np.arange(30, 860, 1)
slope = .2095
intercept = -3.1433
theta_meas = np.array([(slope * i) + intercept for i in col_transects])
dp_gaussian = np.arange(1.0, 1000.0, 2.0)
ndp_LN_900 = np.array([Gaussian(x=i, mu=903.0, sigma=4.1, N=400) for i in dp_gaussian])
ndp_LN_700 = np.array([Gaussian(x=i, mu=701.0, sigma=9.0, N=433) for i in dp_gaussian])
ndp_LN_600 = np.array([Gaussian(x=i, mu=600.0, sigma=10.0, N=1000) for i in dp_gaussian])

Rad900, SL900, SR900, SU900 = PMS.SF_SD(m=m_PSL, wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_900, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
Rad700, SL700, SR700, SU700 = PMS.SF_SD(m=m_PSL, wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_700, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
Rad600, SL600, SR600, SU600 = PMS.SF_SD(m=m_PSL, wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_600, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)

Theta900 = np.array([(i * 180.0)/pi for i in Rad900])
Theta700 = np.array([(i * 180.0)/pi for i in Rad700])
Theta600 = np.array([(i * 180.0)/pi for i in Rad600])

SL900_pchip = pchip_interpolate(xi=Theta900, yi=SL900, x=theta_meas)
SU900_pchip = pchip_interpolate(xi=Theta900, yi=SU900, x=theta_meas)
SR900_pchip = pchip_interpolate(xi=Theta900, yi=SR900, x=theta_meas)

SL700_pchip = pchip_interpolate(xi=Theta700, yi=SL700, x=theta_meas)
SU700_pchip = pchip_interpolate(xi=Theta700, yi=SU700, x=theta_meas)
SR700_pchip = pchip_interpolate(xi=Theta700, yi=SR700, x=theta_meas)

SL600_pchip = pchip_interpolate(xi=Theta600, yi=SL600, x=theta_meas)
SU600_pchip = pchip_interpolate(xi=Theta600, yi=SU600, x=theta_meas)
SR600_pchip = pchip_interpolate(xi=Theta600, yi=SR600, x=theta_meas)

SL900_norm = SL900_pchip / np.sum(SL900_pchip)
SU900_norm = SU900_pchip / np.sum(SU900_pchip)
SR900_norm = SR900_pchip / np.sum(SR900_pchip)

SL700_norm = SL700_pchip / np.sum(SL700_pchip)
SU700_norm = SU700_pchip / np.sum(SU700_pchip)
SR700_norm = SR700_pchip / np.sum(SR700_pchip)

SL600_norm = SL600_pchip / np.sum(SL600_pchip)
SU600_norm = SU600_pchip / np.sum(SU600_pchip)
SR600_norm = SR600_pchip / np.sum(SR600_pchip)



intensity_stitch = 5300
intensity_stitch_array = np.repeat(intensity_stitch, repeats=len(theta_meas))
intensity_stitch_array_norm = Normalization(intensity_stitch_array)
intensity_stitch_norm = intensity_stitch_array_norm[0]
PF1_2dlist = []
PF2_2dlist = []
sep_idx = []
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
for index, row in df_900_SL.iterrows():
    pf = np.array(row.loc['30':'859']).astype(float)
    ax[0].semilogy(theta_meas, pf, ls='-', label=np.array(row.loc['Sample':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf), ls='-', label=np.array(row.loc['Sample':'Time']))
ax[0].semilogy(theta_meas, SL900_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements')
ax[0].set_xlabel('Degrees')
ax[0].set_ylabel('Intensity')
ax[0].grid(True)
ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax[1].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax[1].set_title('Measurements')
ax[1].set_xlabel('Degrees')
ax[1].set_ylabel('Intensity')
ax[1].grid(True)
ax[1].legend(loc=1)
plt.savefig(save_directory + '/measurements_made.pdf', format='pdf')
plt.savefig(save_directory + '/measurements_made.png', format='png')
plt.show()

'''
pf1_2dlabels = []
pf2_2dlabels = []
pf1_2dlist = []
pf2_2dlist = []
for index, row in df_900_SL.iterrows():
    counter = 0
    pf1_details = []
    pf1 = np.zeros(830)
    pf2_details = []
    pf2 =np.zeros(830)
    pf = np.array(row['30':'859']).astype(float)
    for label, content in row['30':'859'].items():
        if content >= intensity_stitch:
            pf1[counter] = content
        if content < intensity_stitch:
            pf2[counter] = content
        counter += 1
    pf1_details.append(np.array(row['Sample':'Time']))
    pf1_2dlist.append(pf1)
    pf2_details.append(np.array(row['Sample':'Time']))
    pf2_2dlist.append(pf2)
'''


pf1_2dlabels = []
pf2_2dlabels = []
pf1_2dlist = []
pf2_2dlist = []
for index, row in df_900_SL.iterrows():
    counter = 0
    pf1_details = []
    pf1 = np.zeros(830)
    pf2_details = []
    pf2 =np.zeros(830)
    pf = np.array(row['30':'859']).astype(float)
    pf_norm = Normalization(pf)
    for element in pf_norm:
        if element >= intensity_stitch_norm:
            pf1[counter] = element
        if element < intensity_stitch_norm:
            pf2[counter] = element
        counter += 1
    pf1_details.append(np.array(row['Sample':'Time']))
    pf1_2dlist.append(pf1)
    pf2_details.append(np.array(row['Sample':'Time']))
    pf2_2dlist.append(pf2)


# selecting for the phase functtions we wanna compare to
T1_label = np.array(df_900_SL.loc[(df_900_SL['Exposure Time (s)'] == 6) & (df_900_SL['Laser Power (mW)'] == 10)].loc[:, 'Sample':'Time'])
T2_label = np.array(df_900_SL.loc[(df_900_SL['Exposure Time (s)'] == 6) & (df_900_SL['Laser Power (mW)'] == 25)].loc[:, 'Sample':'Time'])
#print(T1_label)
T1 = np.array(df_900_SL.loc[(df_900_SL['Exposure Time (s)'] == 6) & (df_900_SL['Laser Power (mW)'] == 10)].loc[:, '30':'859']).flatten()
T2 = np.array(df_900_SL.loc[(df_900_SL['Exposure Time (s)'] == 6) & (df_900_SL['Laser Power (mW)'] == 25)].loc[:, '30':'859']).flatten()
#print(T1)
T1 = Normalization(T1)
T2 = Normalization(T2)
pf_combos_norm_list = []
m_list = []
label_list = []
theta_list = []
f1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(30, 18))
for counter, element in enumerate(pf1_2dlist):
    for counter2, element2 in enumerate(pf2_2dlist):
        drop_nan_idx = []
        pf_combos = np.add(np.array(element), np.array(pf2_2dlist[counter2]))
        pf_combos[pf_combos == 0] = np.nan
        nan_bool = np.isnan(pf_combos)
        for counter3, element3 in enumerate(nan_bool):
            if element3 == True:
                drop_nan_idx.append(counter3)
        pf_combos_dropnan = np.delete(pf_combos, drop_nan_idx)
        theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx)
        SL900_norm_dropnan = np.delete(SL900_norm, drop_nan_idx)
        m = M(pf_combos_dropnan, SL900_norm_dropnan)
        label_string = str(np.array(df_900_SL.loc[counter,'Sample':'Time'])) + ' &\n' + str(np.array(df_900_SL.loc[counter2, 'Sample':'Time']))
        pf_combos_norm_list.append(pf_combos_dropnan)
        m_list.append(m)
        theta_list.append(theta_meas_dropnan)
        label_list.append(label_string)
        #col_combos = np.add(np.array(pf1_2dlabels[counter]), np.array(pf2_2dlabels[counter2]))
        #theta_combos = np.array([(slope * i) + intercept for i in col_combos])
        ax1[0].semilogy(theta_meas_dropnan, pf_combos_dropnan, ls='-', label=label_string)
m_list = np.array(m_list)
#print(m_list)
m_min_val = np.amin(m_list)
m_min_idx = np.argmin(m_list)
ax1[0].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax1[0].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax1[0].set_title('Normalized Measurements')
ax1[0].set_xlabel('Degrees')
ax1[0].set_ylabel('Intensity')
ax1[0].grid(True)
ax1[1].semilogy(theta_list[m_min_idx], pf_combos_norm_list[m_min_idx], ls='-', color='red', label=label_list[m_min_idx])
ax1[1].semilogy(theta_meas, T1, ls='-', color='green', label=T1_label)
ax1[1].semilogy(theta_meas, T2, ls='-', color='blue', label=T2_label)
ax1[1].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax1[1].set_title('Best Normalized Measurement')
ax1[1].set_xlabel('Degrees')
ax1[1].set_ylabel('Intensity')
ax1[1].grid(True)
ax1[1].legend(loc=1)
plt.savefig(save_directory + '/all_stitches.pdf', format='pdf')
plt.savefig(save_directory + '/all_stitches.png', format='png')
plt.show()


# NLLS result
#result_SL900_T1 = least_squares(Residuals_SL, x0=[903.0, 4.1, 1.598, 0.000], method='trf', args=(wavelength_red, T1, theta_meas), bounds=([850.0, 1.0, 1.30, 0.000],[950.0, 100.0, 1.70, 0.100]))

# minimum values
'''
print('-------Solution Reached!-------')
print('\u03bc: ', result_SL900_T1.x[0])
print('\u03c3: ', result_SL900_T1.x[1])
print('m: ', result_SL900_T1.x[2])
print('k: ', result_SL900_T1.x[3])
print('iterations: ', result_SL900_T1.nfev)
print('status: ', result_SL900_T1.status)
'''
# NLLS result
#result_SL900_T2 = least_squares(Residuals_SL, x0=[903.0, 4.1, 1.598, 0.000], method='trf', args=(wavelength_red, T2, theta_meas), bounds=([850.0, 1.0, 1.30, 0.000],[950.0, 100.0, 1.70, 0.100]))

# minimum values
'''
print('-------Solution Reached!-------')
print('\u03bc: ', result_SL900_T2.x[0])
print('\u03c3: ', result_SL900_T2.x[1])
print('m: ', result_SL900_T2.x[2])
print('k: ', result_SL900_T2.x[3])
print('iterations: ', result_SL900_T2.nfev)
print('status: ', result_SL900_T2.status)
'''
# NLLS result
result_SL900_stitch = least_squares(Residuals_SL, x0=[903, 4.1, 1.50, 0.0], method='trf', args=(wavelength_red, pf_combos_norm_list[m_min_idx], theta_list[m_min_idx]), bounds=([850, 1.0, 1.30, 0.0],[950, 100.0, 1.70, 0.1]))

# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_SL900_stitch.x[0])
print('\u03c3: ', result_SL900_stitch.x[1])
print('m: ', result_SL900_stitch.x[2])
print('k: ', result_SL900_stitch.x[3])
print('iterations: ', result_SL900_stitch.nfev)
print('status: ', result_SL900_stitch.status)

# get distributions
sizes = np.arange(850.0, 950.0, 2.0)
#distribution_T1 = np.array([Gaussian(x=i, mu=result_SL900_T1.x[0], sigma=result_SL900_T1.x[1], N=400) for i in sizes])
#distribution_T2 = np.array([Gaussian(x=i, mu=result_SL900_T2.x[0], sigma=result_SL900_T2.x[1], N=400) for i in sizes])
distribution_theory = np.array([Gaussian(x=i, mu=903.0, sigma=4.1, N=400) for i in sizes])
distribution_stitch = np.array([Gaussian(x=i, mu=result_SL900_stitch.x[0], sigma=result_SL900_stitch.x[1], N=400) for i in sizes])
# calculate best agreement
#m_T1 = complex(result_SL900_T1.x[2], result_SL900_T1.x[3])
#m_T2 = complex(result_SL900_T2.x[2], result_SL900_T2.x[3])
m_TS = complex(result_SL900_stitch.x[2], result_SL900_stitch.x[3])
#rad_T1, SL_T1, SR_T1, SU_T1 = PMS.SF_SD(m=m_T1, wavelength=wavelength_red, dp=sizes, ndp=distribution_T1, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
#rad_T2, SL_T2, SR_T2, SU_T2 = PMS.SF_SD(m=m_T2, wavelength=wavelength_red, dp=sizes, ndp=distribution_T2, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
rad_TS, SL_TS, SR_TS, SU_TS = PMS.SF_SD(m=m_TS, wavelength=wavelength_red, dp=sizes, ndp=distribution_stitch, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
# rad to theta
#theta_T1 = np.array([(i * 180.0)/pi for i in rad_T1])
#theta_T2 = np.array([(i * 180.0)/pi for i in rad_T2])
theta_TS = np.array([(i * 180.0)/pi for i in rad_TS])
# pchip
#SL_T1_pchip = pchip_interpolate(xi=theta_T1, yi=SL_T1, x=theta_meas)
#SL_T2_pchip = pchip_interpolate(xi=theta_T2, yi=SL_T2, x=theta_meas)
SL_TS_pchip = pchip_interpolate(xi=theta_TS, yi=SL_TS, x=theta_meas)
# normalize
#SL_T1_pchip_norm = Normalization(SL_T1_pchip)
#SL_T2_pchip_norm = Normalization(SL_T2_pchip)
SL_TS_pchip_norm = Normalization(SL_TS_pchip)
# plot it all up
f2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(21, 14))
ax2[0, 0].semilogy(theta_list[m_min_idx], pf_combos_norm_list[m_min_idx], ls='-', color='red', label=label_list[m_min_idx])
#ax2[0, 0].semilogy(theta_meas, T1, ls='-', color='green', label=T1_label)
#ax2[0, 0].semilogy(theta_meas, T2, ls='-', color='blue', label=T2_label)
ax2[0, 0].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
#ax2[0, 0].semilogy(theta_meas, SL_T1_pchip_norm, color='lawngreen', ls='-', linewidth=4, label='Theoretical Match: ' + str(T1_label))
#ax2[0, 0].semilogy(theta_meas, SL_T2_pchip_norm, color='cyan', ls='-', linewidth=4, label='Theoretical Match: ' + str(T2_label))
ax2[0, 0].semilogy(theta_meas, SL_TS_pchip_norm, color='coral', ls='-', linewidth=4, label='Theoretical Match: ' + str(label_list[m_min_idx]))
ax2[0, 0].set_title('Normalized Measurements, Stitched Measurement, and Theory')
ax2[0, 0].set_xlabel('Degrees')
ax2[0, 0].set_ylabel('Intensity')
ax2[0, 0].grid(True)
ax2[0, 0].legend(loc=1)
ax2[0, 1].plot(sizes, distribution_stitch, ls='-', color='red', label=label_list[m_min_idx] + 'Dist.')
#ax2[0, 1].plot(sizes, distribution_T1, ls='-', color='green', label=str(T1_label) + 'Dist.')
#ax2[0, 1].plot(sizes, distribution_T2, ls='-', color='blue', label=str(T2_label) + 'Dist.')
ax2[0, 1].plot(sizes, distribution_theory, ls='-', color='black', label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[0, 1].set_title('Retrieved Distributions')
ax2[0, 1].set_xlabel('Sizes (nm)')
ax2[0, 1].set_ylabel('Counts (p/cc)')
ax2[0, 1].grid(True)
ax2[0, 1].legend(loc=1)
bar_x_label=['TS', 'T1', 'T2', 'T0']
ax2[1, 0].bar(bar_x_label[0], result_SL900_stitch.x[2], color='red', label='TS: ' + label_list[m_min_idx])
#ax2[1, 0].bar(bar_x_label[1], result_SL900_T1.x[2], color='green', label='T1: ' + str(T1_label))
#ax2[1, 0].bar(bar_x_label[2], result_SL900_T2.x[2], color='blue', label='T2: ' + str(T2_label))
ax2[1, 0].bar(bar_x_label[3], m_PSL, color='black', label='T0: Specified Thermo Fisher Scientific PSL n')
ax2[1, 0].text(x=0, y=result_SL900_stitch.x[2] + (result_SL900_stitch.x[2] * 0.01), s=str(result_SL900_stitch.x[2]), color='red')
#ax2[1, 0].text(x=1, y=result_SL900_T1.x[2] + (result_SL900_T1.x[2] * 0.01), s=str(result_SL900_T1.x[2]), color='green')
#ax2[1, 0].text(x=2, y=result_SL900_T2.x[2] + (result_SL900_T2.x[2] * 0.01), s=str(result_SL900_T2.x[2]), color='blue')
ax2[1, 0].text(x=3, y=m_PSL + (m_PSL * 0.01), s=str(m_PSL), color='black')
ax2[1, 0].set_title('Retrieved Real Refractive Index')
ax2[1, 0].set_xlabel('Phase Functions')
ax2[1, 0].set_ylabel('n')
ax2[1, 0].grid(True)
ax2[1, 0].legend(loc=4)
ax2[1, 1].bar(bar_x_label[0], result_SL900_stitch.x[3], color='red', label='TS: ' + label_list[m_min_idx])
#ax2[1, 1].bar(bar_x_label[1], result_SL900_T1.x[3], color='green', label='T1: ' + str(T1_label))
#ax2[1, 1].bar(bar_x_label[2], result_SL900_T2.x[3], color='blue', label='T2: ' + str(T2_label))
ax2[1, 1].bar(bar_x_label[3], 0.0, color='black', label='T0: Specified Thermo Fisher Scientific PSL k')
ax2[1, 1].text(x=0, y=result_SL900_stitch.x[3] + (result_SL900_stitch.x[3] * 0.01), s=str(result_SL900_stitch.x[3]), color='red')
#ax2[1, 1].text(x=1, y=result_SL900_T1.x[3] + (result_SL900_T1.x[3] * 0.01), s=str(result_SL900_T1.x[3]), color='green')
#ax2[1, 1].text(x=2, y=result_SL900_T2.x[3] + (result_SL900_T2.x[3] * 0.01), s=str(result_SL900_T2.x[3]), color='blue')
ax2[1, 1].text(x=3, y=m_PSL + (m_PSL * 0.01), s=str(m_PSL), color='black')
ax2[1, 1].set_ylim([1E-11, 1E-9])
ax2[1, 1].set_yscale('log')
ax2[1, 1].set_title('Retrieved Imaginary Refactive Index')
ax2[1, 1].set_xlabel('Phase Functions')
ax2[1, 1].set_ylabel('k')
ax2[1, 1].grid(True)
ax2[1, 1].legend(loc=1)
plt.savefig(save_directory + '/best_&_retrieval.pdf', format='pdf')
plt.savefig(save_directory + '/best_&_retrieval.png', format='png')
plt.show()