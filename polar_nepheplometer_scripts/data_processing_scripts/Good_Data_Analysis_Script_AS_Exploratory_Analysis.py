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
    return x / np.sum(x)


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
save_directory = '/home/austen/Desktop/Recent'
file_directory = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
df = pd.read_csv(file_directory, sep=',', header=0)

# eliminate extra column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#print(df)

# multiindex dataframe
df.set_index(['Sample', 'Size (nm)', 'Polarization'], inplace=True)

#print(df)

# pandas dataframe.xs returns a cross-section of the data, so basically I am filtering out data that isn't PSL, size 900, and pol = SL
#xs_tuple = ('PSL', 900, 'SL')
df_800_SL = df.xs(('AS', 800, 'SL')).reset_index()
df_800_SU = df.xs(('AS', 800, 'SU')).reset_index()
df_800_SR = df.xs(('AS', 800, 'SR')).reset_index()

df_700_SL = df.xs(('AS', 700, 'SL')).reset_index()
df_700_SU = df.xs(('AS', 700, 'SU')).reset_index()
df_700_SR = df.xs(('AS', 700, 'SR')).reset_index()

df_600_SL = df.xs(('AS', 600, 'SL')).reset_index()
df_600_SU = df.xs(('AS', 600, 'SU')).reset_index()
df_600_SR = df.xs(('AS', 600, 'SR')).reset_index()

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
wavelength_red = 663
dp_gaussian = np.arange(1.0, 1000.0, 2.0)
col_transects = np.arange(30, 860, 1)
slope = .2095
intercept = -3.1433
theta_meas = [(slope * i) + intercept for i in col_transects]
ndp_LN_800 = np.array([LogNormal(size=i, mu=800.0, gsd=1.05, N=260) for i in dp_gaussian])
ndp_LN_700 = np.array([LogNormal(size=i, mu=700.0, gsd=1.05, N=433) for i in dp_gaussian])
ndp_LN_600 = np.array([LogNormal(size=i, mu=600.0, gsd=1.05, N=1000) for i in dp_gaussian])

Rad800, SL800, SR800, SU800 = PMS.SF_SD(m=m_AS, wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_800, nMedium=1.0, space='theta', normalization=None)
Rad700, SL700, SR700, SU700 = PMS.SF_SD(m=m_AS, wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_700, nMedium=1.0, space='theta', normalization=None)
Rad600, SL600, SR600, SU600 = PMS.SF_SD(m=m_AS, wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_600, nMedium=1.0, space='theta', normalization=None)

Theta800 = np.array([(i * 180.0)/pi for i in Rad800])
Theta700 = np.array([(i * 180.0)/pi for i in Rad700])
Theta600 = np.array([(i * 180.0)/pi for i in Rad600])

SL800_pchip = pchip_interpolate(xi=Theta800, yi=SL800, x=theta_meas)
SU800_pchip = pchip_interpolate(xi=Theta800, yi=SU800, x=theta_meas)
SR800_pchip = pchip_interpolate(xi=Theta800, yi=SR800, x=theta_meas)

SL700_pchip = pchip_interpolate(xi=Theta700, yi=SL700, x=theta_meas)
SU700_pchip = pchip_interpolate(xi=Theta700, yi=SU700, x=theta_meas)
SR700_pchip = pchip_interpolate(xi=Theta700, yi=SR700, x=theta_meas)

SL600_pchip = pchip_interpolate(xi=Theta600, yi=SL600, x=theta_meas)
SU600_pchip = pchip_interpolate(xi=Theta600, yi=SU600, x=theta_meas)
SR600_pchip = pchip_interpolate(xi=Theta600, yi=SR600, x=theta_meas)


SL800_norm = SL800_pchip / np.sum(SL800_pchip)
SU800_norm = SU800_pchip / np.sum(SU800_pchip)
SR800_norm = SR800_pchip / np.sum(SR800_pchip)

SL700_norm = SL700_pchip / np.sum(SL700_pchip)
SU700_norm = SU700_pchip / np.sum(SU700_pchip)
SR700_norm = SR700_pchip / np.sum(SR700_pchip)

SL600_norm = SL600_pchip / np.sum(SL600_pchip)
SU600_norm = SU600_pchip / np.sum(SU600_pchip)
SR600_norm = SR600_pchip / np.sum(SR600_pchip)


# alright compare each row to mie theory and log the sum of the squared error to the data frame
summed_error_list = []
avg_time_list = []
for index, row in df_800_SL.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SL800_norm)
    # the difference at each angle is small because the measurement and the theory are normalized!
    #error = np.divide(mag_diff, SL900_norm)
    #summed_error = np.sum(error)
    summed_error_list.append(m)


df_800_SL['Summed Error'] = summed_error_list
df_800_SL['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_800_SU.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SU800_norm)
    summed_error_list.append(m)


df_800_SU['Summed Error'] = summed_error_list
df_800_SU['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_800_SR.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SR800_norm)
    summed_error_list.append(m)


df_800_SR['Summed Error'] = summed_error_list
df_800_SR['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_700_SL.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SL700_norm)
    summed_error_list.append(m)


df_700_SL['Summed Error'] = summed_error_list
df_700_SL['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_700_SU.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SU700_norm)
    summed_error_list.append(m)


df_700_SU['Summed Error'] = summed_error_list
df_700_SU['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_700_SR.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SR700_norm)
    summed_error_list.append(m)


df_700_SR['Summed Error'] = summed_error_list
df_700_SR['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_600_SL.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SL600_norm)
    summed_error_list.append(m)


df_600_SL['Summed Error'] = summed_error_list
df_600_SL['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_600_SU.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_norm = pf / np.sum(pf)
    m = M(pf_norm, SU600_norm)
    summed_error_list.append(m)


df_600_SU['Summed Error'] = summed_error_list
df_600_SU['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
for index, row in df_600_SR.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf_pchip_norm = pf / np.sum(pf)
    m = M(pf_pchip_norm, SR600_norm)
    summed_error_list.append(m)


df_600_SR['Summed Error'] = summed_error_list
df_600_SR['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []

# lets start exploring find the minimum of each column
min_error_SL_800 = df_800_SL['Summed Error'].idxmin()
min_error_SU_800 = df_800_SU['Summed Error'].idxmin()
min_error_SR_800 = df_800_SR['Summed Error'].idxmin()

min_error_SL_700 = df_700_SL['Summed Error'].idxmin()
min_error_SU_700 = df_700_SU['Summed Error'].idxmin()
min_error_SR_700 = df_700_SR['Summed Error'].idxmin()

min_error_SL_600 = df_600_SL['Summed Error'].idxmin()
min_error_SU_600 = df_600_SU['Summed Error'].idxmin()
min_error_SR_600 = df_600_SR['Summed Error'].idxmin()

print('-------SL 900 Minimum Error-------')
print('row: ', min_error_SL_800)
print(df_800_SL.loc[min_error_SL_800, 'Exposure Time (s)':'Time'])
print('-------SU 900 Minimum Error-------')
print('row: ', min_error_SU_800)
print(df_800_SU.loc[min_error_SU_800, 'Exposure Time (s)':'Time'])
print('-------SR 900 Minimum Error-------')
print('row: ', min_error_SR_800)
print(df_800_SR.loc[min_error_SR_800, 'Exposure Time (s)':'Time'])

print('-------SL 700 Minimum Error-------')
print('row: ', min_error_SL_700)
print(df_700_SL.loc[min_error_SL_700, 'Exposure Time (s)':'Time'])
print('-------SU 700 Minimum Error-------')
print('row: ', min_error_SU_700)
print(df_700_SU.loc[min_error_SU_700, 'Exposure Time (s)':'Time'])
print('-------SR 700 Minimum Error-------')
print('row: ', min_error_SR_700)
print(df_700_SR.loc[min_error_SR_700, 'Exposure Time (s)':'Time'])

print('-------SL 600 Minimum Error-------')
print('row: ', min_error_SL_600)
print(df_600_SL.loc[min_error_SL_600, 'Exposure Time (s)':'Time'])
print('-------SU 600 Minimum Error-------')
print('row: ', min_error_SU_600)
print(df_600_SU.loc[min_error_SU_600, 'Exposure Time (s)':'Time'])
print('-------SR 600 Minimum Error-------')
print('row: ', min_error_SR_600)
print(df_600_SR.loc[min_error_SR_600, 'Exposure Time (s)':'Time'])

best_SL_800 = Normalization(np.array(df_800_SL.loc[min_error_SL_800, '30':'859']))
best_SU_800 = Normalization(np.array(df_800_SU.loc[min_error_SU_800, '30':'859']))
best_SR_800 = Normalization(np.array(df_800_SR.loc[min_error_SR_800, '30':'859']))

best_SL_700 = Normalization(np.array(df_700_SL.loc[min_error_SL_700, '30':'859']))
best_SU_700 = Normalization(np.array(df_700_SU.loc[min_error_SU_700, '30':'859']))
best_SR_700 = Normalization(np.array(df_700_SR.loc[min_error_SR_700, '30':'859']))

best_SL_600 = Normalization(np.array(df_600_SL.loc[min_error_SL_600, '30':'859']))
best_SU_600 = Normalization(np.array(df_600_SU.loc[min_error_SU_600, '30':'859']))
best_SR_600 = Normalization(np.array(df_600_SR.loc[min_error_SR_600, '30':'859']))

f0 = plt.figure(constrained_layout=True, figsize=(18, 18))
spec = GridSpec(ncols=4, nrows=3, figure=f0)
# size 300nm data
f0_ax00 = f0.add_subplot(spec[0, 0])
f0_ax00.semilogy(theta_meas, best_SL_800, color='red', label='Meas. 800 SL\n' + str(df_800_SL.loc[min_error_SL_800, 'Exposure Time (s)':'Time']))
f0_ax00.semilogy(theta_meas, SL800_norm, color ='black', label='Theory 800 SL')
f0_ax00.set_title('AS 800 SL')
f0_ax00.set_ylabel('Norm. Intensity')
f0_ax00.set_xlabel('Degrees')
f0_ax00.grid(True)
f0_ax00.legend(loc=1)

f0_ax01 = f0.add_subplot(spec[0, 1])
f0_ax01.semilogy(theta_meas, best_SU_800, color='green', label='Meas. 800 SU\n' + str(df_800_SU.loc[min_error_SU_800, 'Exposure Time (s)':'Time']))
f0_ax01.semilogy(theta_meas, SU800_norm, color='black', label='Theory 800 SU')
f0_ax01.set_title('AS 800 SU')
f0_ax01.set_ylabel('Norm. Intensity')
f0_ax01.set_xlabel('Degrees')
f0_ax01.grid(True)
f0_ax01.legend(loc=1)

f0_ax02 = f0.add_subplot(spec[0, 2])
f0_ax02.semilogy(theta_meas, best_SR_800, color='blue', label='Meas. 800 SR\n' + str(df_800_SR.loc[min_error_SR_800, 'Exposure Time (s)':'Time']))
f0_ax02.semilogy(theta_meas, SR800_norm, color='black', label='Theory 800 SR')
f0_ax02.set_title('AS 800 SR')
f0_ax02.set_ylabel('Norm. Intensity')
f0_ax02.set_xlabel('Degrees')
f0_ax02.grid(True)
f0_ax02.legend(loc=1)

f0_ax03 = f0.add_subplot(spec[0, 3])
f0_ax03.plot(dp_gaussian, ndp_LN_800, color='black', label='Theory 800 Dist.')
f0_ax03.set_title('800nm Distribution')
f0_ax03.set_ylabel('Particles/cc')
f0_ax03.set_xlabel('Size (nm)')
f0_ax03.grid(True)
f0_ax03.legend(loc=1)

f0_ax10 = f0.add_subplot(spec[1, 0])
f0_ax10.semilogy(theta_meas, best_SL_700, color='red', label='Meas. 700 SL\n' + str(df_700_SL.loc[min_error_SL_700, 'Exposure Time (s)':'Time']))
f0_ax10.semilogy(theta_meas, SL700_norm, color='black', label='Theory 700 SL')
f0_ax10.set_title('AS 700 SL')
f0_ax10.set_ylabel('Norm. Intensity')
f0_ax10.set_xlabel('Degrees')
f0_ax10.grid(True)
f0_ax10.legend(loc=1)

f0_ax11 = f0.add_subplot(spec[1, 1])
f0_ax11.semilogy(theta_meas, best_SU_700, color='green', label='Meas. 700 SU\n' + str(df_700_SU.loc[min_error_SU_700, 'Exposure Time (s)':'Time']))
f0_ax11.semilogy(theta_meas, SU700_norm, color='black', label='Theory 700 SU')
f0_ax11.set_title('AS 700 SU')
f0_ax11.set_ylabel('Norm. Intensity')
f0_ax11.set_xlabel('Degrees')
f0_ax11.grid(True)
f0_ax11.legend(loc=1)

f0_ax12 = f0.add_subplot(spec[1, 2])
f0_ax12.semilogy(theta_meas, best_SR_700, color='blue', label='Meas. 700 SR\n' + str(df_700_SR.loc[min_error_SR_700, 'Exposure Time (s)':'Time']))
f0_ax12.semilogy(theta_meas, SR700_norm, color='black', label='Theory 700 SR')
f0_ax12.set_title('AS 700 SR')
f0_ax12.set_ylabel('Norm. Intensity')
f0_ax12.set_xlabel('Degrees')
f0_ax12.grid(True)
f0_ax12.legend(loc=1)

f0_ax13 = f0.add_subplot(spec[1, 3])
f0_ax13.plot(dp_gaussian, ndp_LN_700, color='black', label='Theory 700 Dist.')
f0_ax13.set_title('700nm Distribution')
f0_ax13.set_ylabel('Particles/cc')
f0_ax13.set_xlabel('Sizes (nm)')
f0_ax13.grid(True)
f0_ax13.legend(loc=1)

f0_ax20 = f0.add_subplot(spec[2, 0])
f0_ax20.semilogy(theta_meas, best_SL_600, color='red', label='Meas. 600 SL\n' + str(df_600_SL.loc[min_error_SL_600, 'Exposure Time (s)':'Time']))
f0_ax20.semilogy(theta_meas, SL600_norm, color='black', label='Theory 600 SL')
f0_ax20.set_title('AS 600 SL')
f0_ax20.set_ylabel('Norm. Intensity')
f0_ax20.set_xlabel('Degrees')
f0_ax20.grid(True)
f0_ax20.legend(loc=1)

f0_ax21 = f0.add_subplot(spec[2, 1])
f0_ax21.semilogy(theta_meas, best_SU_600, color='green', label='Meas. 600 SU\n' + str(df_600_SU.loc[min_error_SU_600, 'Exposure Time (s)':'Time']))
f0_ax21.semilogy(theta_meas, SU600_norm, color='black', label='Theory 600 SU')
f0_ax21.set_title('AS 600 SU')
f0_ax21.set_ylabel('Norm. Intensity')
f0_ax21.set_xlabel('Degrees')
f0_ax21.grid(True)
f0_ax21.legend(loc=1)

f0_ax22 = f0.add_subplot(spec[2, 2])
f0_ax22.semilogy(theta_meas, best_SR_600, color='blue', label='Meas. 600 SR\n' + str(df_600_SR.loc[min_error_SR_600, 'Exposure Time (s)':'Time']))
f0_ax22.semilogy(theta_meas, SR600_norm, color='black', label='Theory 600 SR')
f0_ax22.set_title('AS 600 SR')
f0_ax22.set_ylabel('Norm. Intensity')
f0_ax22.set_xlabel('Degrees')
f0_ax22.grid(True)
f0_ax22.legend(loc=1)

f0_ax23 = f0.add_subplot(spec[2, 3])
f0_ax23.plot(dp_gaussian, ndp_LN_600, color='black', label='Theory 600 Dist.')
f0_ax23.set_title('600nm Distribution')
f0_ax23.set_ylabel('Particles/cc')
f0_ax23.set_xlabel('Sizes (nm)')
f0_ax23.grid(True)
f0_ax23.legend(loc=1)

plt.savefig(save_directory + '/AS_Best_Agreement.png', format='png')
plt.savefig(save_directory + '/AS_Best_Agreement.pdf', format='pdf')
plt.show()


'''
# retrieval using NLLS
result_800 = least_squares(Residuals_SLSRSU, x0=[650.0, 1.05, 1.500, 500, 0.001], method='trf', args=(wavelength_red, best_SL_800, best_SR_800, best_SU_800, theta_meas, theta_meas, theta_meas, dp_gaussian), bounds=([1.0, 1.000, 1.00, 1, 0.000], [1000.0, 1.500, 2.000, 1000000, 1.000]))
ndp_result_800 = np.array([LogNormal(size=i, mu=result_800.x[0], gsd=result_800.x[1], N=result_800.x[2]) for i in dp_gaussian])
rads_result_800, SL_result_800, SR_result_800, SU_result_800 = PMS.SF_SD(m=complex(result_800.x[3], result_800.x[4]), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_result_800, nMedium=1.0, space='theta', normalization=None)
theta_result_800 = (rads_result_800 * 180.0) / pi
SL_result_800_norm = SL_result_800 / np.sum(SL_result_800)
SR_result_800_norm = SR_result_800 / np.sum(SR_result_800)
SU_result_800_norm = SU_result_800 / np.sum(SU_result_800)

result_700 = least_squares(Residuals_SLSRSU, x0=[650.0, 1.05, 1.500, 500, 0.001], method='trf', args=(wavelength_red, best_SL_700, best_SR_700, best_SU_700, theta_meas, theta_meas, theta_meas, dp_gaussian), bounds=([1.0, 1.000, 1.00, 1, 0.000], [1000.0, 1.500, 2.000, 1000000, 1.000]))
ndp_result_700 = np.array([LogNormal(size=i, mu=result_700.x[0], gsd=result_700.x[1], N=result_700.x[2]) for i in dp_gaussian])
rads_result_700, SL_result_700, SR_result_700, SU_result_700 = PMS.SF_SD(m=complex(result_700.x[3], result_700.x[4]), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_result_700, nMedium=1.0, space='theta', normalization=None)
theta_result_700 = (rads_result_700 * 180.0) / pi
SL_result_700_norm = SL_result_700 / np.sum(SL_result_700)
SR_result_700_norm = SR_result_700 / np.sum(SR_result_700)
SU_result_700_norm = SU_result_700 / np.sum(SU_result_700)

result_600 = least_squares(Residuals_SLSRSU, x0=[650.0, 1.05, 1.500, 500, 0.001], method='trf', args=(wavelength_red, best_SL_600, best_SR_600, best_SU_600, theta_meas, theta_meas, theta_meas, dp_gaussian), bounds=([1.0, 1.000, 1.00, 1, 0.000], [1000.0, 1.500, 2.000, 1000000, 1.000]))
ndp_result_600 = np.array([LogNormal(size=i, mu=result_600.x[0], gsd=result_600.x[1], N=result_600.x[2]) for i in dp_gaussian])
rads_result_600, SL_result_600, SR_result_600, SU_result_600 = PMS.SF_SD(m=complex(result_600.x[3], result_600.x[4]), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_result_600, nMedium=1.0, space='theta', normalization=None)
theta_result_600 = (rads_result_600 * 180.0) / pi
SL_result_600_norm = SL_result_600 / np.sum(SL_result_600)
SR_result_600_norm = SR_result_600 / np.sum(SR_result_600)
SU_result_600_norm = SU_result_600 / np.sum(SU_result_600)


# plot retrieval results
f1 = plt.figure(constrained_layout=True, figsize=(18, 18))
spec = GridSpec(ncols=4, nrows=3, figure=f0)
# size 300nm data
f1_ax00 = f1.add_subplot(spec[0, 0])
f1_ax00.semilogy(theta_meas, best_SL_800, color='red', label='Meas. 800 SL\n' + str(df_800_SL.loc[min_error_SL_800, 'Exposure Time (s)':'Time']))
f1_ax00.semilogy(Theta800, SL800_norm, color ='black', label='Theory 800 SL')
f1_ax00.semilogy(theta_result_800, SL_result_800_norm, color='brown', label='Retrieved 800 SL')
f1_ax00.set_title('AS 800 SL')
f1_ax00.set_ylabel('Norm. Intensity')
f1_ax00.set_xlabel('Degrees')
f1_ax00.grid(True)
f1_ax00.legend(loc=1)

f1_ax01 = f0.add_subplot(spec[0, 1])
f1_ax01.semilogy(theta_meas, best_SU_800, color='green', label='Meas. 800 SU\n' + str(df_800_SU.loc[min_error_SU_800, 'Exposure Time (s)':'Time']))
f1_ax01.semilogy(Theta800, SU800_norm, color='black', label='Theory 800 SU')
f1_ax00.semilogy(theta_result_800, SU_result_800_norm, color='brown', label='Retrieved 800 SU')
f1_ax01.set_title('AS 800 SU')
f1_ax01.set_ylabel('Norm. Intensity')
f1_ax01.set_xlabel('Degrees')
f1_ax01.grid(True)
f1_ax01.legend(loc=1)

f1_ax02 = f0.add_subplot(spec[0, 2])
f1_ax02.semilogy(theta_meas, best_SR_800, color='blue', label='Meas. 800 SR\n' + str(df_800_SR.loc[min_error_SR_800, 'Exposure Time (s)':'Time']))
f1_ax02.semilogy(Theta800, SR800_norm, color='black', label='Theory 800 SR')
f1_ax00.semilogy(theta_result_800, SR_result_800_norm, color='brown', label='Retrieved 800 SR')
f1_ax02.set_title('AS 800 SR')
f1_ax02.set_ylabel('Norm. Intensity')
f1_ax02.set_xlabel('Degrees')
f1_ax02.grid(True)
f1_ax02.legend(loc=1)

f1_ax03 = f0.add_subplot(spec[0, 3])
f1_ax03.plot(dp_gaussian, ndp_LN_800, color='black', label='Theory 800 Dist.')
f1_ax03.plot(dp_gaussian, ndp_result_800, color='brown', label='Retrieved 800 Dist.')
f1_ax03.set_title('800nm Distribution')
f1_ax03.set_ylabel('Particles/cc')
f1_ax03.set_xlabel('Size (nm)')
f1_ax03.grid(True)
f1_ax03.legend(loc=1)

f1_ax10 = f0.add_subplot(spec[1, 0])
f1_ax10.semilogy(theta_meas, best_SL_700, color='red', label='Meas. 700 SL\n' + str(df_700_SL.loc[min_error_SL_700, 'Exposure Time (s)':'Time']))
f1_ax10.semilogy(Theta700, SL700_norm, color='black', label='Theory 700 SL')
f1_ax10.semilogy(theta_result_700, SL_result_700_norm, color='brown', label='Retrieved 700 SL')
f1_ax10.set_title('AS 700 SL')
f1_ax10.set_ylabel('Norm. Intensity')
f1_ax10.set_xlabel('Degrees')
f1_ax10.grid(True)
f1_ax10.legend(loc=1)

f1_ax11 = f0.add_subplot(spec[1, 1])
f1_ax11.semilogy(theta_meas, best_SU_700, color='green', label='Meas. 700 SU\n' + str(df_700_SU.loc[min_error_SU_700, 'Exposure Time (s)':'Time']))
f1_ax11.semilogy(Theta700, SU700_norm, color='black', label='Theory 700 SU')
f1_ax11.semilogy(theta_result_700, SU_result_700_norm, color='brown', label='Retrieved 700 SU')
f1_ax11.set_title('AS 700 SU')
f1_ax11.set_ylabel('Norm. Intensity')
f1_ax11.set_xlabel('Degrees')
f1_ax11.grid(True)
f1_ax11.legend(loc=1)

f1_ax12 = f0.add_subplot(spec[1, 2])
f1_ax12.semilogy(theta_meas, best_SR_700, color='blue', label='Meas. 700 SR\n' + str(df_700_SR.loc[min_error_SR_700, 'Exposure Time (s)':'Time']))
f1_ax12.semilogy(Theta700, SR700_norm, color='black', label='Theory 700 SR')
f1_ax12.semilogy(theta_result_700, SR_result_700_norm, color='brown', label='Retrieved 700 SR')
f1_ax12.set_title('AS 700 SR')
f1_ax12.set_ylabel('Norm. Intensity')
f1_ax12.set_xlabel('Degrees')
f1_ax12.grid(True)
f1_ax12.legend(loc=1)

f1_ax13 = f0.add_subplot(spec[1, 3])
f1_ax13.plot(dp_gaussian, ndp_LN_700, color='black', label='Theory 700 Dist.')
f1_ax13.plot(dp_gaussian, ndp_result_700, color='brown', label='Retrieved 700 Dist.')
f1_ax13.set_title('700nm Distribution')
f1_ax13.set_ylabel('Particles/cc')
f1_ax13.set_xlabel('Sizes (nm)')
f1_ax13.grid(True)
f1_ax13.legend(loc=1)

f1_ax20 = f0.add_subplot(spec[2, 0])
f1_ax20.semilogy(theta_meas, best_SL_600, color='red', label='Meas. 600 SL\n' + str(df_600_SL.loc[min_error_SL_600, 'Exposure Time (s)':'Time']))
f1_ax20.semilogy(Theta600, SL600_norm, color='black', label='Theory 600 SL')
f1_ax20.semilogy(theta_result_600, SL_result_600_norm, color='brown', label='Retrieved 600 SL')
f1_ax20.set_title('AS 600 SL')
f1_ax20.set_ylabel('Norm. Intensity')
f1_ax20.set_xlabel('Degrees')
f1_ax20.grid(True)
f1_ax20.legend(loc=1)

f1_ax21 = f0.add_subplot(spec[2, 1])
f1_ax21.semilogy(theta_meas, best_SU_600, color='green', label='Meas. 600 SU\n' + str(df_600_SU.loc[min_error_SU_600, 'Exposure Time (s)':'Time']))
f1_ax21.semilogy(Theta600, SU600_norm, color='black', label='Theory 600 SU')
f1_ax21.semilogy(theta_result_600, SU_result_600_norm, color='brown', label='Retrieved 600 SU')
f1_ax21.set_title('AS 600 SU')
f1_ax21.set_ylabel('Norm. Intensity')
f1_ax21.set_xlabel('Degrees')
f1_ax21.grid(True)
f1_ax21.legend(loc=1)

f1_ax22 = f0.add_subplot(spec[2, 2])
f1_ax22.semilogy(theta_meas, best_SR_600, color='blue', label='Meas. 600 SR\n' + str(df_600_SR.loc[min_error_SR_600, 'Exposure Time (s)':'Time']))
f1_ax22.semilogy(Theta600, SR600_norm, color='black', label='Theory 600 SR')
f1_ax22.semilogy(theta_result_600, SR_result_600_norm, color='brown', label='Retrieved 600 SR')
f1_ax22.set_title('AS 600 SR')
f1_ax22.set_ylabel('Norm. Intensity')
f1_ax22.set_xlabel('Degrees')
f1_ax22.grid(True)
f1_ax22.legend(loc=1)

f1_ax23 = f0.add_subplot(spec[2, 3])
f1_ax23.plot(dp_gaussian, ndp_LN_600, color='black', label='Theory 600 Dist.')
f1_ax23.plot(dp_gaussian, ndp_result_600, color='brown', label='Retrieved 600 Dist.')
f1_ax23.set_title('600nm Distribution')
f1_ax23.set_ylabel('Particles/cc')
f1_ax23.set_xlabel('Sizes (nm)')
f1_ax23.grid(True)
f1_ax23.legend(loc=1)

plt.savefig(save_directory + '/AS_Best_Retrieval.png', format='png')
plt.savefig(save_directory + '/PSL_Best_Retrieval.pdf', format='pdf')
plt.show()
'''
'''
# how does data that is slightly saturated and not saturated compare to the theory?
dir_5mW_900SL = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-06-23/Analysis/5mW/PSL_900_5_SL_5_120_2020 6 23_14 4 25.txt'
dir_1mW_900SL = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-06-23/Analysis/1mW/PSL_900_5_SL_1_120_2020 6 23_13 50 52.txt'
data_5mW_900SL = pd.read_csv(dir_5mW_900SL, sep=',', header=0)
data_1mW_900SL = pd.read_csv(dir_1mW_900SL, sep=',', header=0)

pf_5mW_900SL_riemann = np.array(data_5mW_900SL['Sample Intensity'])
pf_1mW_900SL_riemann = np.array(data_1mW_900SL['Sample Intensity'])
pf_5mW_900SL_riemann_norm = pf_5mW_900SL_riemann / np.sum(pf_5mW_900SL_riemann)
pf_1mW_900SL_riemann_norm = pf_1mW_900SL_riemann / np.sum(pf_1mW_900SL_riemann)

pf_5mW_900SL_gfit = np.array(data_5mW_900SL['Sample Intensity gfit'])
pf_1mW_900SL_gfit = np.array(data_1mW_900SL['Sample Intensity gfit'])
pf_5mW_900SL_gfit_norm = pf_5mW_900SL_gfit / np.sum(pf_5mW_900SL_gfit)
pf_1mW_900SL_gfit_norm = pf_1mW_900SL_gfit / np.sum(pf_1mW_900SL_gfit)

pf_5mW_900SL_colums = np.array(data_5mW_900SL['Sample Columns'])
pf_1mW_900SL_colums = np.array(data_1mW_900SL['Sample Columns'])
pf_5mW_theta = np.array([(0.2095 * i) + -3.1433 for i in pf_5mW_900SL_colums])
pf_1mW_theta = np.array([(0.2095 * i) + -3.1433 for i in pf_1mW_900SL_colums])

f1, ax1 = plt.subplots(figsize=(12, 18))
ax1.semilogy(pf_1mW_theta, pf_1mW_900SL_riemann_norm, color='lawngreen', label='PSL SL 900nm 1mW Riemann')
ax1.semilogy(pf_5mW_theta, pf_5mW_900SL_riemann_norm, color='orange', label='PSL SL 900nm 5mW Riemann')
ax1.semilogy(pf_1mW_theta, pf_1mW_900SL_gfit_norm, color='fuchsia', label='PSL SL 900nm 1mW gfit')
ax1.semilogy(pf_5mW_theta, pf_5mW_900SL_gfit_norm, color='darkviolet', label='PSL SL 900nm 5mW gfit')
ax1.semilogy(Theta900, SL900_norm, color='black', label='Theory PSL SL 900nm')
ax1.set_title('Comparison of Unsaturated and Saturated SL\n 900nm Measurements to Mie Theory')
ax1.set_ylabel('Norm. Intensity')
ax1.set_xlabel('degrees')
ax1.grid(True)
ax1.legend(loc=1)
plt.savefig(save_directory +'/sat_v_unsat_v_Mie.png', format='png')
plt.savefig(save_directory +'/sat_v_unsat_v_Mie.pdf', format='pdf')
plt.show()
'''
'''
# select columns highlighting measurement conditions, prepare for PCA analysis, how important are the exposure time, laser power, and averaging time, to the explained variance
subset = ['Sample', 'Size (nm)', 'Exposure Time (s)', 'Polarization', 'Laser Power (mW)', 'Number of Averages', 'Summed Error', 'Averaging Time (s)']
df_900_SL_pca = df_900_SL[subset]
df_900_SU_pca = df_900_SU[subset]
df_900_SR_pca = df_900_SR[subset]
df_PSL900 = pd.concat([df_900_SL_pca, df_900_SU_pca, df_900_SR_pca], ignore_index=True)

df_700_SL_pca = df_700_SL[subset]
df_700_SU_pca = df_700_SU[subset]
df_700_SR_pca = df_700_SR[subset]
df_PSL700 = pd.concat([df_700_SL_pca, df_700_SU_pca, df_700_SR_pca], ignore_index=True)

df_600_SL_pca = df_600_SL[subset]
df_600_SU_pca = df_600_SU[subset]
df_600_SR_pca = df_600_SR[subset]
df_PSL600 = pd.concat([df_600_SL_pca, df_600_SU_pca, df_600_SR_pca], ignore_index=True)

# Standardize PCA data, it is affected by scale
# Separating out the features
conditions = ['Exposure Time (s)', 'Averaging Time (s)', 'Laser Power (mW)']
x_PSL900 = df_PSL900.loc[:, conditions].values
# Separating out the target
y_PSL900 = df_PSL900.loc[:,['Polarization']].values
# Standardizing the features
x_PSL900_standardized = StandardScaler().fit_transform(x_PSL900)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x_PSL900_standardized)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, df_PSL900[['Polarization']]], axis=1)
#print(pca.components_)
#print(pca.explained_variance_ratio_)
#print(finalDf)

f2 = plt.figure(figsize=(8, 8))
# 1 x 1 grid, first subplot





#ax0.set_xlabel('Principal Component 1', fontsize=15)
#ax0.set_ylabel('Principal Component 2', fontsize=15)

# colors are gonna represent laser powers
ax0 = f2.add_subplot(221, projection='3d')
indicesToKeep_SL = df_PSL900['Polarization'] == 'SL'
img0 = ax0.scatter(xs=df_PSL900.loc[indicesToKeep_SL, 'Exposure Time (s)'], ys=df_PSL900.loc[indicesToKeep_SL, 'Averaging Time (s)'], zs=df_PSL900.loc[indicesToKeep_SL, 'Summed Error'], c=df_PSL900.loc[indicesToKeep_SL, 'Laser Power (mW)'], marker='.', cmap=plt.hot(), s=50)
ax0.set_title('900nm PSL SL', fontsize=20)
ax0.set_xlabel('Exposure Time (s)')
ax0.set_ylabel('Averaging Time (s)')
ax0.set_zlabel('Sum of the Differences')

# I may try to plot the principal component vectors on the 3d plots...
vector_colors = ['lawngreen', 'aqua', 'fuchsia']
for length, vector, vector_color in zip(pca.explained_variance_, pca.components_, vector_colors):
    v = vector * 2 * (np.sqrt(length)/df_PSL900.shape[0])
    X, Y, Z = df_PSL900.loc[indicesToKeep_SL, ['Exposure Time (s)', 'Averaging Time (s)', 'Summed Error']].mean(axis=0)
    U, V, W = v
    #print(v)
    #print(X, Y, Z)
    ax0.quiver(X, Y, Z, U, V, W, color=vector_color)
    #print(df_PSL900.loc[indicesToKeep_SL, ['Exposure Time (s)', 'Averaging Time (s)', 'Summed Error']])


ax1 = f2.add_subplot(222, projection='3d')
indicesToKeep_SU = df_PSL900['Polarization'] == 'SU'
img1 = ax1.scatter(xs=df_PSL900.loc[indicesToKeep_SU, 'Exposure Time (s)'], ys=df_PSL900.loc[indicesToKeep_SU, 'Averaging Time (s)'], zs=df_PSL900.loc[indicesToKeep_SU, 'Summed Error'], c=df_PSL900.loc[indicesToKeep_SU, 'Laser Power (mW)'], marker='.', cmap=plt.hot(), s=50)
ax1.set_title('900nm PSL SU', fontsize=20)
ax1.set_xlabel('Exposure Time (s)')
ax1.set_ylabel('Averaging Time (s)')
ax1.set_zlabel('Sum of the Differences')
# I may try to plot the principal component vectors on the 3d plots...
start_point = [pca.mean_, pca.mean_, pca.mean_]
vector_colors = ['lawngreen', 'aqua', 'fuchsia']
for length, vector, vector_color in zip(pca.explained_variance_, pca.components_, vector_colors):
    v = vector * 2 * (np.sqrt(length)/df_PSL900.shape[0])
    X, Y, Z = df_PSL900.loc[indicesToKeep_SU, ['Exposure Time (s)', 'Averaging Time (s)', 'Summed Error']].mean(axis=0)
    U, V, W = v
    ax1.quiver(X, Y, Z, U, V, W, color=vector_color)

ax2 = f2.add_subplot(223, projection='3d')
indicesToKeep_SR = df_PSL900['Polarization'] == 'SR'
img2 = ax2.scatter(xs=df_PSL900.loc[indicesToKeep_SR, 'Exposure Time (s)'], ys=df_PSL900.loc[indicesToKeep_SR, 'Averaging Time (s)'], zs=df_PSL900.loc[indicesToKeep_SR, 'Summed Error'], c=df_PSL900.loc[indicesToKeep_SR, 'Laser Power (mW)'], marker='.', cmap=plt.hot(), s=50)
ax2.set_title('900nm PSL SR', fontsize=20)
ax2.set_xlabel('Exposure Time (s)')
ax2.set_ylabel('Averaging Time (s)')
ax2.set_zlabel('Sum of the Differences')
# I may try to plot the principal component vectors on the 3d plots...
start_point = [pca.mean_, pca.mean_, pca.mean_]
vector_colors = ['lawngreen', 'aqua', 'fuchsia']
for length, vector, vector_color in zip(pca.explained_variance_, pca.components_, vector_colors):
    v = vector * 2 * (np.sqrt(length)/df_PSL900.shape[0])
    X, Y, Z = df_PSL900.loc[indicesToKeep_SR, ['Exposure Time (s)', 'Averaging Time (s)', 'Summed Error']].mean(axis=0)
    U, V, W = v
    ax2.quiver(X, Y, Z, U, V, W, color=vector_color)
plt.colorbar(img0, ax=ax0)
plt.colorbar(img1, ax=ax1)
plt.colorbar(img2, ax=ax2)


#for target, color in zip(targets, colors):
#    indicesToKeep = finalDf['Polarization'] == target
#    ax0.scatter(finalDf.loc[indicesToKeep, 'principal component 1'], finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
#ax0.legend(targets)
#ax0.grid()

plt.savefig(save_directory + '/PCA_PSL_900.pdf', format='pdf')
plt.savefig(save_directory + '/PCA_PSL_900.png', format='png')
plt.show()
'''


