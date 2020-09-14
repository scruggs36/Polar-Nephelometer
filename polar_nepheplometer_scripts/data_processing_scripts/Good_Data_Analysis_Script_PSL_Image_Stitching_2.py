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
    return a

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
print(df)

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
ndp_LN_900 = np.array([Gaussian(x=i, mu=903.0, sigma=4.1, N=260) for i in dp_gaussian])
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



idx_a = 0
#idx_b = 224
idx_b = 299
#idx_c = 225
idx_c = 300
idx_d = 829
# alright compare each row to mie theory and log the sum of the squared error to the data frame
summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_900_SL.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    #pf_pchip = pchip_interpolate(xi=theta_meas, yi=pf, x=Theta900)
    pf_norm = pf / np.nansum(pf)
    print(pf_norm)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SL900_norm[idx_a:idx_b])
    #print(pf_norm1)
    m2 = M(pf_norm2, SL900_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_900SL_df = pd.DataFrame(np.array(m_2dlist))
#res_900SL_df.columns = col_transects.astype(str)
#print(res_900SL_df)
df_900_SL['Summed Error 1'] = m1_list
df_900_SL['Summed Error 2'] = m2_list
df_900_SL['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_900_SU.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SU900_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SU900_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_900SU_df = pd.DataFrame(np.array(m_2dlist))
#res_900SU_df.columns = col_transects.astype(str)
df_900_SU['Summed Error 1'] = m1_list
df_900_SU['Summed Error 2'] = m2_list
df_900_SU['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_900_SR.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SR900_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SR900_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_900SR_df = pd.DataFrame(np.array(m_2dlist))
#res_900SR_df.columns = col_transects.astype(str)
df_900_SR['Summed Error 1'] = m1_list
df_900_SR['Summed Error 2'] = m2_list
df_900_SR['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_700_SL.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SL700_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SL700_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_700SL_df = pd.DataFrame(np.array(m_2dlist))
#res_700SL_df.columns = col_transects.astype(str)
df_700_SL['Summed Error 1'] = m1_list
df_700_SL['Summed Error 2'] = m2_list
df_700_SL['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_700_SU.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SU700_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SU700_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_700SU_df = pd.DataFrame(np.array(m_2dlist))
#res_700SU_df.columns = col_transects.astype(str)
df_700_SU['Summed Error 1'] = m1_list
df_700_SU['Summed Error 2'] = m2_list
df_700_SU['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_700_SR.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SR700_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SR700_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_700SR_df = pd.DataFrame(np.array(m_2dlist))
#res_700SR_df.columns = col_transects.astype(str)
df_700_SR['Summed Error 1'] = m1_list
df_700_SR['Summed Error 2'] = m2_list
df_700_SR['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_600_SL.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SL600_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SL600_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_600SL_df = pd.DataFrame(np.array(m_2dlist))
#res_600SL_df.columns = col_transects.astype(str)
df_600_SL['Summed Error 1'] = m1_list
df_600_SL['Summed Error 2'] = m2_list
df_600_SL['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_600_SU.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SU600_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SU600_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_600SU_df = pd.DataFrame(np.array(m_2dlist))
#res_600SU_df.columns = col_transects.astype(str)
df_600_SU['Summed Error 1'] = m1_list
df_600_SU['Summed Error 2'] = m2_list
df_600_SU['Averaging Time (s)'] = avg_time_list


summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []
for index, row in df_600_SR.iterrows():
    et = float(row.loc['Exposure Time (s)'])
    num_avg = float(row.loc['Number of Averages'])
    avg_time = et * num_avg
    avg_time_list.append(avg_time)
    # for the purpose of troubleshooting things, we are just limiting ourselves to the first entry of the df
    pf = np.array(row.loc['30':'859']).astype(float)
    pf[np.isnan(pf)] = 0
    pf_norm = pf / np.nansum(pf)
    # split pfs into two parts
    pf_norm1 = pf_norm[idx_a:idx_b]
    pf_norm2 = pf_norm[idx_c:idx_d]
    m1 = M(pf_norm1, SR600_norm[idx_a:idx_b])
    m2 = M(pf_norm2, SR600_norm[idx_c:idx_d])
    m1_list.append(m1)
    m2_list.append(m2)
    #m_2dlist.append(m)


#res_600SR_df = pd.DataFrame(np.array(m_2dlist))
#res_600SR_df.columns = col_transects.astype(str)
df_600_SR['Summed Error 1'] = m1_list
df_600_SR['Summed Error 2'] = m2_list
df_600_SR['Averaging Time (s)'] = avg_time_list
summed_error_list = []
avg_time_list = []
m1_list = []
m2_list = []


# select best Pf halves and join them
best_900SL_1 = df_900_SL['Summed Error 1'].idxmin()
#print(df_900_SL['Summed Error 1'])
best_900SL_2 = df_900_SL['Summed Error 2'].idxmin()
#print(best_900SL_2)
best_900SL = np.concatenate((np.array(df_900_SL.loc[best_900SL_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_900_SL.loc[best_900SL_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_900SU_1 = df_900_SU['Summed Error 1'].idxmin()
best_900SU_2 = df_900_SU['Summed Error 2'].idxmin()
best_900SU = np.concatenate((np.array(df_900_SU.loc[best_900SU_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_900_SU.loc[best_900SU_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_900SR_1 = df_900_SR['Summed Error 1'].idxmin()
best_900SR_2 = df_900_SR['Summed Error 2'].idxmin()
best_900SR = np.concatenate((np.array(df_900_SR.loc[best_900SR_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_900_SR.loc[best_900SR_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_700SL_1 = df_700_SL['Summed Error 1'].idxmin()
best_700SL_2 = df_700_SL['Summed Error 2'].idxmin()
best_700SL = np.concatenate((np.array(df_700_SL.loc[best_700SL_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_700_SL.loc[best_700SL_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_700SU_1 = df_700_SU['Summed Error 1'].idxmin()
best_700SU_2 = df_700_SU['Summed Error 2'].idxmin()
best_700SU = np.concatenate((np.array(df_700_SU.loc[best_700SU_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_700_SU.loc[best_700SU_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_700SR_1 = df_700_SR['Summed Error 1'].idxmin()
best_700SR_2 = df_700_SR['Summed Error 2'].idxmin()
best_700SR = np.concatenate((np.array(df_700_SR.loc[best_700SR_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_700_SR.loc[best_700SR_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_600SL_1 = df_600_SL['Summed Error 1'].idxmin()
best_600SL_2 = df_600_SL['Summed Error 2'].idxmin()
best_600SL = np.concatenate((np.array(df_600_SL.loc[best_600SL_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_600_SL.loc[best_600SL_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_600SU_1 = df_600_SL['Summed Error 1'].idxmin()
best_600SU_2 = df_600_SL['Summed Error 2'].idxmin()
best_600SU = np.concatenate((np.array(df_600_SU.loc[best_600SU_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_600_SU.loc[best_600SU_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

best_600SR_1 = df_600_SL['Summed Error 1'].idxmin()
best_600SR_2 = df_600_SL['Summed Error 2'].idxmin()
best_600SR = np.concatenate((np.array(df_600_SR.loc[best_600SR_1, str(idx_a + 30):str(idx_b + 30)]), np.array(df_600_SR.loc[best_600SR_2, str(idx_c + 30):str(idx_d + 30)])), axis=None)

# Normalization
best_900SL_norm = Normalization(np.array(best_900SL))
best_900SU_norm = Normalization(np.array(best_900SU))
best_900SR_norm = Normalization(np.array(best_900SR))

best_700SL_norm = Normalization(np.array(best_700SL))
best_700SU_norm = Normalization(np.array(best_700SU))
best_700SR_norm = Normalization(np.array(best_700SR))

best_600SL_norm = Normalization(np.array(best_600SL))
best_600SU_norm = Normalization(np.array(best_600SU))
best_600SR_norm = Normalization(np.array(best_600SR))


# 1. build a library and a tool to automatically background subtract data in a directory
# 2. analyze all the data we have back logged
# 3. plot image stitched phase function, labeling all the points as different colors/symbols that come from different
# measurements, compare them to mie theory and previous results, plot histogram showing all the different measurements
# used in the stitching of the phasefunction in figure f0

f0 = plt.figure(constrained_layout=True, figsize=(18, 18))
spec = GridSpec(ncols=4, nrows=3, figure=f0)
# size 300nm data
f0_ax00 = f0.add_subplot(spec[0, 0])
f0_ax00.semilogy(theta_meas, best_900SL_norm, color='red', label='Meas. 900 SL stitched: ' + df_900_SL.loc[best_900SL_1, 'Date'] + ' ' + df_900_SL.loc[best_900SL_1, 'Time'] + ' &\n' + df_900_SL.loc[best_900SL_2, 'Date'] + ' ' + df_900_SL.loc[best_900SL_2, 'Time'])
f0_ax00.semilogy(theta_meas, SL900_norm, color ='black', label='Theory 900 SL')
f0_ax00.axvline(theta_meas[idx_b], color='black')
f0_ax00.set_title('PSL 900 SL')
f0_ax00.set_ylabel('Norm. Intensity')
f0_ax00.set_xlabel('Degrees')
f0_ax00.grid(True)
f0_ax00.legend(loc=1)

f0_ax01 = f0.add_subplot(spec[0, 1])
f0_ax01.semilogy(theta_meas, best_900SU_norm, color='green', label='Meas. 900 SU stitched: ' + df_900_SU.loc[best_900SU_1, 'Date'] + ' ' + df_900_SU.loc[best_900SU_1, 'Time'] + ' &\n' + df_900_SL.loc[best_900SU_2, 'Date'] + ' ' + df_900_SL.loc[best_900SU_2, 'Time'])
f0_ax01.semilogy(theta_meas, SU900_norm, color='black', label='Theory 900 SU')
f0_ax01.axvline(theta_meas[idx_b], color='black')
f0_ax01.set_title('PSL 900 SU')
f0_ax01.set_ylabel('Norm. Intensity')
f0_ax01.set_xlabel('Degrees')
f0_ax01.grid(True)
f0_ax01.legend(loc=1)

f0_ax02 = f0.add_subplot(spec[0, 2])
f0_ax02.semilogy(theta_meas, best_900SR_norm, color='blue', label='Meas. 900 SR stitched: ' + df_900_SR.loc[best_900SR_1, 'Date'] + ' ' + df_900_SR.loc[best_900SR_1, 'Time'] + ' &\n' + df_900_SR.loc[best_900SR_2, 'Date'] + ' ' + df_900_SR.loc[best_900SR_2, 'Time'])
f0_ax02.semilogy(theta_meas, SR900_norm, color='black', label='Theory 900 SR')
f0_ax02.axvline(theta_meas[idx_b], color='black')
f0_ax02.set_title('PSL 900 SR')
f0_ax02.set_ylabel('Norm. Intensity')
f0_ax02.set_xlabel('Degrees')
f0_ax02.grid(True)
f0_ax02.legend(loc=1)

f0_ax03 = f0.add_subplot(spec[0, 3])
f0_ax03.plot(dp_gaussian, ndp_LN_900, color='black', label='Theory 900 Dist.')
f0_ax03.set_title('900nm Distribution')
f0_ax03.set_ylabel('Particles/cc')
f0_ax03.set_xlabel('Size (nm)')
f0_ax03.grid(True)
f0_ax03.legend(loc=1)

f0_ax10 = f0.add_subplot(spec[1, 0])
f0_ax10.semilogy(theta_meas, best_700SL_norm, color='red', label='Meas. 700 SL stitched: ' + df_700_SL.loc[best_700SL_1, 'Date'] + ' ' + df_700_SL.loc[best_700SL_1, 'Time'] + ' &\n' + df_700_SL.loc[best_700SL_2, 'Date'] + ' ' + df_700_SL.loc[best_700SL_2, 'Time'])
f0_ax10.semilogy(theta_meas, SL700_norm, color='black', label='Theory 700 SL')
f0_ax10.axvline(theta_meas[idx_b], color='black')
f0_ax10.set_title('PSL 700 SL')
f0_ax10.set_ylabel('Norm. Intensity')
f0_ax10.set_xlabel('Degrees')
f0_ax10.grid(True)
f0_ax10.legend(loc=1)

f0_ax11 = f0.add_subplot(spec[1, 1])
f0_ax11.semilogy(theta_meas, best_700SU_norm, color='green', label='Meas. 700 SU stitched: ' + df_700_SU.loc[best_700SU_1, 'Date'] + ' ' + df_700_SU.loc[best_700SU_1, 'Time'] + ' &\n' + df_700_SU.loc[best_700SU_2, 'Date'] + ' ' + df_700_SU.loc[best_700SU_2, 'Time'])
f0_ax11.semilogy(theta_meas, SU700_norm, color='black', label='Theory 700 SU')
f0_ax11.axvline(theta_meas[idx_b], color='black')
f0_ax11.set_title('PSL 700 SU')
f0_ax11.set_ylabel('Norm. Intensity')
f0_ax11.set_xlabel('Degrees')
f0_ax11.grid(True)
f0_ax11.legend(loc=1)

f0_ax12 = f0.add_subplot(spec[1, 2])
f0_ax12.semilogy(theta_meas, best_700SR_norm, color='blue', label='Meas. 700 SR stitched: ' + df_700_SR.loc[best_700SR_1, 'Date'] + ' ' + df_700_SR.loc[best_700SR_1, 'Time'] + ' &\n' + df_700_SR.loc[best_700SR_2, 'Date'] + ' ' + df_700_SR.loc[best_700SR_2, 'Time'])
f0_ax12.semilogy(theta_meas, SR700_norm, color='black', label='Theory 700 SR')
f0_ax12.axvline(theta_meas[idx_b], color='black')
f0_ax12.set_title('PSL 700 SR')
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
f0_ax20.semilogy(theta_meas, best_600SL_norm, color='red', label='Meas. 600 SL stitched: ' + df_600_SL.loc[best_600SL_1, 'Date'] + ' ' + df_600_SL.loc[best_600SL_1, 'Time'] + ' &\n' + df_600_SL.loc[best_600SL_2, 'Date'] + ' ' + df_600_SL.loc[best_600SL_2, 'Time'])
f0_ax20.semilogy(theta_meas, SL600_norm, color='black', label='Theory 600 SL')
f0_ax20.axvline(theta_meas[idx_b], color='black')
f0_ax20.set_title('PSL 600 SL')
f0_ax20.set_ylabel('Norm. Intensity')
f0_ax20.set_xlabel('Degrees')
f0_ax20.grid(True)
f0_ax20.legend(loc=1)

f0_ax21 = f0.add_subplot(spec[2, 1])
f0_ax21.semilogy(theta_meas, best_600SU_norm, color='green', label='Meas. 600 SU stitched: ' + df_600_SU.loc[best_600SU_1, 'Date'] + ' ' + df_600_SU.loc[best_600SU_1, 'Time'] + ' &\n' + df_600_SU.loc[best_600SU_2, 'Date'] + ' ' + df_600_SU.loc[best_600SU_2, 'Time'])
f0_ax21.semilogy(theta_meas, SU600_norm, color='black', label='Theory 600 SU')
f0_ax21.axvline(theta_meas[idx_b], color='black')
f0_ax21.set_title('PSL 600 SU')
f0_ax21.set_ylabel('Norm. Intensity')
f0_ax21.set_xlabel('Degrees')
f0_ax21.grid(True)
f0_ax21.legend(loc=1)

f0_ax22 = f0.add_subplot(spec[2, 2])
f0_ax22.semilogy(theta_meas, best_600SR_norm, color='blue', label='Meas. 600 SR stitched: ' + df_600_SR.loc[best_600SR_1, 'Date'] + ' ' + df_600_SR.loc[best_600SR_1, 'Time'] + ' &\n' + df_600_SR.loc[best_600SR_2, 'Date'] + ' ' + df_600_SR.loc[best_600SR_2, 'Time'])
f0_ax22.semilogy(theta_meas, SR600_norm, color='black', label='Theory 600 SR')
f0_ax22.axvline(theta_meas[idx_b], color='black')
f0_ax22.set_title('PSL 600 SR')
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


plt.savefig(save_directory + '/PSL_Best_Agreement_Stitched_2.png', format='png')
plt.savefig(save_directory + '/PSL_Best_Agreement_Stitched_2.pdf', format='pdf')
plt.show()


