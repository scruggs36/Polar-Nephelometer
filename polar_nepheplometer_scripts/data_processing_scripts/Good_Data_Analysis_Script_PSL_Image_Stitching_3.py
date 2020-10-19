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
    a = np.sum((meas_pf - mie_pf)**2)
    a_0 = (meas_pf - mie_pf)**2
    b = np.sum(np.abs(meas_pf - mie_pf))
    b_0 = np.abs(meas_pf - mie_pf)
    c = np.sum(np.divide(np.abs(meas_pf - mie_pf), mie_pf))
    c_0 = np.divide(np.abs(meas_pf - mie_pf), mie_pf)
    return c


def Residuals_SL(x, w_n, SL_M, SL_Theta, n, k):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(1.0, 1000.0, 2.0)
    counts = [Gaussian(element, x[0], x[1], 400) for element in sizes]
    rad_mie, SL, SR, SU = PMS.SF_SD(m=complex(n, k), wavelength=w_n, dp=sizes, ndp=counts, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
    theta_mie = [(i * 180.0) / pi for i in rad_mie]
    SL_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SL, x=SL_Theta, der=0, axis=0))
    residuals = M(SL_pchip_norm, SL_M)
    #print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3], 'Residuals: ', residuals)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Residuals: ', residuals)
    return residuals


def Residuals_SU(x, w_n, SU_M, SU_Theta, n, k):
    # pre allocation
    SU_2darray = []
    # Data
    SU_M = np.array(SU_M)
    SU_M = SU_M[~np.isnan(SU_M)]
    SU_Theta = SU_Theta[~np.isnan(SU_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(1.0, 1000.0, 2.0)
    counts = [Gaussian(element, x[0], x[1], 400) for element in sizes]
    rad_mie, SL, SR, SU = PMS.SF_SD(m=complex(n, k), wavelength=w_n, dp=sizes, ndp=counts, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
    theta_mie = [(i * 180.0) / pi for i in rad_mie]
    SU_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SU, x=SU_Theta, der=0, axis=0))
    residuals = M(SU_pchip_norm, SU_M)
    #print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3], 'Residuals: ', residuals)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Residuals: ', residuals)
    return residuals


def Residuals_SR(x, w_n, SR_M, SR_Theta, n, k):
    # pre allocation
    SR_2darray = []
    # Data
    SR_M = np.array(SR_M)
    SR_M = SR_M[~np.isnan(SR_M)]
    SR_Theta = SR_Theta[~np.isnan(SR_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(1.0, 1000.0, 2.0)
    counts = [Gaussian(element, x[0], x[1], 400) for element in sizes]
    rad_mie, SL, SR, SU = PMS.SF_SD(m=complex(n, k), wavelength=w_n, dp=sizes, ndp=counts, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
    theta_mie = [(i * 180.0) / pi for i in rad_mie]
    SR_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SR, x=SR_Theta, der=0, axis=0))
    residuals = M(SR_pchip_norm, SR_M)
    #print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3], 'Residuals: ', residuals)
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Residuals: ', residuals)
    return residuals


def Residuals_SLSRSU(x, w_n, SL_M, SR_M, SU_M, SL_Theta, SR_Theta, SU_Theta, n, k):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    SR_M = np.array(SR_M)
    SR_M = SR_M[~np.isnan(SR_M)]
    SR_Theta = SR_Theta[~np.isnan(SR_M)]
    SU_M = np.array(SU_M)
    SU_M = SU_M[~np.isnan(SU_M)]
    SU_Theta = SU_Theta[~np.isnan(SU_M)]
    # theta_cal = np.array([slope * element + intercept for element in SU_C])
    sizes = np.arange(1.0, 1000.0, 2.0)
    counts = [Gaussian(element, x[0], x[1], 400) for element in sizes]
    rad_mie, SL, SR, SU = PMS.SF_SD(complex(n, k), wavelength=w_n, dp=sizes, ndp=counts, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
    theta_mie = np.array(rad_mie) * (180.0 / pi)
    SL_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SL, x=SL_Theta, der=0, axis=0))
    SR_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SR, x=SR_Theta, der=0, axis=0))
    SU_pchip_norm = Normalization(pchip_interpolate(xi=theta_mie, yi=SU, x=SU_Theta, der=0, axis=0))
    sl_diff = M(SL_pchip_norm, SL_M)
    sr_diff = M(SR_pchip_norm, SR_M)
    su_diff = M(SU_pchip_norm, SU_M)
    residuals = sl_diff + sr_diff + su_diff
    print('Guess mu: ', x[0], 'Guess sigma: ', x[1], 'Residuals: ', residuals)
    return residuals


# import data
save_directory = '/home/austen/Desktop/Recent/PSL_Temporary'
#file_directory = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
file_directory = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
bkg_directory = '/home/austen/Desktop/Recent/Good_Data_gfit_bkg.txt'
signal_directory = '/home/austen/Desktop/Recent/Good_Data_gfit_bc.txt'

# create snr dataframe for background signal, signal, and signal to noise ratio
header = ['Sample', 'Size (nm)', 'Exposure Time (s)', 'Polarization', 'Laser Power (mW)', 'Number of Averages', 'Concentration (p/cc)', 'Date', 'Time', 'Calibration Slope', 'Calibration Intercept']
bkg_df = pd.read_csv(bkg_directory, sep=',', header=0)
sig_df = pd.read_csv(signal_directory, sep=',', header=0)
sig_df_labels = sig_df.loc[:, header]
snr_df_vals = sig_df.loc[:,'30':'826'].div(bkg_df.loc[:,'30':'826'])
snr_df = pd.concat([sig_df_labels, snr_df_vals], axis=1)

# create dataframe for total signal
df = pd.read_csv(file_directory, sep=',', header=0)

# eliminate extra column
#df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#print(df)

# multiindex dataframe
df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Date'], inplace=True)
bkg_df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Date'], inplace=True)
sig_df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Date'], inplace=True)
snr_df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Date'], inplace=True)
#print(df)

# selecting sample, size, and date of measurement
sample_string = 'AS'
sample_size = 600
#2020-09-14
sample_date_SL = '2020-10-17'
#2020-09-20
sample_date_SU = '2020-10-18'
#2020-09-20
sample_date_SR = '2020-10-18'

# pandas dataframe.xs returns a cross-section of the data, so basically I am filtering out data that isn't PSL, size 900, and pol = SL
#xs_tuple = ('PSL', 900, 'SL')
# this df_900_SL1 and df_900_SL2, are used to merge data, this is from the same measurement period, just ran past 12AM
df_SL_All = df.xs((sample_string, sample_size, 'SL')).reset_index()
bkg_SL_All = bkg_df.xs((sample_string, sample_size, 'SL')).reset_index()
sig_SL_All = sig_df.xs((sample_string, sample_size, 'SL')).reset_index()
snr_SL_All = snr_df.xs((sample_string, sample_size, 'SL')).reset_index()
# in the future the contents in the brackets [] needs to be filtered into index value to keep (they are True) and index value to discard (False) '2020-09-14'
df_SL = df_SL_All[df_SL_All.Date == sample_date_SL].reset_index()
bkg_SL = bkg_SL_All[bkg_SL_All.Date == sample_date_SL].reset_index()
sig_SL = sig_SL_All[sig_SL_All.Date == sample_date_SL].reset_index()
snr_SL = snr_SL_All[snr_SL_All.Date == sample_date_SL].reset_index()
## this is the old way, we used to have to concatenate these with different dates, now we add the dates to keep dates and which can be made for each size, and boom uses only the dates we care about!
#df_900_SU_All = df.xs(('PSL', 900, 'SU', '2020-08-08')).reset_index()
#df_900_SL = pd.concat([df_900_SL1, df_900_SL2], ignore_index=True)

df_SU_All = df.xs((sample_string, sample_size, 'SU')).reset_index()
bkg_SU_All = bkg_df.xs((sample_string, sample_size, 'SU')).reset_index()
sig_SU_All = sig_df.xs((sample_string, sample_size, 'SU')).reset_index()
snr_SU_All = snr_df.xs((sample_string, sample_size, 'SU')).reset_index()
df_SU = df_SU_All[df_SU_All.Date == sample_date_SU].reset_index()
bkg_SU = bkg_SU_All[bkg_SU_All.Date == sample_date_SU].reset_index()
sig_SU = sig_SU_All[sig_SU_All.Date == sample_date_SU].reset_index()
snr_SU = snr_SU_All[snr_SU_All.Date == sample_date_SU].reset_index()

df_SR_All = df.xs((sample_string, sample_size, 'SR')).reset_index()
bkg_SR_All = bkg_df.xs((sample_string, sample_size, 'SR')).reset_index()
sig_SR_All = sig_df.xs((sample_string, sample_size, 'SR')).reset_index()
snr_SR_All = snr_df.xs((sample_string, sample_size, 'SR')).reset_index()
df_SR = df_SR_All[df_SR_All.Date == sample_date_SR].reset_index()
bkg_SR = bkg_SR_All[bkg_SR_All.Date == sample_date_SR].reset_index()
sig_SR = sig_SR_All[sig_SR_All.Date == sample_date_SR].reset_index()
snr_SR = snr_SR_All[snr_SR_All.Date == sample_date_SR].reset_index()


# view some of the data, use .loc if multiindexed,
#print(df_900_SL.loc[:, 'Exposure Time (s)'], df_900_SL.loc[:, 'Laser Power (mW)'], df_900_SL.loc[:, 'Number of Averages'])
#print(df_900_SU.loc[:, 'Exposure Time (s)'], df_900_SU.loc[:, 'Laser Power (mW)'], df_900_SU.loc[:, 'Number of Averages'])
#print(df_900_SR.loc[:, 'Exposure Time (s)'], df_900_SR.loc[:, 'Laser Power (mW)'], df_900_SR.loc[:, 'Number of Averages'])

# if not just reset the index of the multiindexed data and use as normal
print(df_SL)

# compute Mie theory for PSL

'''
PSLs:
Mean    Mean Uncertainty     Size Dist Sigma
600nm     9nm                     10.0nm
701nm     6nm                     9.0nm
800nm     14nm                    5.6nm
903nm     12nm                    4.1nm
'''

n_AS = 1.525
k_AS = 0.00
n_PSL = 1.58514608
k_PSL = 0.0
wavelength_red = 663
col_transects = np.arange(30, 827, 1)
slope = .2095
intercept = -3.1433
theta_meas = np.array([(slope * i) + intercept for i in col_transects])
dp_gaussian = np.arange(1.0, 1000.0, 2.0)
ndp_LN_900 = np.array([Gaussian(x=i, mu=903.0, sigma=4.1, N=400) for i in dp_gaussian])
#ndp_LN_700 = np.array([Gaussian(x=i, mu=701.0, sigma=9.0, N=433) for i in dp_gaussian])
ndp_LN_600 = np.array([Gaussian(x=i, mu=600.0, sigma=10.0, N=1000) for i in dp_gaussian])

Rad900, SL900, SR900, SU900 = PMS.SF_SD(m=complex(n_PSL, k_PSL), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_900, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
#Rad700, SL700, SR700, SU700 = PMS.SF_SD(m=complex(n_PSL, k_PSL), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_700, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
Rad600, SL600, SR600, SU600 = PMS.SF_SD(m=complex(n_PSL, k_PSL), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_600, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)

Theta900 = np.array([(i * 180.0)/pi for i in Rad900])
#Theta700 = np.array([(i * 180.0)/pi for i in Rad700])
Theta600 = np.array([(i * 180.0)/pi for i in Rad600])
#'''
SL900_pchip = pchip_interpolate(xi=Theta900, yi=SL900, x=theta_meas)
SU900_pchip = pchip_interpolate(xi=Theta900, yi=SU900, x=theta_meas)
SR900_pchip = pchip_interpolate(xi=Theta900, yi=SR900, x=theta_meas)
#'''
'''
SL700_pchip = pchip_interpolate(xi=Theta700, yi=SL700, x=theta_meas)
SU700_pchip = pchip_interpolate(xi=Theta700, yi=SU700, x=theta_meas)
SR700_pchip = pchip_interpolate(xi=Theta700, yi=SR700, x=theta_meas)
'''

SL600_pchip = pchip_interpolate(xi=Theta600, yi=SL600, x=theta_meas)
SU600_pchip = pchip_interpolate(xi=Theta600, yi=SU600, x=theta_meas)
SR600_pchip = pchip_interpolate(xi=Theta600, yi=SR600, x=theta_meas)

#'''
SL900_norm = SL900_pchip / np.sum(SL900_pchip)
SU900_norm = SU900_pchip / np.sum(SU900_pchip)
SR900_norm = SR900_pchip / np.sum(SR900_pchip)
#'''
'''
#SL700_norm = SL700_pchip / np.sum(SL700_pchip)
#SU700_norm = SU700_pchip / np.sum(SU700_pchip)
#SR700_norm = SR700_pchip / np.sum(SR700_pchip)
'''

SL600_norm = SL600_pchip / np.sum(SL600_pchip)
SU600_norm = SU600_pchip / np.sum(SU600_pchip)
SR600_norm = SR600_pchip / np.sum(SR600_pchip)




SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
legend_properties = {'weight':'bold'}

intensity_stitch = 5500
intensity_stitch_array = np.repeat(intensity_stitch, repeats=len(theta_meas))
intensity_stitch_array_norm = Normalization(intensity_stitch_array)
intensity_stitch_norm = intensity_stitch_array_norm[0]
PF1_2dlist = []
PF2_2dlist = []
sep_idx = []
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
for index, row in df_SL.iterrows():
    #print(np.array(row))
    pf_SL = np.array(row.loc['30':'826']).astype(float)
    ax[0].semilogy(theta_meas, pf_SL, ls='-', label=np.array(row.loc['Date':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf_SL), ls='-', label=np.array(row.loc['Date':'Time']))
ax[0].semilogy(theta_meas, SL900_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].grid(True)
#ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax[1].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax[1].set_title('Normalized Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].grid(True)
#ax[1].legend(loc=1)
plt.savefig(save_directory + '/measurements_made_SL.pdf', format='pdf')
plt.savefig(save_directory + '/measurements_made_SL.png', format='png')
plt.show()


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
for index, row in df_SU.iterrows():
    #print(np.array(row))
    pf_SU = np.array(row.loc['30':'826']).astype(float)
    ax[0].semilogy(theta_meas, pf_SU, ls='-', label=np.array(row.loc['Date':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf_SU), ls='-', label=np.array(row.loc['Date':'Time']))
ax[0].semilogy(theta_meas, SU900_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].grid(True)
#ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SU900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax[1].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax[1].set_title('Normalized Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].grid(True)
#ax[1].legend(loc=1)
plt.savefig(save_directory + '/measurements_made_SU.pdf', format='pdf')
plt.savefig(save_directory + '/measurements_made_SU.png', format='png')
plt.show()


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 18))
for index, row in df_SR.iterrows():
    #print(np.array(row))
    pf_SR = np.array(row.loc['30':'826']).astype(float)
    ax[0].semilogy(theta_meas, pf_SR, ls='-', label=np.array(row.loc['Date':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf_SR), ls='-', label=np.array(row.loc['Date':'Time']))
ax[0].semilogy(theta_meas, SR900_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].grid(True)
#ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SR900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax[1].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax[1].set_title('Normalized Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].grid(True)
#ax[1].legend(loc=1)
plt.savefig(save_directory + '/measurements_made_SR.pdf', format='pdf')
plt.savefig(save_directory + '/measurements_made_SR.png', format='png')
plt.show()
'''
pf1_2dlabels = []
pf2_2dlabels = []
pf1_2dlist = []
pf2_2dlist = []
for index, row in df_SL.iterrows():
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

# starting to split SL measurements at threshhold value
pf1_2dlabels_SL = []
pf2_2dlabels_SL = []
pf1_2dlist_SL = []
pf2_2dlist_SL = []
snr1_2dlist_SL = []
snr2_2dlist_SL = []
for index, row in df_SL.iterrows():
    counter = 0
    pf1_details_SL = []
    pf1_SL = np.zeros(797)
    snr1_SL = np.zeros(797)
    pf2_details_SL = []
    pf2_SL =np.zeros(797)
    snr2_SL = np.zeros(797)
    pf_SL = np.array(row['30':'826']).astype(float)
    pf_norm_SL = Normalization(pf_SL)
    for element in pf_norm_SL:
        if element >= intensity_stitch_norm:
            pf1_SL[counter] = element
            snr1_SL[counter] = snr_SL.loc[index, str(30 + counter)]
        if element < intensity_stitch_norm:
            pf2_SL[counter] = element
            snr2_SL[counter] = snr_SL.loc[index, str(30 + counter)]
        counter += 1
    pf1_details_SL.append(np.array(row['Date':'Time']))
    split_idx_SL = len(np.trim_zeros(pf1_SL))
    pf1_2dlist_SL.append(pf1_SL)
    snr1_2dlist_SL.append(snr1_SL)
    pf2_details_SL.append(np.array(row['Date':'Time']))
    pf2_2dlist_SL.append(pf2_SL)
    snr2_2dlist_SL.append(snr2_SL)


# list to array for snr
snr1_2darray_SL = np.array(snr1_2dlist_SL)
snr2_2darray_SL = np.array(snr2_2dlist_SL)
#print(snr1_2darray_SL[0])
print(snr1_2darray_SL[0].shape)
print(snr2_2darray_SL[0].shape)

# starting to split SU measurements at threshhold value
pf1_2dlabels_SU = []
pf2_2dlabels_SU = []
pf1_2dlist_SU = []
pf2_2dlist_SU = []
snr1_2dlist_SU = []
snr2_2dlist_SU = []
for index, row in df_SU.iterrows():
    counter = 0
    pf1_details_SU = []
    pf1_SU = np.zeros(797)
    snr1_SU = np.zeros(797)
    pf2_details_SU = []
    pf2_SU =np.zeros(797)
    snr2_SU = np.zeros(797)
    pf_SU = np.array(row['30':'826']).astype(float)
    pf_norm_SU = Normalization(pf_SU)
    for element in pf_norm_SU:
        if element >= intensity_stitch_norm:
            pf1_SU[counter] = element
            snr1_SU[counter] = snr_SU.loc[index, str(30 + counter)]
        if element < intensity_stitch_norm:
            pf2_SU[counter] = element
            snr2_SU[counter] = snr_SU.loc[index, str(30 + counter)]
        counter += 1
    pf1_details_SU.append(np.array(row['Date':'Time']))
    split_idx_SU = len(np.trim_zeros(pf1_SU))
    pf1_2dlist_SU.append(pf1_SU)
    snr1_2dlist_SU.append(snr1_SU)
    pf2_details_SU.append(np.array(row['Date':'Time']))
    pf2_2dlist_SU.append(pf2_SU)
    snr2_2dlist_SU.append(snr2_SU)


# list to array for snr
snr1_2darray_SU = np.array(snr1_2dlist_SU)
snr2_2darray_SU = np.array(snr2_2dlist_SU)
#print(snr1_2darray_SL[0])
#print(snr1_2darray_SU.shape)

pf1_2dlabels_SR = []
pf2_2dlabels_SR = []
pf1_2dlist_SR = []
pf2_2dlist_SR = []
snr1_2dlist_SR = []
snr2_2dlist_SR = []
for index, row in df_SR.iterrows():
    counter = 0
    pf1_details_SR = []
    pf1_SR = np.zeros(797)
    snr1_SR = np.zeros(797)
    pf2_details_SR = []
    pf2_SR =np.zeros(797)
    snr2_SR = np.zeros(797)
    pf_SR = np.array(row['30':'826']).astype(float)
    pf_norm_SR = Normalization(pf_SR)
    for element in pf_norm_SR:
        if element >= intensity_stitch_norm:
            pf1_SR[counter] = element
            snr1_SR[counter] = snr_SR.loc[index, str(30 + counter)]
        if element < intensity_stitch_norm:
            pf2_SR[counter] = element
            snr2_SR[counter] = snr_SR.loc[index, str(30 + counter)]
        counter += 1
    pf1_details_SR.append(np.array(row['Date':'Time']))
    split_idx_SR = len(np.trim_zeros(pf1_SR))
    pf1_2dlist_SR.append(pf1_SR)
    snr1_2dlist_SR.append(snr1_SR)
    pf2_details_SR.append(np.array(row['Date':'Time']))
    pf2_2dlist_SR.append(pf2_SR)
    snr2_2dlist_SR.append(snr2_SR)


# list to array for snr
snr1_2darray_SR = np.array(snr1_2dlist_SR)
snr2_2darray_SR = np.array(snr2_2dlist_SR)
#print(snr1_2darray_SR)
#print(snr2_2darray_SR)
#print(snr1_2darray_SR[0])
#print(snr1_2darray_SR.shape)


# selecting for the phase functtions we wanna compare to
#T1_label = np.array(df_SL.loc[(df_SL['Exposure Time (s)'] == 6) & (df_SL['Laser Power (mW)'] == 10)].loc[:, 'Date':'Time'])
#T2_label = np.array(df_SL.loc[(df_SL['Exposure Time (s)'] == 6) & (df_SL['Laser Power (mW)'] == 25)].loc[:, 'Date':'Time'])
#print(T1_label)
#T1 = np.array(df_SL.loc[(df_SL['Exposure Time (s)'] == 6) & (df_SL['Laser Power (mW)'] == 10)].loc[:, '30':'826'])[0]
#T2 = np.array(df_SL.loc[(df_SL['Exposure Time (s)'] == 6) & (df_SL['Laser Power (mW)'] == 25)].loc[:, '30':'826'])[0]
#print(T1)
#T1 = Normalization(T1)
#T2 = Normalization(T2)

#add in other SR and SU polarizations! 09/24/2020
# this SL figure is working! need to get the SU and SR to be exactly like it and then move the threshold line up for the SL case! 10/08/2020
# also fix the correlation coefficient right now its a 2 x 2 matrix, we need to get one number that makes sense! (its one of the elements in the matrix) 10/08/2020
pf_combos_norm_list_SL = []
snr_combos_list_SL = []
m_list_SL = []
corr_coeff_list_SL = []
label_list_SL = []
theta_list_SL = []
f1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(30, 18))
for counter, element in enumerate(pf1_2dlist_SL):
    for counter2, element2 in enumerate(pf2_2dlist_SL):
        drop_nan_idx_SL = []
        pf_combos_SL = np.add(np.array(element), np.array(pf2_2dlist_SL[counter2]))
        snr_combos_SL = np.add(snr1_2darray_SL[counter], snr2_2darray_SL[counter2])
        pf_combos_SL[pf_combos_SL == 0] = np.nan
        nan_bool_SL = np.isnan(pf_combos_SL)
        for counter3, element3 in enumerate(nan_bool_SL):
            if element3 == True:
                drop_nan_idx_SL.append(counter3)
        if any(nan_bool_SL) == True:
            pf_combos_dropnan = np.delete(pf_combos_SL, drop_nan_idx_SL)
            snr_combos_dropnan = np.delete(snr_combos_SL, drop_nan_idx_SL)
            theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx_SL)
            SL_norm_dropnan = np.delete(SL900_norm, drop_nan_idx_SL)
        else:
            pf_combos_dropnan = pf_combos_SL
            snr_combos_dropnan = snr_combos_SL
            theta_meas_dropnan = theta_meas
            SL_norm_dropnan = SL900_norm
        m = M(pf_combos_dropnan, SL_norm_dropnan)
        corr_coeff_SL = np.corrcoef(pf_combos_dropnan, SL_norm_dropnan)[0][1]
        label_string = str(np.array(df_SL.loc[counter, 'Date':'Time'])) + ' &\n' + str(np.array(df_SL.loc[counter2, 'Date':'Time']))
        pf_combos_norm_list_SL.append(pf_combos_dropnan)
        snr_combos_list_SL.append(snr_combos_dropnan)
        m_list_SL.append(m)
        corr_coeff_list_SL.append(corr_coeff_SL)
        theta_list_SL.append(theta_meas_dropnan)
        label_list_SL.append(np.array(label_string))
        #col_combos = np.add(np.array(pf1_2dlabels[counter]), np.array(pf2_2dlabels[counter2]))
        #theta_combos = np.array([(slope * i) + intercept for i in col_combos])
        ax1[0, 0].semilogy(theta_meas_dropnan, pf_combos_dropnan, ls='-', label=label_string)
        ax1[1, 0].plot(theta_meas_dropnan, snr_combos_dropnan, label=label_string)
m_list_SL = np.array(m_list_SL)
#print(m_list)
m_min_val_SL = np.amin(m_list_SL)
m_min_idx_SL = np.argmin(m_list_SL)
ax1[0, 0].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax1[0, 0].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax1[0, 0].set_title('SL Normalized Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].grid(True)
ax1[1, 0].set_title('SL SNR Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].grid(True)
ax1[0, 1].semilogy(theta_list_SL[m_min_idx_SL], pf_combos_norm_list_SL[m_min_idx_SL], ls='-', color='red', label=str(label_list_SL[m_min_idx_SL]) + '\n correlation coefficient: ' + str(corr_coeff_list_SL[m_min_idx_SL]) + '\n minimum residual sum: ' + str(m_list_SL[m_min_idx_SL]))
#ax1[1].semilogy(theta_meas, T1, ls='-', color='green', label=T1_label)
#ax1[1].semilogy(theta_meas, T2, ls='-', color='blue', label=T2_label)
ax1[0, 1].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax1[0, 1].set_title('Best SL Normalized Stitched Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 1].grid(True)
ax1[0, 1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
ax1[1, 1].plot(theta_list_SL[m_min_idx_SL], snr_combos_list_SL[m_min_idx_SL], ls='-', color='red', label=str(label_list_SL[m_min_idx_SL]))
ax1[1, 1].set_title('Best SL SNR Stitched Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 1].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 1].grid(True)
ax1[1, 1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/SL_stitches.pdf', format='pdf')
plt.savefig(save_directory + '/SL_stitches.png', format='png')
plt.show()

df_pf_combos_norm_list_SL = pd.DataFrame(pf_combos_norm_list_SL)
#print(df_pf_combos_norm_list_SL)
df_theta_list_SL = pd.DataFrame(theta_list_SL)
#print(df_theta_list_SL)
df_labels_list_SL = pd.DataFrame(label_list_SL)
#print(df_labels_list_SL)
df_SL_stitched = pd.concat([df_labels_list_SL, df_pf_combos_norm_list_SL], axis=1)
df_SL_stitched_theta = pd.concat([df_labels_list_SL, df_theta_list_SL], axis=1)
df_SL_stitched['Residuals'] = m_list_SL
df_SL_stitched['Correlation Coefficicent'] = corr_coeff_list_SL
df_SL_stitched.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SL900.txt', sep=',', header=True, index=False)
df_SL_stitched_theta.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SL900_theta.txt', sep=',', header=True, index=False)
#print(df_SL_stitched)


#add in other SR and SU polarizations! 09/24/2020
pf_combos_norm_list_SU = []
snr_combos_list_SU = []
m_list_SU = []
corr_coeff_list_SU = []
label_list_SU = []
theta_list_SU = []
f1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(30, 18))
for counter, element in enumerate(pf1_2dlist_SU):
    for counter2, element2 in enumerate(pf2_2dlist_SU):
        drop_nan_idx_SU = []
        pf_combos_SU = np.add(np.array(element), np.array(pf2_2dlist_SU[counter2]))
        snr_combos_SU = np.add(snr1_2darray_SU[counter], snr2_2darray_SU[counter2])
        pf_combos_SU[pf_combos_SU == 0] = np.nan
        nan_bool_SU = np.isnan(pf_combos_SU)
        for counter3, element3 in enumerate(nan_bool_SU):
            if element3 == True:
                drop_nan_idx_SU.append(counter3)
        if any(nan_bool_SU) == True:
            pf_combos_dropnan = np.delete(pf_combos_SU, drop_nan_idx_SU)
            snr_combos_dropnan = np.delete(snr_combos_SU, drop_nan_idx_SU)
            theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx_SU)
            SU_norm_dropnan = np.delete(SU900_norm, drop_nan_idx_SU)
        else:
            pf_combos_dropnan = pf_combos_SU
            snr_combos_dropnan = snr_combos_SU
            theta_meas_dropnan = theta_meas
            SU_norm_dropnan = SU900_norm
        m = M(pf_combos_dropnan, SU_norm_dropnan)
        corr_coeff_SU = np.corrcoef(pf_combos_dropnan, SU_norm_dropnan)[0][1]
        label_string = str(np.array(df_SU.loc[counter, 'Date':'Time'])) + ' &\n' + str(np.array(df_SU.loc[counter2, 'Date':'Time']))
        pf_combos_norm_list_SU.append(pf_combos_dropnan)
        snr_combos_list_SU.append(snr_combos_dropnan)
        m_list_SU.append(m)
        corr_coeff_list_SU.append(corr_coeff_SU)
        theta_list_SU.append(theta_meas_dropnan)
        label_list_SU.append(label_string)
        #col_combos = np.add(np.array(pf1_2dlabels[counter]), np.array(pf2_2dlabels[counter2]))
        #theta_combos = np.array([(slope * i) + intercept for i in col_combos])
        ax1[0, 0].semilogy(theta_meas_dropnan, pf_combos_dropnan, ls='-', label=label_string)
        ax1[1, 0].plot(theta_meas_dropnan, snr_combos_dropnan, label=label_string)
m_list_SU = np.array(m_list_SU)
#print(m_list)
m_min_val_SU = np.amin(m_list_SU)
m_min_idx_SU = np.argmin(m_list_SU)
ax1[0, 0].semilogy(theta_meas, SU900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax1[0, 0].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax1[0, 0].set_title('SU Normalized Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].grid(True)
ax1[1, 0].set_title('SU SNR Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].grid(True)
ax1[0, 1].semilogy(theta_list_SU[m_min_idx_SU], pf_combos_norm_list_SU[m_min_idx_SU], ls='-', color='green', label=str(label_list_SU[m_min_idx_SU]) + '\ncorrelation coefficient: ' + str(corr_coeff_list_SU[m_min_idx_SU])+ '\nminimum residual sum: ' + str(m_list_SU[m_min_idx_SU]))
#ax1[1].semilogy(theta_meas, T1, ls='-', color='green', label=T1_label)
#ax1[1].semilogy(theta_meas, T2, ls='-', color='blue', label=T2_label)
ax1[0, 1].semilogy(theta_meas, SU900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax1[0, 1].set_title('Best SU Normalized Stitched Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 1].grid(True)
ax1[0, 1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
ax1[1, 1].plot(theta_list_SU[m_min_idx_SU], snr_combos_list_SU[m_min_idx_SU], ls='-', color='green', label=str(label_list_SU[m_min_idx_SU]))
ax1[1, 1].set_title('Best SU SNR Stitched Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 1].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 1].grid(True)
ax1[1, 1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/SU_stitches.pdf', format='pdf')
plt.savefig(save_directory + '/SU_stitches.png', format='png')
plt.show()

df_pf_combos_norm_list_SU = pd.DataFrame(pf_combos_norm_list_SU)
#print(df_pf_combos_norm_list_SU)
df_theta_list_SU = pd.DataFrame(theta_list_SU)
#print(df_theta_list_SU)
df_labels_list_SU = pd.DataFrame(label_list_SU)
#print(df_labels_list_SU)
df_SU_stitched = pd.concat([df_labels_list_SU, df_pf_combos_norm_list_SU], axis=1)
df_SU_stitched_theta = pd.concat([df_labels_list_SU, df_theta_list_SU], axis=1)
df_SU_stitched['Residuals'] = m_list_SU
df_SU_stitched['Correlation Coefficicent'] = corr_coeff_list_SU
df_SU_stitched.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SU900.txt', sep=',', header=True, index=False)
df_SU_stitched_theta.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SU900_theta.txt', sep=',', header=True, index=False)


#add in other SR and SU polarizations! 09/24/2020
pf_combos_norm_list_SR = []
snr_combos_list_SR = []
m_list_SR = []
corr_coeff_list_SR = []
label_list_SR = []
theta_list_SR = []
f1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(30, 18))
for counter, element in enumerate(pf1_2dlist_SR):
    for counter2, element2 in enumerate(pf2_2dlist_SR):
        drop_nan_idx_SR = []
        pf_combos_SR = np.add(np.array(element), np.array(pf2_2dlist_SR[counter2]))
        snr_combos_SR = np.add(snr1_2darray_SR[counter], snr2_2darray_SR[counter2])
        pf_combos_SR[pf_combos_SR == 0] = np.nan
        nan_bool_SR = np.isnan(pf_combos_SR)
        for counter3, element3 in enumerate(nan_bool_SR):
            if element3 == True:
                drop_nan_idx_SR.append(counter3)
        if any(nan_bool_SR) == True:
            pf_combos_dropnan = np.delete(pf_combos_SR, drop_nan_idx_SR)
            snr_combos_dropnan = np.delete(snr_combos_SR, drop_nan_idx_SR)
            theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx_SR)
            SR_norm_dropnan = np.delete(SR900_norm, drop_nan_idx_SR)
        else:
            pf_combos_dropnan = pf_combos_SR
            snr_combos_dropnan = snr_combos_SR
            theta_meas_dropnan = theta_meas
            SR_norm_dropnan = SR900_norm
        m = M(pf_combos_dropnan, SR_norm_dropnan)
        corr_coeff_SR = np.corrcoef(pf_combos_dropnan, SR_norm_dropnan)[0][1]
        label_string = str(np.array(df_SR.loc[counter, 'Date':'Time'])) + ' &\n' + str(np.array(df_SR.loc[counter2, 'Date':'Time']))
        pf_combos_norm_list_SR.append(pf_combos_dropnan)
        snr_combos_list_SR.append(snr_combos_dropnan)
        m_list_SR.append(m)
        corr_coeff_list_SR.append(corr_coeff_SR)
        theta_list_SR.append(theta_meas_dropnan)
        label_list_SR.append(label_string)
        '''
        #col_combos = np.add(np.array(pf1_2dlabels[counter]), np.array(pf2_2dlabels[counter2]))
        #theta_combos = np.array([(slope * i) + intercept for i in col_combos])
        '''
        ax1[0, 0].semilogy(theta_meas_dropnan, pf_combos_dropnan, ls='-', label=label_string)
        ax1[1, 0].plot(theta_meas_dropnan, snr_combos_dropnan, label=label_string)
m_list_SR = np.array(m_list_SR)
#print(m_list)
m_min_val_SR = np.amin(m_list_SR)
m_min_idx_SR = np.argmin(m_list_SR)
ax1[0, 0].semilogy(theta_meas, SR900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax1[0, 0].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax1[0, 0].set_title('SR Normalized Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].grid(True)
ax1[1, 0].set_title('SR SNR Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].grid(True)
ax1[0, 1].semilogy(theta_list_SR[m_min_idx_SR], pf_combos_norm_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR] + '\ncorrelation coefficient: ' + str(corr_coeff_list_SR[m_min_idx_SR])+ '\nminimum residual sum: ' + str(m_list_SR[m_min_idx_SR]))
#ax1[1].semilogy(theta_meas, T1, ls='-', color='green', label=T1_label)
#ax1[1].semilogy(theta_meas, T2, ls='-', color='blue', label=T2_label)
ax1[0, 1].semilogy(theta_meas, SR900_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax1[0, 1].set_title('Best SR Normalized Stitched Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 1].grid(True)
ax1[1, 1].plot(theta_list_SR[m_min_idx_SR], snr_combos_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR])
ax1[1, 1].set_title('Best SR SNR Stitched Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 1].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 1].grid(True)
ax1[1, 1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
ax1[0, 1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/SR_stitches.pdf', format='pdf')
plt.savefig(save_directory + '/SR_stitches.png', format='png')
plt.show()


df_pf_combos_norm_list_SR = pd.DataFrame(pf_combos_norm_list_SR)
#print(df_pf_combos_norm_list_SR)
df_theta_list_SR = pd.DataFrame(theta_list_SR)
#print(df_theta_list_SR)
df_labels_list_SR = pd.DataFrame(label_list_SR)
#print(df_labels_list_SR)
df_SR_stitched = pd.concat([df_labels_list_SR, df_pf_combos_norm_list_SR], axis=1)
df_SR_stitched_theta = pd.concat([df_labels_list_SR, df_theta_list_SR], axis=1)
df_SR_stitched['Residuals'] = m_list_SR
df_SR_stitched['Correlation Coefficicent'] = corr_coeff_list_SR
df_SR_stitched.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SR900.txt', sep=',', header=True, index=False)
df_SR_stitched_theta.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SR900_theta.txt', sep=',', header=True, index=False)

'''
geoff_theory_theta = pd.DataFrame(np.vstack((theta_meas, theta_meas, theta_meas)))
geoff_theory = pd.DataFrame(np.vstack((SL900_norm,  SU900_norm, SR900_norm)))
geoff_pf_theta = pd.DataFrame(np.vstack((df_theta_list_SL.loc[m_min_idx_SL, :], df_theta_list_SU.loc[m_min_idx_SU, :], df_theta_list_SR.loc[m_min_idx_SR, :])))
geoff_pf = pd.DataFrame(np.vstack((df_pf_combos_norm_list_SL.loc[m_min_idx_SL, :], df_pf_combos_norm_list_SU.loc[m_min_idx_SU, :], df_pf_combos_norm_list_SR.loc[m_min_idx_SR, :])))
geoff_theory_theta['Polarization'] = ['SL', 'SU', 'SR']
geoff_theory_theta['Size'] = [903, 903, 903]
geoff_theory_theta['Type'] = ['Mie Theory', 'Mie Theory', 'Mie Theory']
geoff_theory['Polarization'] = ['SL', 'SU', 'SR']
geoff_theory['Size'] = [903, 903, 903]
geoff_theory['Type'] = ['Mie Theory', 'Mie Theory', 'Mie Theory']
geoff_pf_theta['Polarization'] = ['SL', 'SU', 'SR']
geoff_pf_theta['Size'] = [903, 903, 903]
geoff_pf_theta['Type'] = ['Measurement', 'Measurement', 'Measurement']
geoff_pf['Polarization'] = ['SL', 'SU', 'SR']
geoff_pf['Size'] = [903, 903, 903]
geoff_pf['Type'] = ['Measurement', 'Measurement', 'Measurement']
geoff_theory_theta.to_csv(save_directory + '/geoff_theory_theta.txt', sep=',', header=True, index=False)
geoff_theory.to_csv(save_directory + '/geoff_theory.txt', sep=',', header=True, index=False)
geoff_pf_theta.to_csv(save_directory + '/geoff_meas_theta.txt', sep=',', header=True, index=False)
geoff_pf.to_csv(save_directory + '/geoff_meas.txt', sep=',', header=True, index=False)
'''

# NLLS result
result_SL_stitch = least_squares(Residuals_SL, x0=[903, 4.1], method='trf', args=(wavelength_red, pf_combos_norm_list_SL[m_min_idx_SL], theta_list_SL[m_min_idx_SL], n_PSL, k_PSL), bounds=([850, 1.0],[950, 100.0]))

# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_SL_stitch.x[0])
print('\u03c3: ', result_SL_stitch.x[1])
#print('m: ', result_SL900_stitch.x[2])
#print('k: ', result_SL900_stitch.x[3])
print('iterations: ', result_SL_stitch.nfev)
print('status: ', result_SL_stitch.status)

# NLLS result
result_SU_stitch = least_squares(Residuals_SU, x0=[903, 4.1], method='trf', args=(wavelength_red, pf_combos_norm_list_SU[m_min_idx_SU], theta_list_SU[m_min_idx_SU], n_PSL, k_PSL), bounds=([850, 1.0],[950, 100.0]))

# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_SU_stitch.x[0])
print('\u03c3: ', result_SU_stitch.x[1])
#print('m: ', result_SL900_stitch.x[2])
#print('k: ', result_SL900_stitch.x[3])
print('iterations: ', result_SU_stitch.nfev)
print('status: ', result_SU_stitch.status)

# NLLS result
result_SR_stitch = least_squares(Residuals_SR, x0=[903, 4.1], method='trf', args=(wavelength_red, pf_combos_norm_list_SR[m_min_idx_SR], theta_list_SR[m_min_idx_SR], n_PSL, k_PSL), bounds=([850, 1.0],[950, 100.0]))

# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_SR_stitch.x[0])
print('\u03c3: ', result_SR_stitch.x[1])
#print('m: ', result_SL900_stitch.x[2])
#print('k: ', result_SL900_stitch.x[3])
print('iterations: ', result_SR_stitch.nfev)
print('status: ', result_SR_stitch.status)

# get distributions
sizes = np.arange(1.0, 1000.0, 2.0)
distribution_theory = np.array([Gaussian(x=i, mu=903.0, sigma=4.1, N=400) for i in sizes])
distribution_stitch_SL = np.array([Gaussian(x=i, mu=result_SL_stitch.x[0], sigma=result_SL_stitch.x[1], N=400) for i in sizes])
distribution_stitch_SU = np.array([Gaussian(x=i, mu=result_SU_stitch.x[0], sigma=result_SU_stitch.x[1], N=400) for i in sizes])
distribution_stitch_SR = np.array([Gaussian(x=i, mu=result_SR_stitch.x[0], sigma=result_SR_stitch.x[1], N=400) for i in sizes])
# calculate best agreement
m_TS = complex(n_PSL, k_PSL)
rad_TS1, SL_TS1, SR_TS1, SU_TS1 = PMS.SF_SD(m=m_TS, wavelength=wavelength_red, dp=sizes, ndp=distribution_stitch_SL, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
rad_TS2, SL_TS2, SR_TS2, SU_TS2 = PMS.SF_SD(m=m_TS, wavelength=wavelength_red, dp=sizes, ndp=distribution_stitch_SU, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
rad_TS3, SL_TS3, SR_TS3, SU_TS3 = PMS.SF_SD(m=m_TS, wavelength=wavelength_red, dp=sizes, ndp=distribution_stitch_SR, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)

# rad to theta
theta_TS1 = np.array([(i * 180.0)/pi for i in rad_TS1])
theta_TS2 = np.array([(i * 180.0)/pi for i in rad_TS2])
theta_TS3 = np.array([(i * 180.0)/pi for i in rad_TS3])
# pchip
SL_TS_pchip = pchip_interpolate(xi=theta_TS1, yi=SL_TS1, x=theta_meas)
SU_TS_pchip = pchip_interpolate(xi=theta_TS2, yi=SU_TS2, x=theta_meas)
SR_TS_pchip = pchip_interpolate(xi=theta_TS3, yi=SR_TS3, x=theta_meas)
# normalize
SL_TS_pchip_norm = Normalization(SL_TS_pchip)
SU_TS_pchip_norm = Normalization(SU_TS_pchip)
SR_TS_pchip_norm = Normalization(SR_TS_pchip)
# plot it all up
f2, ax2 = plt.subplots(nrows=1, ncols=4, figsize=(40, 14))
ax2[0].semilogy(theta_list_SL[m_min_idx_SL], pf_combos_norm_list_SL[m_min_idx_SL], ls='-', color='red', label=label_list_SL[m_min_idx_SL])
ax2[0].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[0].semilogy(theta_meas, SL_TS_pchip_norm, color='coral', ls='-', linewidth=4, label='Theoretical Match: ' + str(label_list_SL[m_min_idx_SL]))
ax2[0].set_title('SL Normalized Stitched Measurement,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax2[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[0].grid(True)
ax2[0].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax2[1].semilogy(theta_list_SU[m_min_idx_SU], pf_combos_norm_list_SU[m_min_idx_SU], ls='-', color='green', label=label_list_SU[m_min_idx_SU])
ax2[1].semilogy(theta_meas, SU900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[1].semilogy(theta_meas, SU_TS_pchip_norm, color='lawngreen', ls='-', linewidth=4, label='Theoretical Match: ' + str(label_list_SU[m_min_idx_SU]))
ax2[1].set_title('SU Normalized Stitched Measurement,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax2[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[1].grid(True)
ax2[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax2[2].semilogy(theta_list_SR[m_min_idx_SR], pf_combos_norm_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR])
ax2[2].semilogy(theta_meas, SR900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[2].semilogy(theta_meas, SR_TS_pchip_norm, color='cyan', ls='-', linewidth=4, label='Theoretical Match: ' + str(label_list_SR[m_min_idx_SR]))
ax2[2].set_title('SR Normalized Stitched Measurements,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax2[2].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[2].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[2].grid(True)
ax2[2].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax2[3].plot(sizes, distribution_stitch_SL, ls='-', color='red', label='SL: ' + '\u03bc: ' + str('{:.3f}'.format(result_SL_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SL_stitch.x[1])) + 'Dist.')
ax2[3].plot(sizes, distribution_stitch_SU, ls='-', color='green', label='SU: ' + '\u03bc: ' + str('{:.3f}'.format(result_SU_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SU_stitch.x[1])) + 'Dist.')
ax2[3].plot(sizes, distribution_stitch_SR, ls='-', color='blue', label='SR: ' + '\u03bc: ' + str('{:.3f}'.format(result_SR_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SR_stitch.x[1])) + 'Dist.')
ax2[3].plot(sizes, distribution_theory, ls='-', color='black', label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[3].set_title('Retrieved Distributions', fontsize=BIGGER_SIZE, fontweight='bold')
ax2[3].set_xlabel('Sizes (nm)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[3].set_ylabel('Counts (p/cc)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[3].grid(True)
ax2[3].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/best_&_retrievals_independent.pdf', format='pdf')
plt.savefig(save_directory + '/best_&_retrievals_independent.png', format='png')
plt.show()


# NLLS result
result_SLSRSU_stitch = least_squares(Residuals_SLSRSU, x0=[903, 4.1], method='trf', args=(wavelength_red, pf_combos_norm_list_SL[m_min_idx_SL], pf_combos_norm_list_SR[m_min_idx_SR], pf_combos_norm_list_SU[m_min_idx_SU], theta_list_SL[m_min_idx_SL], theta_list_SR[m_min_idx_SR], theta_list_SU[m_min_idx_SU], n_PSL, k_PSL), bounds=([850, 1.0],[950, 100.0]))
# minimum values
print('-------Solution Reached!-------')
print('\u03bc: ', result_SLSRSU_stitch.x[0])
print('\u03c3: ', result_SLSRSU_stitch.x[1])
#print('m: ', result_SL900_stitch.x[2])
#print('k: ', result_SL900_stitch.x[3])
print('iterations: ', result_SLSRSU_stitch.nfev)
print('status: ', result_SLSRSU_stitch.status)

# get distributions
distribution_stitch_SLSRSU = np.array([Gaussian(x=i, mu=result_SLSRSU_stitch.x[0], sigma=result_SLSRSU_stitch.x[1], N=400) for i in sizes])

# calculate best agreement
m_TS = complex(n_PSL, k_PSL)
rad_combo, SL_combo, SR_combo, SU_combo = PMS.SF_SD(m=m_TS, wavelength=wavelength_red, dp=sizes, ndp=distribution_stitch_SLSRSU, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)

# rad to theta
theta_combo = np.array([(i * 180.0)/pi for i in rad_combo])

# pchip
SL_combo_pchip = pchip_interpolate(xi=theta_combo, yi=SL_combo, x=theta_meas)
SU_combo_pchip = pchip_interpolate(xi=theta_combo, yi=SU_combo, x=theta_meas)
SR_combo_pchip = pchip_interpolate(xi=theta_combo, yi=SR_combo, x=theta_meas)

# normalize
SL_combo_pchip_norm = Normalization(SL_combo_pchip)
SU_combo_pchip_norm = Normalization(SU_combo_pchip)
SR_combo_pchip_norm = Normalization(SR_combo_pchip)

# plot it all up
f3, ax3 = plt.subplots(nrows=1, ncols=4, figsize=(40, 14))
ax3[0].semilogy(theta_list_SL[m_min_idx_SL], pf_combos_norm_list_SL[m_min_idx_SL], ls='-', color='red', label=label_list_SL[m_min_idx_SL])
ax3[0].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax3[0].semilogy(theta_meas, SL_combo_pchip_norm, color='purple', ls='-', linewidth=4, label='Retrieved Match: \u03bc' + str('{:.3f}'.format(result_SLSRSU_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SLSRSU_stitch.x[1])))
ax3[0].set_title('SL Normalized Stitched Measurements,\n Theory, and Retrieved', fontsize=BIGGER_SIZE, fontweight='bold')
ax3[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[0].grid(True)
ax3[0].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax3[1].semilogy(theta_list_SU[m_min_idx_SU], pf_combos_norm_list_SU[m_min_idx_SU], ls='-', color='green', label=label_list_SU[m_min_idx_SU])
ax3[1].semilogy(theta_meas, SU900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax3[1].semilogy(theta_meas, SU_combo_pchip_norm, color='purple', ls='-', linewidth=4, label='Retrieved Match: \u03bc' + str('{:.3f}'.format(result_SLSRSU_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SLSRSU_stitch.x[1])))
ax3[1].set_title('SU Normalized Stitched Measurements,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax3[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[1].grid(True)
ax3[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax3[2].semilogy(theta_list_SR[m_min_idx_SR], pf_combos_norm_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR])
ax3[2].semilogy(theta_meas, SR900_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax3[2].semilogy(theta_meas, SR_combo_pchip_norm, color='purple', ls='-', linewidth=4, label='Retrieved Match: \u03bc' + str('{:.3f}'.format(result_SLSRSU_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SLSRSU_stitch.x[1])))
ax3[2].set_title('SR Normalized Stitched Measurements,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax3[2].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[2].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[2].grid(True)
ax3[2].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax3[3].plot(sizes, distribution_stitch_SLSRSU, ls='-', color='purple', label='SL, SR, SU: ' + '\n' + label_list_SL[m_min_idx_SL] + '\n' + label_list_SR[m_min_idx_SR] + '\n' +label_list_SU[m_min_idx_SU] + 'Dist.')
ax3[3].plot(sizes, distribution_theory, ls='-', color='black', label='Specified Thermo Fisher Scientific PSL Dist.')
ax3[3].set_title('Retrieved Distributions', fontsize=BIGGER_SIZE, fontweight='bold')
ax3[3].set_xlabel('Sizes (nm)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[3].set_ylabel('Counts (p/cc)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[3].grid(True)
ax3[3].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/best_&_retrievals_combined.pdf', format='pdf')
plt.savefig(save_directory + '/best_&_retrievals_combined.png', format='png')
plt.show()



