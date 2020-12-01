'''
Austen K. Scruggs
08/06/2020
Description: Phase function stitching using PSL data
'''

import pandas as pd
import numpy as np
import PyMieScatt as PMS
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt, pi, log
from scipy.optimize import curve_fit
from datetime import date
from scipy.signal import savgol_filter, argrelmax, argrelmin
from scipy.stats import pearsonr
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


# defining cauchy equation functions
def cauchy_2term(wav, A_2term, B_2term):
    return A_2term + (B_2term / wav ** 2)

def cauchy_4term(wav, A_4term, B_4term, C_4term):
    return A_4term + (B_4term / wav ** 2) + (C_4term / (wav ** 4))

def cauchy_6term(wav, A_1_6term, A_2_6term, A_3_6term, A_4_6term, A_5_6term, A_6_6term):
    return np.sqrt(A_1_6term + (A_2_6term * wav ** 2) + (A_3_6term / wav ** 2) + (A_4_6term / (wav ** 4)) + (
                    A_5_6term / (wav ** 6)) + (A_6_6term / (wav ** 8)))



def PSL_CRI(nanometers_list, spectra_shortwave_nm, spectra_longwave_nm):
    today = date.today()
    today_string = str(today.strftime("%b-%d-%Y"))
    # refractive index for PSL calculated for each group (literally until line 240 skip!)
    # wavelength in nanometers
    #w_n_array = [350, 405, 532, 663]
    w_n_array = nanometers_list
    # wavelength in centimeters
    w_c_array = [i * 10**-7 for i in nanometers_list]
    # wavelength in microns
    #w_u_array = [.350, .405, .532, .663]
    w_u_array = [i/1000.0 for i in nanometers_list]
    # wavelength in angstroms
    #w_a_array = [3500, 4050, 5320, 6630]
    w_a_array = [i*10 for i in nanometers_list]
    # UV-Visible spectrum in angstroms
    uv_visible_spectrum_centimeters = np.arange(spectra_shortwave_nm * 10**-7, spectra_longwave_nm * 10**-7, 10E-7)
    # print(len(uv_visible_spectrum_centimeters))
    # UV-Visible spectrum in microns
    uv_visible_spectrum_microns = np.arange(spectra_shortwave_nm/1000.0, spectra_longwave_nm/1000.0, .010)
    # UV-Visible spectrum in nanometers
    uv_visible_spectrum_nanometers = np.arange(spectra_shortwave_nm, spectra_longwave_nm, 10)
    # print(len(uv_visible_spectrum_nanometers))
    # UV-Visible spectrum in angstroms
    uv_visible_spectrum_angstroms = np.arange(spectra_shortwave_nm*10, spectra_longwave_nm*10, 100)
    # complex refractive index, Cauchy parameters for PSL Matheson & Sanderson 1952, wavelength in microns from Greenslade
    A0 = 1.5663
    B0 = 0.00785
    C0 = 0.000334

    # complex refractive index, Cauchy parameters for PSL Bateman 1959 wavelength in centimeters
    A1 = 1.5683
    B1 = 10.087E-11

    # fit Nikalov data for cauchy coefficients, wavelength in microns
    w_Nikalov_microns = np.array([0.436, 0.486, 0.546, 0.588, 0.633, 0.656, 0.703, 0.752, 0.804, 0.833, 0.879, 1.052])
    w_Nikalov_nanometers = w_Nikalov_microns * 1000
    n_RI_Nikalov = np.array([1.617, 1.606, 1.596, 1.592, 1.587, 1.586, 1.582, 1.579, 1.578, 1.577, 1.576, 1.572])
    # print(w_Nikalov_microns)
    popt_Nikalov, pcov_Nikalov = curve_fit(cauchy_6term, w_Nikalov_microns, n_RI_Nikalov, p0=[2.44675093, -1.011623E-3, 2.840749E-2, -3.761631E-4, 8.193491E-5, 9.055861E-4])
    # print(popt_Nikalov)
    # complex refractive index, Cauchy parameters for PSL Nikalov et al 2000, wavelength in microns
    A2 = popt_Nikalov[0]
    B2 = popt_Nikalov[1]
    C2 = popt_Nikalov[2]
    D2 = popt_Nikalov[3]
    E2 = popt_Nikalov[4]
    F2 = popt_Nikalov[5]

    # complex refractive index, Cauchy parameters for PSL Ma et al 2003, wavelength in microns
    A3 = 1.5725
    B3 = 0.003108
    C3 = 0.00034779

    # complex refractive index, Cauchy parameters for PSL Sultanova et al 2003 wavelength in microns
    A4 = 2.44675093
    B4 = -1.011623E-3
    C4 = 2.840749E-2
    D4 = -3.761631E-4
    E4 = 8.193491E-5
    F4 = 2.186304E-5

    # complex refractive index, Cauchy parameters for PSL Kasarova et al 2006 wavelength in microns
    A5 = 2.610025
    B5 = -6.143673E-2
    C5 = -1.312267E-1
    D5 = 6.865432E-2
    E5 = -1.295968E-2
    F5 = 9.055861E-4

    # complex refractive index, Cauchy parameters for PSL Miles et al 2010 wavelength in microns
    A6 = 1.5663
    B6 = 0.00785
    C6 = 0.000334

    # complex refractive index, Cauchy parameters for PSL Jones et al 2013 wavelength in nanometers
    A7 = 1.5718
    B7 = 8412
    C7 = 2.35E8

    # complex refractive index, Cauchy Greenslade 2017, wavelength in microns
    A8 = 1.53811
    B8 = 0.004316
    C8 = 0.000945

    # complex refractive index, Cauchy Gienger 2017, wavelength in microns, uses Sellmeier equation, wavelength in nanometers
    B9 = 1.4432
    wav9 = 142.1

    # n of refractive index for PSL as cauchy equation at specific wavelengths
    PSL_groups = ['Matheson', 'Bateman', 'Nikalov', 'Ma', 'Sultanova', 'Kasarova', 'Miles', 'Jones', 'Greenslade', 'Gienger']
    n_matheson = np.array([A0 + (B0 / w_u ** 2) + (C0 / (w_u ** 4)) for w_u in w_u_array])
    n_bateman = np.array([A1 + (B1 / w_c ** 2) for w_c in w_c_array])
    n_nikalov = np.array([np.sqrt(A2 + (B2 * w_u ** 2) + (C2 / w_u ** 2) + (D2 / (w_u ** 4)) + (E2 / (w_u ** 6)) + (F2 / (w_u ** 8))) for w_u in w_u_array])
    n_ma = np.array([A3 + (B3 / w_u ** 2) + (C3 / (w_u ** 4)) for w_u in w_u_array])
    n_sultanova = np.array([np.sqrt(A4 + (B4 * w_u ** 2) + (C4 / w_u ** 2) + (D4 / (w_u ** 4)) + (E4 / (w_u ** 6)) + (F4 / (w_u ** 8))) for w_u in w_u_array])
    n_kasarova = np.array([np.sqrt(A5 + (B5 * w_u ** 2) + (C5 / w_u ** 2) + (D5 / (w_u ** 4)) + (E5 / (w_u ** 6)) + (F5 / (w_u ** 8))) for w_u in w_u_array])
    n_miles = np.array([A6 + (B6 / w_u ** 2) + (C6 / (w_u ** 4)) for w_u in w_u_array])
    n_jones = np.array([A7 + (B7 / w_n ** 2) + (C7 / (w_n ** 4)) for w_n in w_n_array])
    n_greenslade = np.array([A8 + (B8 / w_u ** 2) + (C8 / (w_u ** 4)) for w_u in w_u_array])
    # below used sellmeier equation
    n_gienger = np.array([np.sqrt(1 + ((B9 * w_n ** 2) / (w_n ** 2 - wav9 ** 2))) for w_n in w_n_array])
    n_at_wavelengths = [n_matheson, n_bateman, n_nikalov, n_ma, n_sultanova, n_kasarova, n_miles, n_jones, n_greenslade, n_gienger]

    # basic statistics
    n_all_groups = [n_matheson, n_bateman, n_nikalov, n_ma, n_sultanova, n_kasarova, n_miles, n_jones, n_greenslade, n_gienger]
    n_groups_red = [n_all_groups[x][3] for x in range(3)]
    n_groups_green = [n_all_groups[x][2] for x in range(3)]
    n_groups_blue = [n_all_groups[x][1] for x in range(3)]
    n_groups_uv = [n_all_groups[x][0] for x in range(3)]
    n_mean_red = np.mean(n_groups_red)
    n_mean_green = np.mean(n_groups_green)
    n_mean_blue = np.mean(n_groups_blue)
    n_mean_uv = np.mean(n_groups_uv)
    n_percentiles_red = np.percentile(n_groups_red, [0, 25, 50, 75, 100])
    n_percentiles_green = np.percentile(n_groups_green, [0, 25, 50, 75, 100])
    n_percentiles_blue = np.percentile(n_groups_blue, [0, 25, 50, 75, 100])
    n_percentiles_uv = np.percentile(n_groups_uv, [0, 25, 50, 75, 100])

    # n (real refractive index) spectrum
    n_matheson_spectrum = [A0 + (B0 / element ** 2) + (C0 / (element ** 4)) for element in uv_visible_spectrum_microns]
    n_bateman_spectrum = [A1 + (B1 / element ** 2) for element in uv_visible_spectrum_centimeters]
    n_nikalov_spectrum = [np.sqrt(A2 + (B2 * element ** 2) + (C2 / element ** 2) + (D2 / (element ** 4)) + (E2 / (element ** 6)) + (F2 / (element ** 8))) for element in uv_visible_spectrum_microns]
    n_ma_spectrum = [A3 + (B3 / element ** 2) + (C3 / (element ** 4)) for element in uv_visible_spectrum_microns]
    n_sultanova_spectrum = [np.sqrt(A4 + (B4 * element ** 2) + (C4 / element ** 2) + (D4 / (element ** 4)) + (E4 / (element ** 6)) + (F4 / (element ** 8))) for element in uv_visible_spectrum_microns]
    n_kasarova_spectrum = [np.sqrt(A5 + (B5 * element ** 2) + (C5 / element ** 2) + (D5 / (element ** 4)) + (E5 / (element ** 6)) + (F5 / (element ** 8))) for element in uv_visible_spectrum_microns]
    n_miles_spectrum = [A6 + (B6 / element ** 2) + (C6 / (element ** 4)) for element in uv_visible_spectrum_microns]
    n_jones_spectrum = [A7 + (B7 / element ** 2) + (C7 / (element ** 4)) for element in uv_visible_spectrum_nanometers]
    n_greenslade_spectrum = [A8 + (B8 / element ** 2) + (C8 / (element ** 4)) for element in uv_visible_spectrum_microns]
    n_gienger_spectrum = [np.sqrt(1 + ((B9 * element ** 2) / (element ** 2 - wav9 ** 2))) for element in uv_visible_spectrum_nanometers]
    n_all_spectra = [n_matheson_spectrum, n_bateman_spectrum, n_nikalov_spectrum, n_ma_spectrum, n_sultanova_spectrum, n_kasarova_spectrum, n_miles_spectrum, n_jones_spectrum, n_greenslade_spectrum, n_gienger_spectrum]


    CRI_Spectra = pd.DataFrame(n_all_spectra, columns=np.append(uv_visible_spectrum_nanometers, uv_visible_spectrum_nanometers[-1]+10), index=PSL_groups)
    CRI_4_Wavelengths = pd.DataFrame(n_at_wavelengths, columns=w_n_array, index=PSL_groups)


    return [CRI_Spectra, CRI_4_Wavelengths, n_groups_uv, n_groups_blue, n_groups_green, n_groups_red]


def M(meas_pf, mie_pf):
    #a = np.sum((meas_pf - mie_pf)**2)
    #a_0 = (meas_pf - mie_pf)**2
    #b = np.sum(np.abs(meas_pf - mie_pf))
    #b_0 = np.abs(meas_pf - mie_pf)
    c = np.sum(np.divide(np.abs(meas_pf - mie_pf), mie_pf))
    #c_0 = np.divide(np.abs(meas_pf - mie_pf), mie_pf)
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
file_directory = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
#file_directory = '/home/austen/Desktop/Recent/Good_Data_gfit.txt'
# contains nan values where it is impossible to fit a gaussian
# develop some code to drop nans at indices of both measurement and theory to use gfit
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
df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Laser Power (mW)','Date'], inplace=True)
bkg_df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Laser Power (mW)','Date'], inplace=True)
sig_df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Laser Power (mW)','Date'], inplace=True)
snr_df.set_index(['Sample', 'Size (nm)', 'Polarization', 'Laser Power (mW)', 'Date'], inplace=True)
#print(df)

# selecting sample, size, and date of measurement
sample_string = 'PSL'
sample_size = 1000
laser_power = 100
# PSL 900 = 2020-09-14, PSL 600 =  2020-09-30
sample_date_SL = '2020-10-21'
#PSL 900 = 2020-09-20, PSL 600 = 2020-09-30
sample_date_SU = '2020-10-21'
#PSL 900 = 2020-09-20, PSL 600 = 2020-09-29
sample_date_SR = '2020-10-21'

# pandas dataframe.xs returns a cross-section of the data, so basically I am filtering out data that isn't PSL, size 900, and pol = SL
#xs_tuple = ('PSL', 900, 'SL')
# this df_900_SL1 and df_900_SL2, are used to merge data, this is from the same measurement period, just ran past 12AM
# drop levels = False keeps the original index like i wanted
df_SL_All = df.xs((sample_string, sample_size, 'SL', laser_power), drop_level=False).reset_index()
bkg_SL_All = bkg_df.xs((sample_string, sample_size, 'SL', laser_power), drop_level=False).reset_index()
sig_SL_All = sig_df.xs((sample_string, sample_size, 'SL', laser_power), drop_level=False).reset_index()
snr_SL_All = snr_df.xs((sample_string, sample_size, 'SL', laser_power), drop_level=False).reset_index()
# in the future the contents in the brackets [] needs to be filtered into index value to keep (they are True) and index value to discard (False) '2020-09-14'
df_SL = df_SL_All[df_SL_All.Date >= sample_date_SL].reset_index()
bkg_SL = bkg_SL_All[bkg_SL_All.Date >= sample_date_SL].reset_index()
sig_SL = sig_SL_All[sig_SL_All.Date >= sample_date_SL].reset_index()
snr_SL = snr_SL_All[snr_SL_All.Date >= sample_date_SL].reset_index()
## this is the old way, we used to have to concatenate these with different dates, now we add the dates to keep dates and which can be made for each size, and boom uses only the dates we care about!
#df_900_SU_All = df.xs(('PSL', 900, 'SU', '2020-08-08')).reset_index()
#df_900_SL = pd.concat([df_900_SL1, df_900_SL2], ignore_index=True)
'''
df_SU_All = df.xs((sample_string, sample_size, 'SU', laser_power)).reset_index()
bkg_SU_All = bkg_df.xs((sample_string, sample_size, 'SU', laser_power)).reset_index()
sig_SU_All = sig_df.xs((sample_string, sample_size, 'SU', laser_power)).reset_index()
snr_SU_All = snr_df.xs((sample_string, sample_size, 'SU', laser_power)).reset_index()
df_SU = df_SU_All[df_SU_All.Date >= sample_date_SU].reset_index()
bkg_SU = bkg_SU_All[bkg_SU_All.Date >= sample_date_SU].reset_index()
sig_SU = sig_SU_All[sig_SU_All.Date >= sample_date_SU].reset_index()
snr_SU = snr_SU_All[snr_SU_All.Date >= sample_date_SU].reset_index()

df_SR_All = df.xs((sample_string, sample_size, 'SR', laser_power)).reset_index()
bkg_SR_All = bkg_df.xs((sample_string, sample_size, 'SR', laser_power)).reset_index()
sig_SR_All = sig_df.xs((sample_string, sample_size, 'SR', laser_power)).reset_index()
snr_SR_All = snr_df.xs((sample_string, sample_size, 'SR', laser_power)).reset_index()
df_SR = df_SR_All[df_SR_All.Date >= sample_date_SR].reset_index()
bkg_SR = bkg_SR_All[bkg_SR_All.Date >= sample_date_SR].reset_index()
sig_SR = sig_SR_All[sig_SR_All.Date >= sample_date_SR].reset_index()
snr_SR = snr_SR_All[snr_SR_All.Date >= sample_date_SR].reset_index()
'''

# view some of the data, use .loc if multiindexed, selecting the rows!
#pf_selected_1 = df_SL_All.loc[(df_SL_All.loc[:, 'Exposure Time (s)']==60) & (df_SL_All.loc[:,'Laser Power (mW)']==25) & (df_SL_All.loc[:,'Date']=='2020-10-01')].loc[:,'30':'826'].values.flatten()
#pf_selected_2 = df_SL_All.loc[(df_SL_All.loc[:, 'Exposure Time (s)']==10) & (df_SL_All.loc[:,'Laser Power (mW)']==25) & (df_SL_All.loc[:,'Date']=='2020-09-14')].loc[:,'30':'826'].values.flatten()
#pf_selected_3 = df_SL_All.loc[(df_SL_All.loc[:, 'Exposure Time (s)']==10) & (df_SL_All.loc[:,'Laser Power (mW)']==10) & (df_SL_All.loc[:,'Date']=='2020-09-14')].loc[:,'30':'826'].values[0].flatten()#.flatten()
#print(pf_selected_1)


# if not just reset the index of the multiindexed data and use as normal
# printing the subset of data we want from the multiindex
print(df_SL)
#print(df_SU)
#print(df_SR)
# compute Mie theory for PSL

'''
PSLs:
Mean    Mean Uncertainty     Size Dist Sigma
600nm     9nm                     10.0nm
701nm     6nm                     9.0nm
800nm     14nm                    5.6nm
903nm     12nm                    4.1nm
1036nm    ????                    ??????
'''
# CRI
n_AS = 1.525
k_AS = 0.00
n_PSL = np.mean(PSL_CRI([350, 405, 532, 663], 300, 1060)[5])
k_PSL = 0.0

# Parameters
mu_sample = 1040
sigma_sample = 10.0
N_sample = 260
wavelength_red = 663

# Mie theory
dp_gaussian = np.arange(1.0, 1200.0, 2.0)
ndp_LN_sample = np.array([Gaussian(x=i, mu=mu_sample, sigma=sigma_sample, N=N_sample) for i in dp_gaussian])
Rad_sample, SL_sample, SR_sample, SU_sample = PMS.SF_SD(m=complex(n_PSL, k_PSL), wavelength=wavelength_red, dp=dp_gaussian, ndp=ndp_LN_sample, nMedium=1.0, space='theta', angularResolution=0.2, normalization=None)
Theta_sample = np.array([(i * 180.0)/pi for i in Rad_sample])

#start and stop for transects
start = 30
stop = 859
delta = stop - start + 1
#'''
# Calibration
# Mie... selecting features
mie_local_max = argrelmax(SL_sample, axis=0, order=50)
mie_local_min = argrelmin(SL_sample, axis=0, order=50)
mie_local_features = np.sort(np.concatenate((mie_local_max, mie_local_min), axis=None))
# delete a feature if needed
#mie_local_features = np.delete(mie_local_features, [0])
mie_local_features_theta = [Theta_sample[x] for x in mie_local_features]
print('Number of Mie Features: ', len(mie_local_features_theta))
print('Mie Features at Theta: ', mie_local_features_theta)
mie_local_features_intensity = [SL_sample[x] for x in mie_local_features]
#Measurement... select features
df_row_number = int(9)
calibration_conditions = np.array(df_SL.loc[df_row_number, 'Sample':'Time'])
print(calibration_conditions)
calibration_measurement = np.array(df_SL.loc[df_row_number, str(start):str(stop)])
col_transects = np.arange(start, stop + 1, 1)
#calibration_measurement_savgol = savgol_filter(calibration_measurement, window_length=51, polyorder=2, deriv=0)
calibration_measurement_local_max = argrelmax(calibration_measurement, axis=0, order=20)
calibration_measurement_local_min = argrelmin(calibration_measurement, axis=0, order=20)
calibration_measurement_local_features = np.sort(np.concatenate((calibration_measurement_local_max, calibration_measurement_local_min), axis=None))
# delete a feature if needed
calibration_measurement_local_features = np.delete(calibration_measurement_local_features, [0, 10, 11])
calibration_measurement_local_features_transects = [col_transects[x] for x in calibration_measurement_local_features]
calibration_measurement_local_features_intensities = [calibration_measurement[x] for x in calibration_measurement_local_features]
print('Number of Measurement Features: ', len(calibration_measurement_local_features_transects))
print('Measurement Features at Transects: ', calibration_measurement_local_features_transects)
# do a linear regression to scattering angles vs profile numbers
# we did linegress just to check to see if the OLS was right!
calibration_measurement_local_features_transects_w_constant = sm.add_constant(calibration_measurement_local_features_transects) # adding the Rayleigh profile numbers will need to be removed after the real PNs are saved by the labview code
model_ols = sm.OLS(mie_local_features_theta, calibration_measurement_local_features_transects_w_constant)
results_ols = model_ols.fit()
print(results_ols.summary())
scaling_factor = np.mean(np.divide(mie_local_features_intensity, calibration_measurement_local_features_intensities))
slope = results_ols.params[1]
intercept = results_ols.params[0]
theta_meas = np.array([(slope * i) + intercept for i in col_transects])


f_cal, ax_cal = plt.subplots(nrows=2, ncols=2, figsize=(36, 18))
ax_cal[0, 0].semilogy(Theta_sample, SL_sample, ls='-', color='red', label='Mie Theory')
ax_cal[0, 0].semilogy(mie_local_features_theta, mie_local_features_intensity, marker='o', ls=' ', color='black', label='Mie Theory Local Max & Min')
ax_cal[0, 0].set_xlabel('\u03b8')
ax_cal[0, 0].set_ylabel('Intensity')
ax_cal[0, 0].set_title('Mie Theory')
ax_cal[0, 0].grid(True)
ax_cal[0, 0].legend(loc=1)
ax_cal[0, 1].semilogy(col_transects, calibration_measurement, ls='-', color='green', label='Uncalibrated')
#ax_cal[0, 1].semilogy(col_transects, calibration_measurement_savgol, ls='-', color='blue', label='Uncalibrated Saviztky-Golay')
ax_cal[0, 1].semilogy(calibration_measurement_local_features_transects, calibration_measurement_local_features_intensities, marker='o', ls=' ', color='black', label='Meas. Local Features\n' + str(calibration_conditions))
ax_cal[0, 1].set_xlabel('Image Transects')
ax_cal[0, 1].set_ylabel('Intensity')
ax_cal[0, 1].set_title('Uncalibrated Measurement')
ax_cal[0, 1].grid(True)
ax_cal[0, 1].legend(loc=1)
ax_cal[1, 0].plot(calibration_measurement_local_features_transects, results_ols.fittedvalues, ls='-', color='fuchsia', label='OLS: y = ' + str('{:.4f}'.format(slope)) + 'x + ' + str('{:.4f}'.format(intercept)))
ax_cal[1, 0].plot(calibration_measurement_local_features_transects, mie_local_features_theta, marker='o', color='black')
ax_cal[1, 0].set_xlabel('Image Transects')
ax_cal[1, 0].set_ylabel('\u03b8')
ax_cal[1, 0].set_title('OLS')
ax_cal[1, 0].grid(True)
ax_cal[1, 0].legend(loc=1)
ax_cal[1, 1].semilogy(Theta_sample, SL_sample, ls='-', color='red', label='Mie Theory')
ax_cal[1, 1].semilogy(theta_meas, calibration_measurement * scaling_factor, ls='-', color='green', label='Calibrated & Scaled to Mie Theory\n' + 'scaling factor: ' + str('{:.3f}'.format(scaling_factor)))
#ax_cal[1, 1].semilogy(theta_meas, calibration_measurement_savgol * scaling_factor, ls='-', color='blue', label='Calibrated Saviztky-Golay & Scaled to Mie Theory')
ax_cal[1, 1].set_xlabel('\u03b8')
ax_cal[1, 1].set_ylabel('Intensity')
ax_cal[1, 1].set_title('Calibrated Measurement')
ax_cal[1, 1].grid(True)
ax_cal[1, 1].legend(loc=1)
plt.savefig(save_directory + '/Calibrated_Measurement.pdf', format='pdf')
plt.savefig(save_directory + '/Calibrated_Measurement.png', format='png')
plt.show()
#'''


# pchip, normalization, and scaling
# repeat creation of theta via repeating setting slope and intercept as constants
#slope = 0.2107
#intercept = -4.5402
#col_transects = np.arange(start, stop + 1, 1)
theta_meas = np.array([(slope * i) + intercept for i in col_transects])
SL_sample_pchip = pchip_interpolate(xi=Theta_sample, yi=SL_sample, x=theta_meas)
SU_sample_pchip = pchip_interpolate(xi=Theta_sample, yi=SU_sample, x=theta_meas)
SR_sample_pchip = pchip_interpolate(xi=Theta_sample, yi=SR_sample, x=theta_meas)

#'''
SL_sample_norm = SL_sample_pchip / np.sum(SL_sample_pchip)
SU_sample_norm = SU_sample_pchip / np.sum(SU_sample_pchip)
SR_sample_norm = SR_sample_pchip / np.sum(SR_sample_pchip)
#'''
# scaling all the data prior to normalization
scalar_list_sl = []
for idx, row in df_SL.iterrows():
    pf_meas = np.array(row[str(start):str(stop)]).astype(float)
    pf_mie = SL_sample_pchip
    scalar = pf_mie[500] / pf_meas[500]
    scalar_list_sl.append(scalar)


df_SL['Scalar'] = scalar_list_sl
print(df_SL['Scalar'])

'''
scalar_list_su = []
for idx, row in df_SU.iterrows():
    pf_meas = np.array(row[str(start):str(stop)]).astype(float)
    pf_mie = SU_sample_pchip
    scalar = pf_mie[500] / pf_meas[500]
    scalar_list_su.append(scalar)


df_SU['Scalar'] = scalar_list_su


scalar_list_sr = []
for idx, row in df_SR.iterrows():
    pf_meas = np.array(row[str(start):str(stop)]).astype(float)
    pf_mie = SR_sample_pchip
    scalar = pf_mie[500] / pf_meas[500]
    scalar_list_sr.append(scalar)


df_SR['Scalar'] = scalar_list_sr
'''

# plot font size
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 18
legend_properties = {'weight':'bold'}
'''
# Geoff needs these comparisons for some reason
f_geoff, ax_geoff = plt.subplots(nrows=1, ncols=2, figsize=(36, 18))
ax_geoff[0].semilogy(theta_meas, pf_selected_1, color='red', ls='-', label='PSL 900 SL Exposure:60s, Laser mW:25mW')
ax_geoff[0].semilogy(theta_meas, pf_selected_2, color='blue', ls='-', label='PSL 900 SL Exposure:10s, Laser mW:25mW')
ax_geoff[0].semilogy(theta_meas, pf_selected_3, color='green', ls='-', label='PSL 900 SL Exposure:10s, Laser mW:10mW')
ax_geoff[0].semilogy(theta_meas, SL900_pchip, color='black', ls='-', linewidth=4, label='Normalized theory')
ax_geoff[0].set_ylabel('Intensity (DN)')
ax_geoff[0].set_xlabel('Degrees')
ax_geoff[0].set_title('PSL SL 900nm High Exposure vs. Low Exposure')
ax_geoff[0].grid(True)
ax_geoff[0].legend(loc=1)
ax_geoff[1].semilogy(theta_meas, Normalization(pf_selected_1), color='red', ls='-', label='PSL 900 SL Exposure:60s, Laser mW:25mW')
ax_geoff[1].semilogy(theta_meas, Normalization(pf_selected_2), color='blue', ls='-', label='PSL 900 SL Exposure:10s, Laser mW:25mW')
ax_geoff[1].semilogy(theta_meas, Normalization(pf_selected_3), color='green', ls='-', label='PSL 900 SL Exposure:10s, Laser mW:10mW')
ax_geoff[1].semilogy(theta_meas, SL900_norm, color='black', ls='-', linewidth=4, label='Normalized theory')
ax_geoff[1].set_ylabel('Intensity (DN)')
ax_geoff[1].set_xlabel('Degrees')
ax_geoff[1].set_title('Normalized PSL SL 900nm High Exposure vs. Low Exposure')
ax_geoff[1].grid(True)
ax_geoff[1].legend(loc=1)
plt.savefig(save_directory + '/SL_High_v_Low.pdf', format='pdf')
plt.savefig(save_directory + '/SL_High_v_Low.png', format='png')
plt.show()
'''


# non-stitched data evaluation
pf_nonstitched_SL_norm_list = []
m_nonstitch_list_SL = []
corr_coeff_nonstitch_list_SL = []
label_nonstitch_list_SL = []
f_sl, ax_sl = plt.subplots(nrows=1, ncols=2, figsize=(36, 10))
for idx, row in df_SL.iterrows():
    #print(np.isscalar(row.loc['Scalar']))
    pf_SL = row.loc['Scalar'] * np.array(row.loc[str(start):str(stop)]).astype(float)
    pf_SL_norm = Normalization(pf_SL)
    #m = M(pf_SL_norm, SL600_norm)
    m = M(pf_SL, SL_sample_pchip)
    #corr_coeff_nonstitch_SL = np.corrcoef(pf_SL_norm, SL600_norm)[0][1]
    #print('Nan values?: ', np.isnan(SL_sample_pchip))
    # we found that for some awful reason the measurement produced nan values!
    corr_coeff_nonstitch_SL = pearsonr(pf_SL, SL_sample_pchip)[0]
    print('correlation coefficient: ', corr_coeff_nonstitch_SL, len(pf_SL)==len(SL_sample_pchip), np.isnan(SL_sample_pchip).any(), np.isnan(pf_SL).any())
    label_string_nonstitch_SL = str(np.array(df_SL.loc[idx, 'Date':'Time']))
    #pf_nonstitched_SL_norm_list.append(pf_SL_norm)
    pf_nonstitched_SL_norm_list.append(pf_SL)
    m_nonstitch_list_SL.append(m)
    corr_coeff_nonstitch_list_SL.append(corr_coeff_nonstitch_SL)
    label_nonstitch_list_SL.append(label_string_nonstitch_SL)
    #ax_sl[0].semilogy(theta_meas, pf_SL_norm, ls='-', label=label_string_nonstitch_SL)
    ax_sl[0].semilogy(theta_meas, pf_SL, ls='-', label=label_string_nonstitch_SL)
#ax_sl[0].semilogy(theta_meas, SL600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_sl[0].semilogy(theta_meas, SL_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_sl[0].set_title('SL Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax_sl[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sl[0].set_ylabel('Intensity (DN)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sl[0].grid(True)
df_SL['Residuals'] = np.array(m_nonstitch_list_SL)
df_SL['Correlation Coefficient'] = np.array(corr_coeff_nonstitch_list_SL)
m_nonstitch_array_SL = np.array(m_nonstitch_list_SL)
m_min_val_nonstitch_SL = np.amin(m_nonstitch_list_SL)
m_min_idx_nonstitch_SL = np.argmin(m_nonstitch_list_SL)
ax_sl[1].semilogy(theta_meas, pf_nonstitched_SL_norm_list[m_min_idx_nonstitch_SL], ls='-', color='red', label=str(label_nonstitch_list_SL[m_min_idx_nonstitch_SL]) + '\n correlation coefficient: ' + str(corr_coeff_nonstitch_list_SL[m_min_idx_nonstitch_SL]) + '\n minimum residual sum: ' + str(m_nonstitch_list_SL[m_min_idx_nonstitch_SL]))
#ax_sl[1].semilogy(theta_meas, SL600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_sl[1].semilogy(theta_meas, SL_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_sl[1].set_title('Best SL Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax_sl[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sl[1].set_ylabel('Intensity (DN)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sl[1].grid(True)
ax_sl[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/SL_nonstitch.pdf', format='pdf')
plt.savefig(save_directory + '/SL_nonstitch.png', format='png')
plt.show()

'''
pf_nonstitched_SU_norm_list = []
m_nonstitch_list_SU = []
corr_coeff_nonstitch_list_SU = []
label_nonstitch_list_SU = []
f_su, ax_su = plt.subplots(nrows=1, ncols=2, figsize=(36, 10))
for idx, row in df_SU.iterrows():
    pf_SU = row['Scalar'] * np.array(row.loc[str(start):str(stop)]).astype(float)
    pf_SU_norm = Normalization(pf_SU)
    #m = M(pf_SU_norm, SU600_norm)
    m = M(pf_SU, SU_sample_pchip)
    #corr_coeff_nonstitch_SU = np.corrcoef(pf_SU_norm, SU600_norm)[0][1]
    corr_coeff_nonstitch_SU = np.corrcoef(pf_SU, SU_sample_pchip)[0][1]
    label_string_nonstitch_SU = str(np.array(df_SU.loc[idx, 'Date':'Time']))
    #pf_nonstitched_SU_norm_list.append(pf_SU_norm)
    pf_nonstitched_SU_norm_list.append(pf_SU)
    m_nonstitch_list_SU.append(m)
    corr_coeff_nonstitch_list_SU.append(corr_coeff_nonstitch_SU)
    label_nonstitch_list_SU.append(label_string_nonstitch_SU)
    #ax_su[0].semilogy(theta_meas, pf_SU_norm, ls='-', label=label_string_nonstitch_SU)
    ax_su[0].semilogy(theta_meas, pf_SU, ls='-', label=label_string_nonstitch_SU)
#ax_su[0].semilogy(theta_meas, SU600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_su[0].semilogy(theta_meas, SU_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_su[0].set_title('SU Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax_su[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_su[0].set_ylabel('Intensity (DN)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_su[0].grid(True)
df_SU['Residuals'] = np.array(m_nonstitch_list_SU)
df_SU['Correlation Coefficient'] = np.array(corr_coeff_nonstitch_list_SU)
m_nonstitch_array_SU = np.array(m_nonstitch_list_SU)
m_min_val_nonstitch_SU = np.amin(m_nonstitch_list_SU)
m_min_idx_nonstitch_SU = np.argmin(m_nonstitch_list_SU)
ax_su[1].semilogy(theta_meas, pf_nonstitched_SU_norm_list[m_min_idx_nonstitch_SU], ls='-', color='green', label=str(label_nonstitch_list_SU[m_min_idx_nonstitch_SU]) + '\n correlation coefficient: ' + str(corr_coeff_nonstitch_list_SU[m_min_idx_nonstitch_SU]) + '\n minimum residual sum: ' + str(m_nonstitch_list_SU[m_min_idx_nonstitch_SU]))
#ax_su[1].semilogy(theta_meas, SU600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_su[1].semilogy(theta_meas, SU_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_su[1].set_title('Best SU Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax_su[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_su[1].set_ylabel('Intensity (DN)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_su[1].grid(True)
ax_su[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/SU_nonstitch.pdf', format='pdf')
plt.savefig(save_directory + '/SU_nonstitch.png', format='png')
plt.show()


pf_nonstitched_SR_norm_list = []
m_nonstitch_list_SR = []
corr_coeff_nonstitch_list_SR = []
label_nonstitch_list_SR = []
f_sr, ax_sr = plt.subplots(nrows=1, ncols=2, figsize=(36, 10))
for idx, row in df_SR.iterrows():
    pf_SR = row['Scalar'] * np.array(row.loc[str(start):str(stop)]).astype(float)
    pf_SR_norm = Normalization(pf_SR)
    #m = M(pf_SR_norm, SR600_norm)
    m = M(pf_SR, SR_sample_pchip)
    #corr_coeff_nonstitch_SR = np.corrcoef(pf_SR_norm, SR600_norm)[0][1]
    corr_coeff_nonstitch_SR = np.corrcoef(pf_SR, SR_sample_pchip)[0][1]
    label_string_nonstitch_SR = str(np.array(df_SR.loc[idx, 'Date':'Time']))
    #pf_nonstitched_SR_norm_list.append(pf_SR_norm)
    pf_nonstitched_SR_norm_list.append(pf_SR)
    m_nonstitch_list_SR.append(m)
    corr_coeff_nonstitch_list_SR.append(corr_coeff_nonstitch_SR)
    label_nonstitch_list_SR.append(label_string_nonstitch_SR)
    #ax_sr[0].semilogy(theta_meas, pf_SR_norm, ls='-', label=label_string_nonstitch_SR)
    ax_sr[0].semilogy(theta_meas, pf_SR, ls='-', label=label_string_nonstitch_SR)
#ax_sr[0].semilogy(theta_meas, SR600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax_sr[0].semilogy(theta_meas, SR_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_sr[0].set_title('SR Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax_sr[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sr[0].set_ylabel('Intensity (DN)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sr[0].grid(True)
df_SR['Residuals'] = np.array(m_nonstitch_list_SR)
df_SR['Correlation Coefficient'] = np.array(corr_coeff_nonstitch_list_SR)
m_nonstitch_array_SR = np.array(m_nonstitch_list_SR)
m_min_val_nonstitch_SR = np.amin(m_nonstitch_list_SR)
m_min_idx_nonstitch_SR = np.argmin(m_nonstitch_list_SR)
ax_sr[1].semilogy(theta_meas, pf_nonstitched_SR_norm_list[m_min_idx_nonstitch_SR], ls='-', color='blue', label=str(label_nonstitch_list_SR[m_min_idx_nonstitch_SR]) + '\n correlation coefficient: ' + str(corr_coeff_nonstitch_list_SR[m_min_idx_nonstitch_SR]) + '\n minimum residual sum: ' + str(m_nonstitch_list_SR[m_min_idx_nonstitch_SR]))
#ax_sr[1].semilogy(theta_meas, SR600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax_sr[1].semilogy(theta_meas, SR_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax_sr[1].set_title('Best SR Measurement', fontsize=BIGGER_SIZE, fontweight='bold')
ax_sr[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sr[1].set_ylabel('Intensity (DN)', fontsize=MEDIUM_SIZE, fontweight='bold')
ax_sr[1].grid(True)
ax_sr[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)
plt.savefig(save_directory + '/SR_nonstitch.pdf', format='pdf')
plt.savefig(save_directory + '/SR_nonstitch.png', format='png')
plt.show()
'''

# show image stitching threshold
intensity_stitch = 8E4
intensity_stitch_array = np.repeat(intensity_stitch, repeats=len(theta_meas))
intensity_stitch_array_norm = Normalization(intensity_stitch_array)
intensity_stitch_norm = intensity_stitch_array_norm[0]
PF1_2dlist = []
PF2_2dlist = []
sep_idx = []
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(36, 10))
for index, row in df_SL.iterrows():
    #print(np.array(row))
    pf_SL = row.loc['Scalar'] * np.array(row.loc[str(start):str(stop)]).astype(float)
    ax[0].semilogy(theta_meas, pf_SL, ls='-', label=np.array(row.loc['Date':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf_SL), ls='-', label=np.array(row.loc['Date':'Time']))
ax[0].semilogy(theta_meas, SL_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].grid(True)
#ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SL_sample_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax[1].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax[1].set_title('Normalized Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].grid(True)
#ax[1].legend(loc=1)
plt.savefig(save_directory + '/measurements_made_SL.pdf', format='pdf')
plt.savefig(save_directory + '/measurements_made_SL.png', format='png')
plt.show()

'''
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(36, 10))
for index, row in df_SU.iterrows():
    #print(np.array(row))
    pf_SU = row.loc['Scalar'] * np.array(row.loc[str(start):str(stop)]).astype(float)
    ax[0].semilogy(theta_meas, pf_SU, ls='-', label=np.array(row.loc['Date':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf_SU), ls='-', label=np.array(row.loc['Date':'Time']))
ax[0].semilogy(theta_meas, SU_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].grid(True)
#ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SU_sample_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
ax[1].semilogy(theta_meas, intensity_stitch_array_norm, color='black', ls='-', label='Intensity Threshold')
ax[1].set_title('Normalized Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[1].grid(True)
#ax[1].legend(loc=1)
plt.savefig(save_directory + '/measurements_made_SU.pdf', format='pdf')
plt.savefig(save_directory + '/measurements_made_SU.png', format='png')
plt.show()


f, ax = plt.subplots(nrows=1, ncols=2, figsize=(36, 10))
for index, row in df_SR.iterrows():
    #print(np.array(row))
    pf_SR = row.loc['Scalar'] * np.array(row.loc[str(start):str(stop)]).astype(float)
    ax[0].semilogy(theta_meas, pf_SR, ls='-', label=np.array(row.loc['Date':'Time']))
    ax[1].semilogy(theta_meas, Normalization(pf_SR), ls='-', label=np.array(row.loc['Date':'Time']))
ax[0].semilogy(theta_meas, SR_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax[0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax[0].set_title('Measurements & Theory', fontsize=BIGGER_SIZE, fontweight='bold')
ax[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax[0].grid(True)
#ax[0].legend(loc=1)
ax[1].semilogy(theta_meas, SR_sample_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm Norm.')
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
    pf1_SL = np.zeros(delta)
    snr1_SL = np.zeros(delta)
    pf2_details_SL = []
    pf2_SL =np.zeros(delta)
    snr2_SL = np.zeros(delta)
    pf_SL = row['Scalar'] * np.array(row[str(start):str(stop)]).astype(float)
    #pf_norm_SL = Normalization(pf_SL)
    pf_norm_SL = pf_SL
    for element in pf_norm_SL:
        if element >= intensity_stitch:
            pf1_SL[counter] = element
            snr1_SL[counter] = snr_SL.loc[index, str(30 + counter)]
        if element < intensity_stitch:
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
'''
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
    pf_SU = row['Scalar'] * np.array(row[str(start):str(stop)]).astype(float)
    #pf_norm_SU = Normalization(pf_SU)
    pf_norm_SU = pf_SU
    for element in pf_norm_SU:
        if element >= intensity_stitch:
            pf1_SU[counter] = element
            snr1_SU[counter] = snr_SU.loc[index, str(30 + counter)]
        if element < intensity_stitch:
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
    pf_SR = row['Scalar'] * np.array(row[str(start):str(stop)]).astype(float)
    #pf_norm_SR = Normalization(pf_SR)
    pf_norm_SR = pf_SR
    for element in pf_norm_SR:
        if element >= intensity_stitch:
            pf1_SR[counter] = element
            snr1_SR[counter] = snr_SR.loc[index, str(30 + counter)]
        if element < intensity_stitch:
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
'''

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
        fp_trim = np.trim_zeros(element)
        lp_trim = np.trim_zeros(element2)
        fp_lastval = fp_trim[-1]
        lp_firstval = lp_trim[0]
        scale = fp_lastval/lp_firstval
        fp_size = fp_trim.size
        lp_size = lp_trim.size
        overlap = (lp_size + fp_size) - delta
        #print(overlap)
        #pf_combos_SL = np.append(fp_trim, scale * np.array(pf2_2dlist_SL[counter2])[overlap:-1])
        #snr_combos_SL = np.add(snr1_2darray_SL[counter], snr2_2darray_SL[counter2][overlap:-1])
        if overlap > 0:
            pf_combos_SL = np.append(fp_trim, scale * lp_trim[overlap - 1:-1])
            #print(pf_combos_SL.size)
            snr_combos_SL = np.append(np.trim_zeros(snr1_2darray_SL[counter]), np.trim_zeros(snr2_2darray_SL[counter2])[overlap - 1:-1])
        if overlap < 0:
            pf_combos_SL = np.append(fp_trim, np.zeros(np.absolute(overlap)))
            pf_combos_SL = np.append(pf_combos_SL, scale * lp_trim)
            #print(pf_combos_SL.size)
            snr_combos_SL = np.append(np.trim_zeros(snr1_2darray_SL[counter]), np.zeros(np.absolute(overlap)))
            snr_combos_SL = np.append(snr_combos_SL, np.trim_zeros(snr2_2darray_SL[counter2]))
        if overlap == 0:
            pf_combos_SL = np.append(fp_trim, scale * lp_trim)
            #print(pf_combos_SL.size)
            snr_combos_SL = np.append(np.trim_zeros(snr1_2darray_SL[counter]), np.trim_zeros(snr2_2darray_SL[counter2]))
        pf_combos_SL[pf_combos_SL == 0] = np.nan
        nan_bool_SL = np.isnan(pf_combos_SL)
        for counter3, element3 in enumerate(nan_bool_SL):
            if element3 == True:
                drop_nan_idx_SL.append(counter3)
        if any(nan_bool_SL) == True:
            pf_combos_dropnan = np.delete(pf_combos_SL, drop_nan_idx_SL)
            snr_combos_dropnan = np.delete(snr_combos_SL, drop_nan_idx_SL)
            theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx_SL)
            #SL_norm_dropnan = np.delete(SL600_norm, drop_nan_idx_SL)
            SL_norm_dropnan = np.delete(SL_sample_pchip, drop_nan_idx_SL)
        else:
            pf_combos_dropnan = pf_combos_SL
            snr_combos_dropnan = snr_combos_SL
            theta_meas_dropnan = theta_meas
            #SL_norm_dropnan = SL600_norm
            SL_norm_dropnan = SL_sample_pchip
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
#ax1[0, 0].semilogy(theta_meas, SL600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 0].semilogy(theta_meas, SL_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax1[0, 0].set_title('SL Normalized Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].grid(True)
ax1[1, 0].set_title('SL SNR Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].grid(True)
ax1[0, 1].semilogy(theta_list_SL[m_min_idx_SL], pf_combos_norm_list_SL[m_min_idx_SL], ls='-', color='red', label=str(label_list_SL[m_min_idx_SL]) + '\n correlation coefficient: ' + str(corr_coeff_list_SL[m_min_idx_SL]) + '\n minimum residual sum: ' + str(m_list_SL[m_min_idx_SL]))
variable_sl = 12
#ax1[0, 1].semilogy(theta_list_SL[variable_sl], pf_combos_norm_list_SL[variable_sl], ls='-', color='magenta', label=str(label_list_SL[variable_sl]) + '\n correlation coefficient: ' + str(corr_coeff_list_SL[variable_sl]) + '\n minimum residual sum: ' + str(m_list_SL[variable_sl]))
#ax1[0, 1].semilogy(theta_meas, SL600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 1].semilogy(theta_meas, SL_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
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
df_SL_stitched.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SL600.txt', sep=',', header=True, index=False)
df_SL_stitched_theta.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SL600_theta.txt', sep=',', header=True, index=False)
#print(df_SL_stitched)

'''
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
        fp_trim = np.trim_zeros(element)
        lp_trim = np.trim_zeros(element2)
        fp_lastval = fp_trim[-1]
        lp_firstval = lp_trim[0]
        scale = fp_lastval / lp_firstval
        fp_size = fp_trim.size
        lp_size = lp_trim.size
        overlap = (lp_size + fp_size) - 797
        # print(overlap)
        # pf_combos_SL = np.append(fp_trim, scale * np.array(pf2_2dlist_SL[counter2])[overlap:-1])
        # snr_combos_SL = np.add(snr1_2darray_SL[counter], snr2_2darray_SL[counter2][overlap:-1])
        if overlap > 0:
            pf_combos_SU = np.append(fp_trim, scale * lp_trim[overlap - 1:-1])
            # print(pf_combos_SL.size)
            snr_combos_SU = np.append(np.trim_zeros(snr1_2darray_SU[counter]), np.trim_zeros(snr2_2darray_SU[counter2])[overlap - 1:-1])
        if overlap < 0:
            pf_combos_SU = np.append(fp_trim, np.zeros(np.absolute(overlap)))
            pf_combos_SU = np.append(pf_combos_SU, scale * lp_trim)
            # print(pf_combos_SL.size)
            snr_combos_SU = np.append(np.trim_zeros(snr1_2darray_SU[counter]), np.zeros(np.absolute(overlap)))
            snr_combos_SU = np.append(snr_combos_SU, np.trim_zeros(snr2_2darray_SU[counter2]))
        if overlap == 0:
            pf_combos_SU = np.append(fp_trim, scale * lp_trim)
            # print(pf_combos_SL.size)
            snr_combos_SU = np.append(np.trim_zeros(snr1_2darray_SU[counter]), np.trim_zeros(snr2_2darray_SU[counter2]))
        pf_combos_SU[pf_combos_SU == 0] = np.nan
        nan_bool_SU = np.isnan(pf_combos_SU)
        for counter3, element3 in enumerate(nan_bool_SU):
            if element3 == True:
                drop_nan_idx_SU.append(counter3)
        if any(nan_bool_SU) == True:
            pf_combos_dropnan = np.delete(pf_combos_SU, drop_nan_idx_SU)
            snr_combos_dropnan = np.delete(snr_combos_SU, drop_nan_idx_SU)
            theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx_SU)
            #SU_norm_dropnan = np.delete(SU600_norm, drop_nan_idx_SU)
            SU_norm_dropnan = np.delete(SU_sample_pchip, drop_nan_idx_SU)
        else:
            pf_combos_dropnan = pf_combos_SU
            snr_combos_dropnan = snr_combos_SU
            theta_meas_dropnan = theta_meas
            #SU_norm_dropnan = SU600_norm
            SU_norm_dropnan = SU_sample_pchip
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
#ax1[0, 0].semilogy(theta_meas, SU600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL 903nm')
ax1[0, 0].semilogy(theta_meas, SU_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax1[0, 0].set_title('SU Normalized Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].grid(True)
ax1[1, 0].set_title('SU SNR Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].grid(True)
ax1[0, 1].semilogy(theta_list_SU[m_min_idx_SU], pf_combos_norm_list_SU[m_min_idx_SU], ls='-', color='green', label=str(label_list_SU[m_min_idx_SU]) + '\ncorrelation coefficient: ' + str(corr_coeff_list_SU[m_min_idx_SU])+ '\nminimum residual sum: ' + str(m_list_SU[m_min_idx_SU]))
variable_su = 15
#ax1[0, 1].semilogy(theta_list_SU[variable_su], pf_combos_norm_list_SU[variable_su], ls='-', color='magenta', label=str(label_list_SU[variable_su]) + '\ncorrelation coefficient: ' + str(corr_coeff_list_SU[variable_su])+ '\nminimum residual sum: ' + str(m_list_SU[variable_su]))
#ax1[0, 1].semilogy(theta_meas, SU600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 1].semilogy(theta_meas, SU_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
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
df_SU_stitched.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SU600.txt', sep=',', header=True, index=False)
df_SU_stitched_theta.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SU600_theta.txt', sep=',', header=True, index=False)
'''
'''
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
        fp_trim = np.trim_zeros(element)
        lp_trim = np.trim_zeros(element2)
        fp_lastval = fp_trim[-1]
        lp_firstval = lp_trim[0]
        scale = fp_lastval / lp_firstval
        fp_size = fp_trim.size
        lp_size = lp_trim.size
        overlap = (lp_size + fp_size) - 797
        # print(overlap)
        # pf_combos_SL = np.append(fp_trim, scale * np.array(pf2_2dlist_SL[counter2])[overlap:-1])
        # snr_combos_SL = np.add(snr1_2darray_SL[counter], snr2_2darray_SL[counter2][overlap:-1])
        if overlap > 0:
            pf_combos_SR = np.append(fp_trim, scale * lp_trim[overlap - 1:-1])
            # print(pf_combos_SL.size)
            snr_combos_SR = np.append(np.trim_zeros(snr1_2darray_SR[counter]), np.trim_zeros(snr2_2darray_SR[counter2])[overlap - 1:-1])
        if overlap < 0:
            pf_combos_SR = np.append(fp_trim, np.zeros(np.absolute(overlap)))
            pf_combos_SR = np.append(pf_combos_SR, scale * lp_trim)
            # print(pf_combos_SL.size)
            snr_combos_SR = np.append(np.trim_zeros(snr1_2darray_SR[counter]), np.zeros(np.absolute(overlap)))
            snr_combos_SR = np.append(snr_combos_SR, np.trim_zeros(snr2_2darray_SR[counter2]))
        if overlap == 0:
            pf_combos_SR = np.append(fp_trim, scale * lp_trim)
            # print(pf_combos_SL.size)
            snr_combos_SR = np.append(np.trim_zeros(snr1_2darray_SR[counter]), np.trim_zeros(snr2_2darray_SR[counter2]))
        pf_combos_SR[pf_combos_SR == 0] = np.nan
        nan_bool_SR = np.isnan(pf_combos_SR)
        for counter3, element3 in enumerate(nan_bool_SR):
            if element3 == True:
                drop_nan_idx_SR.append(counter3)
        if any(nan_bool_SR) == True:
            pf_combos_dropnan = np.delete(pf_combos_SR, drop_nan_idx_SR)
            snr_combos_dropnan = np.delete(snr_combos_SR, drop_nan_idx_SR)
            theta_meas_dropnan = np.delete(theta_meas, drop_nan_idx_SR)
            #SR_norm_dropnan = np.delete(SR600_norm, drop_nan_idx_SR)
            SR_norm_dropnan = np.delete(SR_sample_pchip, drop_nan_idx_SR)
        else:
            pf_combos_dropnan = pf_combos_SR
            snr_combos_dropnan = snr_combos_SR
            theta_meas_dropnan = theta_meas
            #SR_norm_dropnan = SR600_norm
            SR_norm_dropnan = SR_sample_pchip
        m = M(pf_combos_dropnan, SR_norm_dropnan)
        corr_coeff_SR = np.corrcoef(pf_combos_dropnan, SR_norm_dropnan)[0][1]
        label_string = str(np.array(df_SR.loc[counter, 'Date':'Time'])) + ' &\n' + str(np.array(df_SR.loc[counter2, 'Date':'Time']))
        pf_combos_norm_list_SR.append(pf_combos_dropnan)
        snr_combos_list_SR.append(snr_combos_dropnan)
        m_list_SR.append(m)
        corr_coeff_list_SR.append(corr_coeff_SR)
        theta_list_SR.append(theta_meas_dropnan)
        label_list_SR.append(label_string)
        #3apostraphes
        #col_combos = np.add(np.array(pf1_2dlabels[counter]), np.array(pf2_2dlabels[counter2]))
        #theta_combos = np.array([(slope * i) + intercept for i in col_combos])
        #3apostraphes
        ax1[0, 0].semilogy(theta_meas_dropnan, pf_combos_dropnan, ls='-', label=label_string)
        ax1[1, 0].plot(theta_meas_dropnan, snr_combos_dropnan, label=label_string)
m_list_SR = np.array(m_list_SR)
#print(m_list)
m_min_val_SR = np.amin(m_list_SR)
m_min_idx_SR = np.argmin(m_list_SR)
#ax1[0, 0].semilogy(theta_meas, SR600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 0].semilogy(theta_meas, SR_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 0].semilogy(theta_meas, intensity_stitch_array, color='black', ls='-', label='Intensity Threshold')
ax1[0, 0].set_title('SR Normalized Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[0, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[0, 0].grid(True)
ax1[1, 0].set_title('SR SNR Stitched Measurements', fontsize=BIGGER_SIZE, fontweight='bold')
ax1[1, 0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].set_ylabel('SNR', fontsize=MEDIUM_SIZE, fontweight='bold')
ax1[1, 0].grid(True)
ax1[0, 1].semilogy(theta_list_SR[m_min_idx_SR], pf_combos_norm_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR] + '\ncorrelation coefficient: ' + str(corr_coeff_list_SR[m_min_idx_SR])+ '\nminimum residual sum: ' + str(m_list_SR[m_min_idx_SR]))
variable_sr = 23
#ax1[0, 1].semilogy(theta_list_SR[variable_sr], pf_combos_norm_list_SR[variable_sr], ls='-', color='magenta', label=label_list_SR[variable_sr] + '\ncorrelation coefficient: ' + str(corr_coeff_list_SR[variable_sr])+ '\nminimum residual sum: ' + str(m_list_SR[variable_sr]))
#ax1[0, 1].semilogy(theta_meas, SR600_norm, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
ax1[0, 1].semilogy(theta_meas, SR_sample_pchip, color='black', ls='-', linewidth=4, label='Mie Theory PSL')
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
df_SR_stitched.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SR600.txt', sep=',', header=True, index=False)
df_SR_stitched_theta.to_csv('/home/austen/Desktop/Recent/stitched_PSL_SR600_theta.txt', sep=',', header=True, index=False)
'''
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
'''
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
'''
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
ax2[0].semilogy(theta_meas, SL_sample_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[0].semilogy(theta_meas, SL_TS_pchip_norm, color='coral', ls='-', linewidth=4, label='Theoretical Match: ' + str(label_list_SL[m_min_idx_SL]))
ax2[0].set_title('SL Normalized Stitched Measurement,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax2[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[0].grid(True)
ax2[0].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax2[1].semilogy(theta_list_SU[m_min_idx_SU], pf_combos_norm_list_SU[m_min_idx_SU], ls='-', color='green', label=label_list_SU[m_min_idx_SU])
ax2[1].semilogy(theta_meas, SU_sample_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax2[1].semilogy(theta_meas, SU_TS_pchip_norm, color='lawngreen', ls='-', linewidth=4, label='Theoretical Match: ' + str(label_list_SU[m_min_idx_SU]))
ax2[1].set_title('SU Normalized Stitched Measurement,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax2[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax2[1].grid(True)
ax2[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax2[2].semilogy(theta_list_SR[m_min_idx_SR], pf_combos_norm_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR])
ax2[2].semilogy(theta_meas, SR_sample_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
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
ax3[0].semilogy(theta_meas, SL_sample_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax3[0].semilogy(theta_meas, SL_combo_pchip_norm, color='purple', ls='-', linewidth=4, label='Retrieved Match: \u03bc' + str('{:.3f}'.format(result_SLSRSU_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SLSRSU_stitch.x[1])))
ax3[0].set_title('SL Normalized Stitched Measurements,\n Theory, and Retrieved', fontsize=BIGGER_SIZE, fontweight='bold')
ax3[0].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[0].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[0].grid(True)
ax3[0].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax3[1].semilogy(theta_list_SU[m_min_idx_SU], pf_combos_norm_list_SU[m_min_idx_SU], ls='-', color='green', label=label_list_SU[m_min_idx_SU])
ax3[1].semilogy(theta_meas, SU_sample_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
ax3[1].semilogy(theta_meas, SU_combo_pchip_norm, color='purple', ls='-', linewidth=4, label='Retrieved Match: \u03bc' + str('{:.3f}'.format(result_SLSRSU_stitch.x[0])) + '\u03c3: ' + str('{:.3f}'.format(result_SLSRSU_stitch.x[1])))
ax3[1].set_title('SU Normalized Stitched Measurements,\n Theory, and Retrieved Phase Function', fontsize=BIGGER_SIZE, fontweight='bold')
ax3[1].set_xlabel('Degrees', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[1].set_ylabel('Intensity', fontsize=MEDIUM_SIZE, fontweight='bold')
ax3[1].grid(True)
ax3[1].legend(loc=1, fontsize=SMALL_SIZE, prop=legend_properties)

ax3[2].semilogy(theta_list_SR[m_min_idx_SR], pf_combos_norm_list_SR[m_min_idx_SR], ls='-', color='blue', label=label_list_SR[m_min_idx_SR])
ax3[2].semilogy(theta_meas, SR_sample_norm, color='black', ls='-', linewidth=4, label='Specified Thermo Fisher Scientific PSL Dist.')
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



