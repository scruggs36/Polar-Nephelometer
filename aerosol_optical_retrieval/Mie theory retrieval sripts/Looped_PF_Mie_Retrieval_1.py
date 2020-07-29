'''
Austen K. Scruggs
07/27/2020
Description: Compares normalized measured and calculated phase function, makes pretty figures
'''

import pandas as pd
import numpy as np
import os
import PyMieScatt as PMS
import matplotlib.pyplot as plt
from math import sqrt, pi, log
from scipy.interpolate import pchip_interpolate
from scipy.optimize import least_squares
from matplotlib.gridspec import GridSpec
# directories
file_directory = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2020/2020-07-24/Analysis/Measurements'
save_directory = '/home/austen/Desktop/Recent/'


def LogNormal(size, mu, gsd):
    if size ==0.0:
        return 0.0
    else:
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (1 / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))



def Residuals_SX(x, w_n, SX_M, SX_Theta, bin_edges, polarization):
    # pre allocation
    SL_2darray = []
    # Data
    SX_M = np.array(SX_M)
    SX_M = SX_M[~np.isnan(SX_M)]
    SX_Theta = SX_Theta[~np.isnan(SX_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    bin_counts = [LogNormal(size=i, mu=x[0], gsd=x[1]) for i in bin_edges]

    if polarization == 0:
        theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[2], x[3]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        theta_mie = np.array(theta_mie) * (180.0/pi)
        SR_pchip = pchip_interpolate(xi=theta_mie, yi=SR, x=SX_Theta, der=0, axis=0)
        SR_pchip_norm = SR_pchip / np.sum(SR_pchip)
        # somehow residuals are coming out negative...05/14/2020
        sr_diff = np.absolute((SX_M - (SR_pchip_norm)) / SR_pchip_norm) * 100
        residuals = np.sum(sr_diff)
        print('Guess mu: ', x[0], 'Geometric stdev: ', x[1],'Guess n: ', x[2], 'Guess k: ', x[3], 'Summed Error: ', residuals)

    if polarization == 90:
        theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[2], x[3]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        theta_mie = np.array(theta_mie) * (180.0 / pi)
        SU_pchip = pchip_interpolate(xi=theta_mie, yi=SU, x=SX_Theta, der=0, axis=0)
        SU_pchip_norm = SU_pchip / np.sum(SU_pchip)
        # somehow residuals are coming out negative...05/14/2020
        su_diff = np.absolute((SX_M - (SU_pchip_norm)) / SU_pchip_norm) * 100
        residuals = np.sum(su_diff)
        print('Guess mu: ', x[0], 'Geometric stdev: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3], 'Summed Error: ', residuals)

    if polarization == 180:
        theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[2], x[3]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        theta_mie = np.array(theta_mie) * (180.0 / pi)
        SL_pchip = pchip_interpolate(xi=theta_mie, yi=SL, x=SX_Theta, der=0, axis=0)
        SL_pchip_norm = SL_pchip / np.sum(SL_pchip)
        # somehow residuals are coming out negative...05/14/2020
        sl_diff = np.absolute((SX_M - (SL_pchip_norm)) / SL_pchip_norm) * 100
        residuals = np.sum(sl_diff)
        print('Guess mu: ', x[0], 'Geometric stdev: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3], 'Summed Error: ', residuals)
    return residuals



#'''
# read file names in directory
file_list = os.listdir(file_directory)

# loop append to file
type_list = []
size_list = []
pol_list = []
riemann_list = []
gfit_list = []
mt_fit_pf = []
angles_list = []
mu_list = []
sigma_list = []
n_list = []
k_list = []
SX_list = []
theta_list = []
# temporary array to be cleared at end of loop
Cext_arr = []
Csca_arr = []
Cabs_arr = []
g_arr = []
Cpr_arr = []
Cback_arr = []
Cratio_arr = []
# not temporary
dbins_list = []
nbins_list = []
Cext_list = []
Csca_list = []
Cabs_list = []
g_list = []
Cpr_list = []
Cback_list = []
Cratio_list = []
for file in file_list:
    print(file)
    name_parsed = file.split('_')
    # parameters are based on the most recent calibration
    slope = .2095
    intercept = -3.1433
    wavelength = 663
    # data to plug into dataframe (compilation of measurements)
    type = name_parsed[2]
    size = float(name_parsed[3])
    pol = float(name_parsed[4].split('.')[0])
    size_bins = np.linspace(size - 100, size + 100, 201)
    data = pd.read_csv(file_directory + '/' + file, sep=',', header=0)
    riemann = np.array(data['Sample Intensity'])
    gfit = np.array(data['Sample Intensity gfit'])
    angles = np.array([slope * i + intercept for i in data['Sample Columns']])
    type_list.append(type)
    size_list.append(size)
    pol_list.append(pol)
    riemann_list.append(riemann)
    gfit_list.append(gfit)
    angles_list.append(angles)
    #'''
    # fit each phase function to Mie theory, adapt this bit of code...
    result = least_squares(Residuals_SX, x0=[size, 1.05, 1.52, 0.001], method='trf', args=(wavelength, riemann, angles, size_bins, pol), bounds=([1.0, 1.000, 1.000, 0.000], [1000.0, 1.500, 2.000, 1.000]))
    # append solutions
    mu_list.append(result.x[0])
    sigma_list.append(result.x[1])
    n_list.append(result.x[2])
    k_list.append(result.x[3])
    # return normalized SL mie and parameters
    theta_mie, SL_mie, SR_mie, SU_mie = PMS.SF_SD(m=complex(result.x[2], result.x[3]), wavelength=wavelength, dp=size_bins, ndp=[LogNormal(size=i, mu=result.x[0], gsd=result.x[1]) for i in size_bins], nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_mie = np.array(theta_mie) * (180.0/pi)
    SL_mie_norm = SL_mie / np.sum(SL_mie)
    SR_mie_norm = SR_mie / np.sum(SR_mie)
    SU_mie_norm = SU_mie / np.sum(SU_mie)

    if pol == float(0):
        SX_list.append(SL_mie_norm)

    if pol == float(90):
        SX_list.append(SU_mie_norm)

    if pol == float(180):
        SX_list.append(SR_mie_norm)

    theta_list.append(theta_mie)
    dbins = size_bins
    nbins = [LogNormal(i, result.x[0], result.x[1]) for i in size_bins]
    for size in size_bins:
        optical_params = PMS.MieQ(m=complex(result.x[2], result.x[3]), wavelength=wavelength, diameter=size, nMedium=1.0, asDict=True, asCrossSection=True)
        # 1E-14 cm^2 in 1 nm^2, so we are converting cross-section units to cm^2
        Cext_arr.append(optical_params['Cext'] * 1E-14)
        Csca_arr.append(optical_params['Csca'] * 1E-14)
        Cabs_arr.append(optical_params['Cabs'] * 1E-14)
        g_arr.append(optical_params['g'])
        Cpr_arr.append(optical_params['Cpr'] * 1E-14)
        Cback_arr.append(optical_params['Cback'] * 1E-14)
        Cratio_arr.append(optical_params['Cratio'] * 1E-14)
    Cext = np.average(Cext_arr, weights=nbins)
    Csca = np.average(Csca_arr, weights=nbins)
    Cabs = np.average(Cabs_arr, weights=nbins)
    g = np.average(g_arr, weights=nbins)
    Cpr = np.average(Cpr_arr, weights=nbins)
    Cback = np.average(Cback_arr, weights=nbins)
    Cratio = np.average(Cratio_arr, weights=nbins)
    dbins_list.append(dbins)
    nbins_list.append(nbins)
    Cext_list.append(Cext)
    Csca_list.append(Csca)
    Cabs_list.append(Cabs)
    g_list.append(g)
    Cpr_list.append(Cpr)
    Cback_list.append(Cback)
    Cratio_list.append(Cratio)
    Cext_arr = []
    Csca_arr = []
    Cabs_arr = []
    g_arr = []
    Cpr_arr = []
    Cback_arr = []
    Cratio_arr = []
    #'''


# create dataframe
header = ['Size (nm)', 'Polarization (deg)', 'Aerosol Sample', 'Riemann', 'Gaussian', 'Angles (deg)', 'Mie Fit PF', 'dbins', 'nbins', 'mu', 'sigma', 'n', 'k', 'Cext', 'Csca', 'Cabs', 'g', 'Cpr', 'Cback', 'Cratio']
data_ndarray = np.transpose(np.array([size_list, pol_list, type_list, riemann_list, gfit_list, angles_list, SX_list, dbins_list, nbins_list, mu_list, sigma_list, n_list, k_list, Cext_list, Csca_list, Cabs_list, g_list, Cpr_list, Cback_list, Cratio_list]))
df = pd.DataFrame(data_ndarray, columns=header)
# only use this on starting to compile measured data, basically starting a new file
#meas.to_csv(save_directory + 'AS_Measurements.txt', sep=',', header=True, index=False)
# use this to append to exisiting file
df.to_csv(save_directory + 'AS_DF.txt', sep=',')
#'''

# read file in with all its data
# had to put the index bit in beause I did not specify elimination of the index in the creation of the dataframe, I've
# added the index=False bit just recently 07/28/2020
#'''
comp = pd.read_csv(save_directory + 'AS_DF.txt', sep=',', header=0, index_col=0)
#print(comp)
# don't need this,  but used to...had to add this bit in because I didn't choose not to index in the making of the DF
#comp.reset_index(drop=True, inplace=True)
#print(comp)


# gotta do some Multiindexing in Pandas, so we are grouping data by the size and polarization
# to note the result of the Multiindex is a series???
# this is actually multiindexing the data had to upgrade pandas to 1.0.1
#comp_multiindexed = pd.MultiIndex.from_frame(comp)

# okay this worked, set the index to the two columns, then called the values I needed, this worked perfectly!
comp.set_index(['Size (nm)', 'Polarization (deg)'], inplace=True)
#print(comp)
#y0 = comp.loc[300, 0]['Riemann'].split()
y0 = comp.loc[300, 0]['Riemann']
y1 = y0.strip('[]')
y2 = y1.split()
print(np.array(y2).astype('float'))







f0 = plt.figure(constrained_layout=True, figsize=(12, 18), sharex=True, sharey=True)
spec = GridSpec(ncols=8, nrows=4, figure=f0)
# size 300nm data
f0_ax00 = f0.add_subplot(spec[0, 0])
f0_ax00.semilogy(np.array(comp.loc[300, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='300nm \u2225')
f0_ax00.semilogy(np.array(comp.loc[300, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 0]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 0]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 300nm \u2225')
f0_ax00.legend(loc=1)
f0_ax01 = f0.add_subplot(spec[0, 1])
f0_ax01.semilogy(np.array(comp.loc[300, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='300nm \u27f3')
f0_ax01.semilogy(np.array(comp.loc[300, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 90]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 90]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 300nm \u27f3')
f0_ax01.legend(loc=1)
f0_ax02 = f0.add_subplot(spec[0, 2])
f0_ax02.semilogy(np.array(comp.loc[300, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='300nm \u22A5')
f0_ax02.semilogy(np.array(comp.loc[300, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 180]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 180]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 300nm \u22A5')
f0_ax02.legend(loc=1)
f0_ax03 = f0.add_subplot(spec[0, 3])
f0_ax03.plot(np.array(comp.loc[300, 0]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 0]['nbins'].strip('[]').split()).astype('float'), color='red', label='MT dist. 300nm \u2225')
f0_ax03.plot(np.array(comp.loc[300, 90]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 90]['nbins'].strip('[]').split()).astype('float'), color='green', label='MT dist. 300nm \u27f3')
f0_ax03.plot(np.array(comp.loc[300, 180]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 180]['nbins'].strip('[]').split()).astype('float'), color='blue', label='MT dist. 300nm \u22A5')

# size 400nm data
f0_ax10 = f0.add_subplot(spec[1, 0])
f0_ax10.semilogy(np.array(comp.loc[400, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='400nm \u2225')
f0_ax10.semilogy(np.array(comp.loc[400, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 0]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 0]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 400nm \u2225')
f0_ax10.legend(loc=1)
f0_ax11 = f0.add_subplot(spec[1, 1])
f0_ax11.semilogy(np.array(comp.loc[400, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='400nm \u27f3')
f0_ax11.semilogy(np.array(comp.loc[400, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 90]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 90]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 400nm \u27f3')
f0_ax11.legend(loc=1)
f0_ax12 = f0.add_subplot(spec[1, 2])
f0_ax12.semilogy(np.array(comp.loc[400, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='400nm \u22A5')
f0_ax12.semilogy(np.array(comp.loc[400, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 180]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 180]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 400nm \u22A5')
f0_ax12.legend(loc=1)
f0_ax13 = f0.add_subplot(spec[1, 3])
f0_ax13.plot(np.array(comp.loc[400, 0]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 0]['nbins'].strip('[]').split()).astype('float'), color='red', label='MT dist. 400nm \u2225')
f0_ax13.plot(np.array(comp.loc[400, 90]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 90]['nbins'].strip('[]').split()).astype('float'), color='green', label='MT dist. 400nm \u27f3')
f0_ax13.plot(np.array(comp.loc[400, 180]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 180]['nbins'].strip('[]').split()).astype('float'), color='blue', label='MT dist. 400nm \u22A5')

# size 500nm data
f0_ax20 = f0.add_subplot(spec[2, 0])
f0_ax20.semilogy(np.array(comp.loc[500, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='500nm \u2225')
f0_ax20.semilogy(np.array(comp.loc[500, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 0]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 0]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 500nm \u2225')
f0_ax20.legend(loc=1)
f0_ax21 = f0.add_subplot(spec[2, 1])
f0_ax21.semilogy(np.array(comp.loc[500, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='500nm \u27f3')
f0_ax21.semilogy(np.array(comp.loc[500, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 90]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 90]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 500nm \u27f3')
f0_ax21.legend(loc=1)
f0_ax22 = f0.add_subplot(spec[2, 2])
f0_ax22.semilogy(np.array(comp.loc[500, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='500nm \u22A5')
f0_ax22.semilogy(np.array(comp.loc[500, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 180]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 180]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 500nm \u22A5')
f0_ax22.legend(loc=1)
f0_ax23 = f0.add_subplot(spec[2, 3])
f0_ax23.plot(np.array(comp.loc[500, 0]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 0]['nbins'].strip('[]').split()).astype('float'), color='red', label='MT dist. 500nm \u2225')
f0_ax23.plot(np.array(comp.loc[500, 90]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 90]['nbins'].strip('[]').split()).astype('float'), color='green', label='MT dist. 500nm \u27f3')
f0_ax23.plot(np.array(comp.loc[500, 180]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 180]['nbins'].strip('[]').split()).astype('float'), color='blue', label='MT dist. 500nm \u22A5')

# size 600nm data
f0_ax30 = f0.add_subplot(spec[3, 0])
f0_ax30.semilogy(np.array(comp.loc[600, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='600nm \u2225')
f0_ax30.semilogy(np.array(comp.loc[600, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 0]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 0]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 600nm \u2225')
f0_ax30.legend(loc=1)
f0_ax31 = f0.add_subplot(spec[3, 1])
f0_ax31.semilogy(np.array(comp.loc[600, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='600nm \u27f3')
f0_ax31.semilogy(np.array(comp.loc[600, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 90]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 90]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 600nm \u27f3')
f0_ax31.legend(loc=1)
f0_ax32 = f0.add_subplot(spec[3, 2])
f0_ax32.semilogy(np.array(comp.loc[600, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='600nm \u22A5')
f0_ax32.semilogy(np.array(comp.loc[600, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 180]['Mie Fit PF'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 180]['Mie Fit PF'].strip('[]').split()).astype('float')), color='black', label='MT 600nm \u22A5')
f0_ax32.legend(loc=1)
f0_ax33 = f0.add_subplot(spec[3, 3])
f0_ax33.plot(np.array(comp.loc[600, 0]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 0]['nbins'].strip('[]').split()).astype('float'), color='red', label='MT dist. 600nm \u2225')
f0_ax33.plot(np.array(comp.loc[600, 90]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 90]['nbins'].strip('[]').split()).astype('float'), color='green', label='MT dist. 600nm \u27f3')
f0_ax33.plot(np.array(comp.loc[600, 180]['dbins'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 180]['nbins'].strip('[]').split()).astype('float'), color='blue', label='MT dist. 600nm \u22A5')

# optical properties
f0_ax08 = f0.add_subplot(spec[0:1, 4:8], projection='3d')
f0_ax08.scatter(comp['mu'], comp['sigma'], comp['n'], color='blue', marker='o', label='n retrieved')
f0_ax08.scatter(comp['mu'], comp['sigma'], comp['k'], color='red', marker='^', label='k retrieved')
f0_ax08.set_title('Retrieved n and k')
f0_ax08.set_xlabel('Retrieved Mean Size')
f0_ax08.set_ylabel('Retrieved Geom. Stdev')
f0_ax08.set_zlabel('Retrieved CRI Values')
f0_ax08.legend(loc=1)
f0_ax18 = f0.add_subplot(spec[2, 4:6])
f0_ax18.plot(comp['mu'], comp['Cext'], color='blue', ls=' ', marker='o', label='Cext')
f0_ax18.plot(comp['mu'], comp['Csca'], color='green', ls=' ', marker='x', label='Csca')
f0_ax18.plot(comp['mu'], comp['Cabs'], color='red', ls=' ', marker='^', label='Cabs')
f0_ax18.set_xlabel('Retrieved Mean Size')
f0_ax18.set_ylabel('$cm^{2}$/p')
f0_ax18.set_title('Retrieved Cross-Sections')
f0_ax18.legend(loc=1)
f0_ax28 = f0.add_subplot(spec[2, 6:8])
f0_ax28.plot(comp['mu'], comp['Cback'], color='pink', ls=' ', marker='D', label='Cback')
f0_ax28.set_xlabel('Retrieved Mean Size')
f0_ax28.set_ylabel('$cm^{2}$/p')
f0_ax28.set_title('Retrieved Back Scattering Cross-Sections')
f0_ax28.legend(loc=1)
f0_ax38 = f0.add_subplot(spec[3, 4:6])
f0_ax38.plot(comp['mu'], comp['g'], color='purple', ls=' ', marker='*', label='g')
f0_ax38.set_xlabel('Retrieved Mean Size')
f0_ax38.set_ylabel('g')
f0_ax38.set_title('Retrieved Asymmetry Parameter')
f0_ax38.legend(loc=1)
f0_ax48 = f0.add_subplot(spec[3, 6:8])
f0_ax48.plot(comp['mu'], comp['Cratio'], color='black', marker='.', label='Cratio')
f0_ax48.set_xlabel('Retrieved Mean Size')
f0_ax48.set_ylabel('$C_{ratio}$')
f0_ax48.set_title('Retrieved Ratio (Forward:Back) Scattering')
f0_ax38.legend(loc=1)
f0_ax00.set_title('\u2225 Phase Function')
f0_ax01.set_title('\u27f3 Phase Function')
f0_ax02.set_title('\u22A5 Phase Function')
f0_ax03.set_title('Lognormal Distributions Retrieved')
f0_ax00.set_ylabel('Norm. Intensity')
f0_ax10.set_ylabel('Norm. Intensity')
f0_ax20.set_ylabel('Norm. Intensity')
f0_ax30.set_xlabel('Scattering Angle (\u00b0)')
f0_ax30.set_ylabel('Norm. Intensity')
f0_ax31.set_xlabel('Scattering Angle (\u00b0)')
f0_ax32.set_xlabel('Scattering Angle (\u00b0)')
f0_ax33.set_xlabel('Particle Size')
f0_ax33.set_ylabel('#/cc')
f0_ax23.set_ylabel('#/cc')
f0_ax13.set_ylabel('#/cc')
f0_ax03.set_ylabel('#/cc')
plt.ylabel('Normalized Intensity')
plt.suptitle('Recent Ammonium Sulfate Measurements')
plt.savefig(save_directory + 'AS_measurements_retrieved.png', format='png')
plt.savefig(save_directory + 'AS_measurements_retrieved.pdf', format='pdf')
plt.show()


