'''
Austen K. Scruggs
03-10-2020
Description: Hoping to be able to get some useful info from some examples
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import least_squares, linprog
from math import pi, sqrt, log
from scipy.interpolate import pchip_interpolate
from pytmatrix import tmatrix, scatter
from pytmatrix.tmatrix import orientation
import PyMieScatt as PMS
from pytmatrix.psd import PSDIntegrator, BinnedPSD
import pytmatrix.tmatrix_aux as tmatrix_aux

save_dir = '/home/austen/Desktop/Recent/'
mie_dir = '/home/austen/Desktop/pandas_data.txt'

# defining functions
# log normal distribution function, we might want it normalized, check if the equation is right...
def LogNormal(size, mu, gsd):
    if size ==0.0:
        return 0.0
    else:
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (1 / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))


# setting Scatterer class attributes, in a loop to obtain a phase function #or_pdf=orientation.uniform_pdf()
def Tmatrix_PhaseFunction(m, diameter, gsd, wavelength, axis_ratio, theta_array, bins):
    SL_size = []
    SR_size = []
    SL_mat = []
    SR_mat = []
    SL_mat_sig = []
    SR_mat_sig = []
    ext_x_size_sl = []
    ext_x_size_sr = []
    sca_x_size_sl = []
    sca_x_size_sr = []
    ssa_size_sl = []
    ssa_size_sr = []
    asym_size_sl = []
    asym_size_sr = []
    ldr_h_size = []
    ldr_v_size = []
    bin_counts = [LogNormal(size=i, mu=diameter, gsd=gsd) for i in bins]
    bin_edges = np.append(bins, bins[-1] + 1)
    geom_list = [(0.0, theta_i, 0.0, 0.0, 0.0, 0.0) for theta_i in theta_array]
    for counter, bin in enumerate(bins):
        for element in theta_array:
            # so i had to multipy by 1E-9 so that the units of the output differential scattering cross section would be in M^2/(NumDensity * steradians)
            scatterer = tmatrix.Scatterer(radius=(bin/2.0) * 1E-9, wavelength=wavelength * 1E-9, m=m, axis_ratio=axis_ratio, thet0=0.0, thet=element, phi=0.0)
            # The function sca_intensity returns the differential scattering cross section at the angle specified so we loop it to get
            # a phase function of units differential scattering cross section
            SL_i = scatter.sca_intensity(scatterer, h_pol=True)
            SR_i = scatter.sca_intensity(scatterer, h_pol=False)
            SL_size.append(SL_i)
            SR_size.append(SR_i)
        # Had to mult by 1E4, this is a unit conversion from M^2/p to cm^2/p
        ext_x_size_sl.append(scatter.ext_xsect(scatterer, h_pol=True) * 1E4)
        ext_x_size_sr.append(scatter.ext_xsect(scatterer, h_pol=False) * 1E4)
        sca_x_size_sl.append(scatter.sca_xsect(scatterer, h_pol=True) * 1E4)
        sca_x_size_sr.append(scatter.sca_xsect(scatterer, h_pol=False) * 1E4)
        # No need to convert units as these are ratios
        ssa_size_sl.append(scatter.ssa(scatterer, h_pol=True))
        ssa_size_sr.append(scatter.ssa(scatterer, h_pol=False))
        asym_size_sl.append(scatter.asym(scatterer, h_pol=True))
        asym_size_sr.append(scatter.asym(scatterer, h_pol=False))
        ldr_h_size.append(scatter.ldr(scatterer, h_pol=True))
        ldr_v_size.append(scatter.ldr(scatterer, h_pol=False))
        # set up one that gives us the signal matrix and one giving us the differential cross section marix
        # mult by 1E4 moves us from M^2 (meters^2) to cm^2, bin_counts is the particle concentration in the bin
        SL_mat_sig.append((np.array(SL_size) * 1E4 * bin_counts[counter]) * 1E8)
        SR_mat_sig.append((np.array(SR_size) * 1E4 * bin_counts[counter]) * 1E8)
        SL_mat.append(np.array(SL_size) * 1E4)
        SR_mat.append(np.array(SR_size) * 1E4)
        SL_size = []
        SR_size = []
    SL_mat = np.array(SL_mat)
    SR_mat = np.array(SR_mat)
    # remember the differential scattering cross section was returned, so we need to perform a weighted average!
    # this is returning the weighted average differential scattering cross section, I think its in nm^2/(particle * solid angle)
    SL_diff_sca_x_section = np.average(SL_mat, axis=0, weights=bin_counts)
    SR_diff_sca_x_section = np.average(SR_mat, axis=0, weights=bin_counts)
    #The differential (scattering) cross section is defined as the ratio of the intensity of radiant energy scattered in a given direction
    #to the incident irradiance and thus has dimensions of area per unit solid angle.
    #The symbol σ is frequently used for scattering cross section and dσ/dΩ for the differential cross section.
    SL_sig = np.sum(SL_mat_sig, axis=0) * 1E2 # a factor of 100 is needed to convert from a ratio to a value
    SL_sig_norm = SL_sig / np.sum(SL_sig)
    SR_sig = np.sum(SR_mat_sig, axis=0) * 1E2 # a factor of 100 is needed to convert from a ratio to a value
    SR_sig_norm = SR_sig / np.sum(SR_sig)
    # SL_mat_wsum_norm = SL_mat_wsum / np.sum(SL_mat_wsum)
    # SR_mat_wsum_norm = SR_mat_wsum / np.sum(SR_mat_wsum)
    bins_output = bin_edges
    counts_output = bin_counts
    # had to multiply in some cases by 1E4 for units to work, cross sections need to be in
    cext_x_sl = np.average(ext_x_size_sl, weights=counts_output)
    cext_x_sr = np.average(ext_x_size_sr, weights=counts_output)
    csca_x_sl = np.average(sca_x_size_sl, weights=counts_output)
    csca_x_sr = np.average(sca_x_size_sr, weights=counts_output)
    ssa_sl = np.average(ssa_size_sl, weights=counts_output)
    ssa_sr = np.average(ssa_size_sr, weights=counts_output)
    asym_sl = np.average(asym_size_sl, weights=counts_output)
    asym_sr = np.average(asym_size_sr, weights=counts_output)
    ldr_h = np.average(ldr_h_size, weights=counts_output)
    ldr_v = np.average(ldr_v_size, weights=counts_output)
    # there is a unit conversion from cm^-1 to Mm^-1 by multiplying by 1E8
    #bext_sl = np.sum(np.multiply(np.array(ext_x_size_sl), np.array(bin_counts)) * 1E8)
    #bext_sr = np.sum(np.multiply(np.array(ext_x_size_sr), np.array(bin_counts)) * 1E8)
    #bsca_sl = np.sum(np.multiply(np.array(sca_x_size_sl), np.array(bin_counts)) * 1E8)
    #bsca_sr = np.sum(np.multiply(np.array(sca_x_size_sr), np.array(bin_counts)) * 1E8)
    # sl (parallel & horizontal polarization parameters)
    params_sl = {"SL": SL_sig, "SL Norm": SL_sig_norm, "SL Diff Csca": SL_diff_sca_x_section, "Cext": cext_x_sl, "Csca": csca_x_sl, "ssa": ssa_sl, "g": asym_sl, "ldr": ldr_h}
    params_sr = {"SR": SR_sig, "SR Norm": SR_sig_norm, "SR Diff Csca": SR_diff_sca_x_section, "Cext": cext_x_sr, "Csca": csca_x_sr, "ssa": ssa_sr, "g": asym_sr, "ldr": ldr_v}
    '''
    print('-------Tmat Results-------')
    print('Total Number Density in Dist. : ', np.sum(bin_counts))
    print('Tmat SL Sum: ', np.sum(SL_sig))
    print('Tmat SR Sum: ', np.sum(SR_sig))
    print('Tmat Ext Coeff SL: ', bext_sl)
    print('Tmat Ext Coeff SR: ', bext_sr)
    print('Tmat Ext x-section SL: ', cext_x_sl)
    print('Tmat Ext x-section SR: ', cext_x_sr)
    print('Tmat Sca Coeff SL: ', bsca_sl)
    print('Tmat Sca Coeff SR: ', bsca_sr)
    print('Tmat Sca x-section SL: ', csca_x_sl)
    print('Tmat Sca x-section SR: ', csca_x_sr)
    print('Tmat SSA SL: ', ssa_sl)
    print('Tmat SSA SR: ', ssa_sr)
    print('Tmat g SL: ', asym_sl)
    print('Tmat g SR: ', asym_sr)
    print('Tmat LDR SL: ', ldr_h)
    print('Tmat LDR SR: ', ldr_v)
    '''
    '''
    # starting psd module stuff
    points = len(bins)
    scatterer2 = tmatrix.Scatterer(wavelength=wavelength, m=m, axis_ratio=axis_ratio)
    scatterer2.psd_integrator = PSDIntegrator()
    scatterer2.psd_integrator.num_points = int(round(points))
    # the below is a tupple of 4 preset geometries unfortunately they must be used in psd integration
    # geom = (tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_vert_forw, tmatrix_aux.geom_horiz_forw, tmatrix_aux.geom_vert_back)
    # for some stupid reason, the geom tupple of tupples must contain one of the presets, so it calculates them in addition to all scattering geometries corresponding to 0-180 degree phase function
    preset_geoms = [tmatrix_aux.geom_horiz_back, tmatrix_aux.geom_vert_forw, tmatrix_aux.geom_horiz_forw, tmatrix_aux.geom_vert_back]
    # the geometry is a tupple of tupples, we made it from the below which is acceptable geom = ((theta0, theta, phi0, phi, alpha, beta), (theta0, theta, phi0, phi, alpha, beta),...)
    # this loop produces all the scattering geometries responsible for the phase function
    # here is where we had to add the presets for some stupid reason
    geom_list_w_presets = geom_list + preset_geoms
    # convert list of tupples to tupple of tupples
    geom_tuple_w_presets = tuple(geom_list_w_presets)
    #print(geom_tuple_w_presets)
    # setting all our geometries in order to compute the scattering phase matrix and amplitude matrix look up tables
    scatterer2.psd_integrator.geometries = geom_tuple_w_presets
    # set max bin to the end of the bin edges we had input
    scatterer2.psd_integrator.D_max = bin_edges[-1]
    scatterer2.psd_integrator.init_scatter_table(scatterer2, angular_integration=False, verbose=False)
    # save the scatter as a lookup table, i think you have to define the file in advance
    # scatterer.psd_integrator.save_scatter_table(save_dir + 'scatter_table_PSL_900nm.txt', description='900nm PSL scattering table at axis ratio' + str(axis_ratio))
    # get integrated properties later! work on this when you get a good PF
    # bin_edges_and_one
    scatterer2.psd = BinnedPSD(bin_edges=bin_edges, bin_psd=bin_counts)
    bins_output = scatterer2.psd.bin_edges
    counts_output = scatterer2.psd.bin_psd
    '''
    return [params_sl, params_sr, bins_output, counts_output]

def Residuals_SLSR(x, w_n, SL_M, SR_M, SL_Theta, SR_Theta, bin_edges):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    SR_M = np.array(SR_M)
    SR_M = SL_M[~np.isnan(SR_M)]
    SR_Theta = SR_Theta[~np.isnan(SR_M)]
    # theta_cal = np.array([slope * element + intercept for element in SU_C])
    bin_counts = [LogNormal(size=i, mu=x[0], gsd=x[1]) for i in bin_edges]
    theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[2], x[3]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_mie = np.array(theta_mie) * (180.0 / pi)
    SL_pchip = pchip_interpolate(xi=theta_mie, yi=SL, x=SL_Theta, der=0, axis=0)
    SR_pchip = pchip_interpolate(xi=theta_mie, yi=SR, x=SR_Theta, der=0, axis=0)
    SL_pchip_norm = SL_pchip / np.sum(SL_pchip)
    SR_pchip_norm = SR_pchip / np.sum(SR_pchip)
    # somehow residuals are coming out negative...05/14/2020
    sl_diff = np.absolute(((SL_M - SL_pchip_norm) / SL_pchip_norm))
    sr_diff = np.absolute(((SR_M - SR_pchip_norm) / SR_pchip_norm))
    residuals = np.sum(sl_diff) + np.sum(sr_diff)
    print('Guess mu: ', x[0], 'Geometric stdev: ', x[1], 'Guess n: ', x[2], 'Guess k: ', x[3], 'Summed Error: ', residuals)
    return residuals


def Residuals_SL(x, w_n, SL_M, SL_Theta, bin_edges):
    # pre allocation
    SL_2darray = []
    # Data
    SL_M = np.array(SL_M)
    SL_M = SL_M[~np.isnan(SL_M)]
    SL_Theta = SL_Theta[~np.isnan(SL_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    bin_counts = [LogNormal(size=i, mu=x[0], gsd=x[1], N=x[2]) for i in bin_edges]
    theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[3], x[4]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_mie = np.array(theta_mie) * (180.0/pi)
    SL_pchip = pchip_interpolate(xi=theta_mie, yi=SL, x=SL_Theta, der=0, axis=0)
    #SL_pchip_norm = SL_pchip / np.sum(SL_pchip)
    SL_pchip_norm = SL_pchip
    # somehow residuals are coming out negative...05/14/2020
    sl_diff = np.absolute((SL_M - (SL_pchip_norm)) / SL_pchip_norm) * 100
    residuals = np.sum(sl_diff)
    print('Guess mu: ', x[0], 'Geometric stdev: ', x[1], 'Particle Count: ', x[2],'Guess n: ', x[3], 'Guess k: ', x[4], 'Summed Error: ', residuals)
    return residuals


def Residuals_SR(x, w_n, SR_M, SR_Theta, bin_edges):
    # pre allocation
    SR_2darray = []
    # Data
    SR_M = np.array(SR_M)
    SR_M = SR_M[~np.isnan(SR_M)]
    SR_Theta = SR_Theta[~np.isnan(SR_M)]
    #theta_cal = np.array([slope * element + intercept for element in SU_C])
    bin_counts = [LogNormal(size=i, mu=x[0], gsd=x[1], N=x[2]) for i in bin_edges]
    theta_mie, SL, SR, SU = PMS.SF_SD(complex(x[3], x[4]), wavelength=w_n, dp=bin_edges, ndp=bin_counts, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_mie = np.array(theta_mie) * (180.0 / pi)
    #print(theta_mie)
    SR_pchip = pchip_interpolate(xi=theta_mie, yi=SR, x=SR_Theta, der=0, axis=0)
    #SR_pchip_norm = SR_pchip / np.sum(SR_pchip)
    SR_pchip_norm = SR_pchip
    sr_diff = np.absolute((SR_M - (SR_pchip_norm)) / SR_pchip_norm) * 100
    residuals = np.sum(sr_diff)
    print('Guess mu: ', x[0], 'Geometric stdev: ', x[1], 'Particle Count: ', x[2],'Guess n: ', x[3], 'Guess k: ', x[4], 'Summed Error: ', residuals)
    return residuals


# create normalized mie theory phase functions
#theta_mie, SL_mie, SR_mie, SU_mie = PMS.ScatteringFunction(m, wavelength, 900.0, nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)


# import mie data space from other source
#mie_space = pd.read_csv(mie_dir, sep=',', header=0)
#SL_mie = mie_space['SL']
#SR_mie = mie_space['SR']
#theta_mie = mie_space['Theta']


# setting some constants for an individual spheroidal calculation
# diameter, radius, and  wavelength in nanometers
diameter_val = 903.00
gsd_val = 1.005
wavelength_val = 663.00
m_val = complex(1.59, 0.00)
n_medium_val = 1.0
size_bins = np.linspace(850.0, 950.0, 101)
bin_counts = np.array([LogNormal(i, diameter_val, gsd_val) for i in size_bins])
#print('Particle Count/bin: ', bin_counts)

# 0.6 to less than 1.00 is a prolate top, greater than 1.00 to 1.66 is an oblate top, and 1.00 is spherical particle for ax_r
ax_r_array = np.arange(0.6, 1.7, 0.1)
#ax_r_array = [1.0]
#ax_r_val = ax_r_array[0]
theta_scattered = np.arange(0.0, 180.5, 0.5)

'''
# test our Tmatrix_PhaseFunction and various Mie theory Phase Function calculation methods, they all need to yield the same phase function
# they do! it works!!!!!!!!!!!!
# make a phase function! testing everything really
SL_params, SR_params, bins, counts = Tmatrix_PhaseFunction(m=m_val, diameter=diameter_val, gsd=gsd_val, wavelength=wavelength_val, axis_ratio=ax_r_val, theta_array=theta_scattered, bins=size_bins, N=N_val)
theta_mie, SL_mie, SR_mie, SU_mie = PMS.SF_SD(m_val, wavelength=wavelength_val, dp=size_bins, ndp=bin_counts, nMedium=n_medium_val, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
theta_mie = np.array(theta_mie) * (180.0/pi)
print('total signal SL mie: ', np.sum(SL_mie))
SL_mie_norm = SL_mie
SR_mie_norm = SR_mie
#SL_mie_norm = SL_mie / np.sum(SL_mie)
#SR_mie_norm = SR_mie / np.sum(SR_mie)


soln_mie, bins_mie, counts_mie = PMS.Mie_Lognormal(m=m_val, wavelength=wavelength_val, geoStdDev=gsd_val, geoMean=diameter_val, nMedium=n_medium_val, lower=size_bins[0], upper=size_bins[-1], numberOfBins=size_bins[-1]-size_bins[0]+1, numberOfParticles=N_val, asDict=True, returnDistribution=True)
print('-------Mie Results-------')
print('Mie ext coefficient: ', soln_mie['Bext'])
print('Mie ext x-section: ', (soln_mie['Bext']/1E8)/np.sum(counts_mie))
print('Mie sca coefficient: ', soln_mie['Bsca'])
print('Mie sca x-section: ', (soln_mie['Bsca']/1E8)/np.sum(counts_mie))
print('Mie SL Sum: ',np.sum(SL_mie_norm))
'''


'''
# make figures
t_font = 24
l_font = 18
# distribution
f, ax = plt.subplots(figsize=(8, 6))
ax.bar(x=bins[:-1], height=counts, width=np.diff(bins), align='edge', color='blue',log=False, label='psd output')
ax.semilogy(np.array(size_bins), bin_counts, color='red', marker='o', ls=' ', label='psd input')
#ax.semilogy(bins[:-1], counts, color='green', marker='o', ls=' ', label='psd tmat output')
ax.set_xlabel('Diameter', fontsize=l_font)
ax.set_ylabel('Counts', fontsize=l_font)
ax.set_title('Particle Size Distribution', fontsize=t_font)
ax.grid(True)
ax.legend(loc=1, fontsize=l_font)
plt.tight_layout()
plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_PSD' + str('{:.2f}'.format(ax_r_val)) + '.pdf', format='pdf')
plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_PSD' + str('{:.2f}'.format(ax_r_val)) + '.png', format='png')
plt.show()
plt.clf()
plt.close()

f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].semilogy(theta_scattered, SL_params['SL'], color='red', ls='-', label='T-Matrix: SL psd\n w axis ratio: ' + str('{:.2f}'.format(ax_r_val)))
ax0[0].semilogy(theta_mie, SL_mie_norm, color='green', ls='-', label='Mie Theory: SL psd')
#ax0[0].semilogy(theta_mie, SL_MIE_NORM, color='magenta', ls=':', label='Mie Theory: SL single')
#ax0[0].semilogy(theta_mie, SL_mie_norm_2, color='blue', ls=':', label='Mie Theory: SL w sum')
ax0[0].set_xlabel('\u0398(\u00b0)', fontsize=l_font)
ax0[0].set_ylabel('Normalized Intensity', fontsize=l_font)
ax0[0].set_title('Phase Function Resultant from\n Incident Horizontally Polarized Light', fontsize=t_font)
ax0[0].grid(True)
ax0[0].legend(loc=1, fontsize=l_font)
ax0[1].semilogy(theta_scattered, SR_params['SR'], color='red', ls='-', label='T-Matrix: SR psd\n w axis ratio: ' + str('{:.2f}'.format(ax_r_val)))
ax0[1].semilogy(theta_mie, SR_mie_norm, color='green', ls='-', label='Mie Theory: SR psd')
#ax0[1].semilogy(theta_mie, SR_MIE_NORM, color='magenta', ls=':', label='Mie Theory: SR single')
#ax0[1].semilogy(theta_mie, SR_mie_norm_2, color='blue', ls=':', label='Mie Theory: SR w sum')
ax0[1].set_xlabel('\u0398(\u00b0)', fontsize=l_font)
ax0[1].set_ylabel('Normalized Intensity', fontsize=l_font)
ax0[1].set_title('Phase Function Resultant from\n Incident Vertically Polarized Light', fontsize=t_font)
ax0[1].grid(True)
ax0[1].legend(loc=1, fontsize=l_font)
plt.tight_layout()
plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_NLLS' + str('{:.2f}'.format(ax_r_val)) + '.pdf', format='pdf')
plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_NLLS' + str('{:.2f}'.format(ax_r_val)) + '.png', format='png')
plt.show()
plt.clf()
plt.close()

#'''
#'''
# pre-allocation lists

# storing tmat parameters for each axis ratio in lst
ax_r_list = []
SL_tmatpf_list = []
SR_tmatpf_list = []
SL_asym_tmat_list = []
SR_asym_tmat_list = []
SLSR_Cext_tmat_list = []
SLSR_Csca_tmat_list = []

# store found mie theory pf and parameters
SL_pf_list = []
SR_pf_list = []
SLSR_n_list = []
SLSR_k_list = []
SLSR_mu_list = []
SLSR_sigma_list = []
SLSR_Theta_list = []
SLSR_dbins_list = []
SLSR_nbins_list = []
SLSR_Cext_list = []
SLSR_Csca_list = []
SLSR_Cabs_list = []
SLSR_g_list = []
SLSR_Cpr_list = []
SLSR_Cback_list = []
SLSR_Cratio_list = []

# Bext=extinction coefficient, Bsca=scattering coefficient Babs=absorption coefficient
# bigG=bulk asymmetry parameter, Bpr=radiation pressure coefficient, Bback=backscattering coefficient, Bratio= ratio of back scattering to total scattering coefficients

# temporary arrays to store values for MieQ loop
Cext_arr = []
Csca_arr = []
Cabs_arr = []
g_arr = []
Cpr_arr = []
Cback_arr = []
Cratio_arr = []




#SU_list = []
#DLP_list = []
for counter, ax_r_val in enumerate(ax_r_array):
    print('axis ratio: ', ax_r_val)
    ax_r_list.append(ax_r_val)
    # computes the spheroidal tmatrix solution for the given aerosol with specified axis ratio and psd
    params_sl, params_sr, bins, counts = Tmatrix_PhaseFunction(m=m_val, diameter=diameter_val, gsd=gsd_val, wavelength=wavelength_val, axis_ratio=ax_r_val, theta_array=theta_scattered, bins=size_bins)
    SL_tmatpf_list.append(params_sl["SL"])
    SR_tmatpf_list.append(params_sr["SR"])
    SL_asym_tmat_list.append(params_sl["g"])
    SR_asym_tmat_list.append(params_sr["g"])
    SLSR_Cext_tmat_list.append(params_sl["Cext"])
    SLSR_Csca_tmat_list.append(params_sl["Csca"])


    # NLLS SLSR
    result_SL = least_squares(Residuals_SLSR, x0=[903.0, 1.005, 1.59, 0.001], method='trf', args=(wavelength_val, params_sl["SL Norm"], params_sr["SR Norm"], theta_scattered, theta_scattered, size_bins), bounds=([1.0, 1.000, 1.00, 0.000], [1000.0, 1.500, 2.00, 1.000]))
    # append solutions
    SLSR_mu_list.append(result_SL.x[0])
    SLSR_sigma_list.append(result_SL.x[1])
    SLSR_n_list.append(result_SL.x[2])
    SLSR_k_list.append(result_SL.x[3])
    # return normalized SL mie and parameters
    theta_mie, SL_mie, SR_mie, SU_mie = PMS.SF_SD(m=complex(result_SL.x[2], result_SL.x[3]), wavelength=wavelength_val, dp=size_bins, ndp=[LogNormal(size=i, mu=result_SL.x[0], gsd=result_SL.x[1]) for i in size_bins], nMedium=1.0, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
    theta_mie = np.array(theta_mie) * (180.0/pi)
    SL_mie_norm = SL_mie / np.sum(SL_mie)
    SR_mie_norm = SR_mie / np.sum(SR_mie)
    SL_pf_list.append(SL_mie_norm)
    SR_pf_list.append(SR_mie_norm)
    SLSR_Theta_list.append(theta_mie)
    SLSR_dbins = size_bins
    SLSR_nbins = [LogNormal(i, result_SL.x[0], result_SL.x[1]) for i in size_bins]
    for size in size_bins:
        SLSR_optics_params = PMS.MieQ(m=complex(result_SL.x[2], result_SL.x[3]), wavelength=wavelength_val, diameter=size ,nMedium=1.0, asDict=True, asCrossSection=True)
        # 1E-14 cm^2 in 1 nm^2, so we are converting cross-section units to cm^2
        Cext_arr.append(SLSR_optics_params['Cext'] * 1E-14)
        Csca_arr.append(SLSR_optics_params['Csca'] * 1E-14)
        Cabs_arr.append(SLSR_optics_params['Cabs'] * 1E-14)
        g_arr.append(SLSR_optics_params['g'])
        Cpr_arr.append(SLSR_optics_params['Cpr'] * 1E-14)
        Cback_arr.append(SLSR_optics_params['Cback'] * 1E-14)
        Cratio_arr.append(SLSR_optics_params['Cratio'] * 1E-14)
    Cext = np.average(Cext_arr, weights=SLSR_nbins)
    Csca = np.average(Csca_arr, weights=SLSR_nbins)
    Cabs = np.average(Cabs_arr, weights=SLSR_nbins)
    g = np.average(g_arr, weights=SLSR_nbins)
    Cpr = np.average(Cpr_arr, weights=SLSR_nbins)
    Cback = np.average(Cback_arr, weights=SLSR_nbins)
    Cratio = np.average(Cratio_arr, weights=SLSR_nbins)
    SLSR_dbins_list.append(SLSR_dbins)
    SLSR_nbins_list.append(SLSR_nbins)
    SLSR_Cext_list.append(Cext)
    SLSR_Csca_list.append(Csca)
    SLSR_Cabs_list.append(Cabs)
    SLSR_g_list.append(g)
    SLSR_Cpr_list.append(Cpr)
    SLSR_Cback_list.append(Cback)
    SLSR_Cratio_list.append(Cratio)
    # clear temporary arrays
    Cext_arr = []
    Csca_arr = []
    Cabs_arr = []
    g_arr = []
    Cpr_arr = []
    Cback_arr = []
    Cratio_arr = []

    # make figures
    t_font = 24
    l_font = 18
    # distribution
    f_dist, ax_dist = plt.subplots(figsize=(8, 6))
    ax_dist.plot(SLSR_dbins, SLSR_nbins, color='blue', marker='o', ls='-', label='sl psd')
    ax_dist.set_xlabel('Diameter', fontsize=l_font)
    ax_dist.set_ylabel('Normalized Counts', fontsize=l_font)
    ax_dist.set_title('Normalized Particle Size Distributions', fontsize=t_font)
    ax_dist.grid(True)
    ax_dist.legend(loc=1, fontsize=l_font)
    plt.tight_layout()
    plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_PSDS' + str('{:.2f}'.format(ax_r_val)) + '.pdf', format='pdf')
    plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_PSDS' + str('{:.2f}'.format(ax_r_val)) + '.png', format='png')
    #plt.show()
    plt.clf()
    plt.close()

    # make figures
    t_font = 24
    l_font = 18
    fslsr, axslsr = plt.subplots(1, 2, figsize=(12, 6))
    axslsr[0].semilogy(theta_scattered, params_sl['SL Norm'], color='red', ls='-', label='T-Matrix: SL\n axis ratio: ' + str('{:.2f}'.format(ax_r_val)))
    axslsr[0].semilogy(theta_mie, SL_mie_norm, color='blue', ls=':', label='Mie Theory: SL')
    axslsr[0].set_xlabel('\u0398(\u00b0)', fontsize=l_font)
    axslsr[0].set_ylabel('Normalized Intensity', fontsize=l_font)
    axslsr[0].set_title('Phase Function Resultant from\n Incident Horizontally Polarized Light', fontsize=t_font)
    axslsr[0].grid(True)
    axslsr[0].legend(loc=1, fontsize=l_font)
    axslsr[1].semilogy(theta_scattered, params_sr['SR Norm'], color='red', ls='-', label='T-Matrix: SR\n axis ratio: ' + str('{:.2f}'.format(ax_r_val)))
    axslsr[1].semilogy(theta_mie, SR_mie_norm, color='blue', ls=':', label='Mie Theory: SR')
    axslsr[1].set_xlabel('\u0398(\u00b0)', fontsize=l_font)
    axslsr[1].set_ylabel('Normalized Intensity', fontsize=l_font)
    axslsr[1].set_title('Phase Function Resultant from\n Incident Vertically Polarized Light', fontsize=t_font)
    axslsr[1].grid(True)
    axslsr[1].legend(loc=1, fontsize=l_font)
    plt.tight_layout()
    plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_NLLS_SL&SR' + str('{:.2f}'.format(ax_r_val)) + '.pdf', format='pdf')
    plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_NLLS_SL&SR' + str('{:.2f}'.format(ax_r_val)) + '.png', format='png')
    #plt.show()
    plt.clf()
    plt.close()
    print('Percent completed: ', ((counter + 1)/len(ax_r_array)) * 100)
    print('Finished calculation for axis ratio: ', '{:.2f}'.format(ax_r_val))
    print('SL T-Matrix Sum: ', np.sum(params_sl['SL']))
    print('SL Mie Sum: ', np.sum(SL_mie_norm))
    print('SR T-Matrix Sum: ', np.sum(params_sr['SR']))
    print('SR Mie Sum: ', np.sum(SR_mie_norm))


# creating and saving data frame
slsr_data = np.transpose(np.array([SLSR_n_list, SLSR_k_list, SLSR_mu_list, SLSR_sigma_list, ax_r_list, SLSR_dbins_list, SLSR_nbins_list, SLSR_Cext_list, SLSR_Cext_tmat_list, SLSR_Csca_list, SLSR_Csca_tmat_list, SLSR_Cabs_list, SLSR_g_list, SLSR_Cpr_list, SLSR_Cback_list, SLSR_Cratio_list, SLSR_Theta_list, SL_pf_list, SL_asym_tmat_list, SL_tmatpf_list, SR_pf_list, SR_asym_tmat_list, SR_tmatpf_list]))
slsr_head = ["n", "k", "GeoMean", "GeoStdev", "axis ratio", "Dbins", "Nbins", "Cext Mie", "Cext Tmat", "Csca Mie", "Csca Tmat", "Cabs Mie", "g Mie", "Cpr Mie", "Cback Mie", "Cratio Mie", "Theta", "SL Mie", "SL g Tmat", "SL Tmat", "SR Mie", "SR g Tmat", "SR Tmat"]
SLSR_DF = pd.DataFrame(slsr_data, columns=slsr_head)
SLSR_DF.to_csv(save_dir + 'SLSR_DF_NLLS_Tmat_Mie.txt', sep=',')

# make figures
f_font =18
t_font = 16
l_font = 14
f1, ax1 = plt.subplots(3, 3, figsize=(24, 18))
ax1[0, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["n"], marker='o', color='black', ls=' ', label='n')
ax1[0, 0].set_title('n vs. axis ratio', fontsize=t_font)
ax1[0, 0].set_xlabel('axis ratio', fontsize=l_font)
ax1[0, 0].set_ylabel('n', fontsize=l_font)
ax1[0, 0].grid(True)
ax1[0, 0].legend(loc=1, fontsize=l_font)
ax1[0, 1].plot(SLSR_DF["axis ratio"], SLSR_DF["k"], marker='o', color='black', ls=' ', label='k')
ax1[0, 1].set_title('k vs. axis ratio', fontsize=t_font)
ax1[0, 1].set_xlabel('axis ratio', fontsize=l_font)
ax1[0, 1].set_ylabel('k', fontsize=l_font)
ax1[0, 1].grid(True)
ax1[0, 1].legend(loc=1, fontsize=l_font)
ax1[0, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["Cext Mie"], marker='o', color='red', ls=' ', label='$C_{ext}$ Mie')
ax1[0, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["Cext Tmat"], marker='o', color='blue', ls=' ', label='$C_{ext}$ Tmat')
ax1[0, 2].set_title('$C_{ext}$ vs. axis ratio', fontsize=t_font)
ax1[0, 2].set_xlabel('axis ratio', fontsize=l_font)
ax1[0, 2].set_ylabel('$C_{ext}$', fontsize=l_font)
ax1[0, 2].grid(True)
ax1[0, 2].legend(loc=1, fontsize=l_font)
ax1[1, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["Csca Mie"], marker='o', color='red', ls=' ', label='$C_{sca} Mie$')
ax1[1, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["Csca Tmat"], marker='o', color='blue', ls=' ', label='$C_{sca}$ Tmat')
ax1[1, 0].set_title('$C_{sca}$ vs. axis ratio', fontsize=t_font)
ax1[1, 0].set_xlabel('axis ratio', fontsize=l_font)
ax1[1, 0].set_ylabel('$C_{sca}$', fontsize=l_font)
ax1[1, 0].grid(True)
ax1[1, 0].legend(loc=1, fontsize=l_font)
ax1[1, 1].plot(SLSR_DF["axis ratio"], SLSR_DF["Cabs Mie"], marker='o', color='black', ls=' ', label='$C_{abs}$ Mie')
ax1[1, 1].set_title('$C_{abs}$ vs. axis ratio', fontsize=t_font)
ax1[1, 1].set_xlabel('axis ratio', fontsize=l_font)
ax1[1, 1].set_ylabel('$C_{abs}$', fontsize=l_font)
ax1[1, 1].grid(True)
ax1[1, 1].legend(loc=1, fontsize=l_font)
ax1[1, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["g Mie"], marker='o', color='red', ls=' ', label='G Mie')
ax1[1, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["SL g Tmat"], marker='o', color='blue', ls=' ', label='SL G Tmat')
ax1[1, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["SR g Tmat"], marker='o', color='lawngreen', ls=' ', label='SR G Tmat')
ax1[1, 2].set_title('G vs. axis ratio', fontsize=t_font)
ax1[1, 2].set_xlabel('axis ratio', fontsize=l_font)
ax1[1, 2].set_ylabel('G', fontsize=l_font)
ax1[1, 2].grid(True)
ax1[1, 2].legend(loc=1, fontsize=l_font)
ax1[2, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["GeoMean"], marker='o', color='black', ls=' ', label='Geometric Mean')
ax1[2, 0].set_title('Geometric Mean vs. axis ratio', fontsize=t_font)
ax1[2, 0].set_xlabel('axis ratio', fontsize=l_font)
ax1[2, 0].set_ylabel('Geometric Mean', fontsize=l_font)
ax1[2, 0].grid(True)
ax1[2, 0].legend(loc=1, fontsize=l_font)
ax1[2, 1].plot(SLSR_DF["axis ratio"], SLSR_DF["GeoStdev"], marker='o', color='black', ls=' ', label='Geometric Stdev')
ax1[2, 1].set_title('Geometric Standard Dev. vs. axis ratio', fontsize=t_font)
ax1[2, 1].set_xlabel('axis ratio', fontsize=l_font)
ax1[2, 1].set_ylabel('Geometric Stdev', fontsize=l_font)
ax1[2, 1].grid(True)
ax1[2, 1].legend(loc=1, fontsize=l_font)
ax1[2, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["Cratio Mie"], marker='o', color='black', ls=' ', label='Bratio')
ax1[2, 2].set_title('Bratio vs. axis ratio', fontsize=t_font)
ax1[2, 2].set_xlabel('axis ratio', fontsize=l_font)
ax1[2, 2].set_ylabel('Bratio', fontsize=l_font)
ax1[2, 2].grid(True)
ax1[2, 2].legend(loc=1, fontsize=l_font)
f1.suptitle('Optical Parameters vs. Axis Ratio', fontsize=f_font)
#plt.tight_layout()
plt.savefig(save_dir + 'NLLS_tmat_mie_params_SL.pdf', format='pdf')
plt.savefig(save_dir + 'NLLS_tmat_mie_params_SL.png', format='png')
#plt.show()
plt.clf()
plt.close()


print('Script Completed Running!')
print('-------Great Job!-------')
#'''

