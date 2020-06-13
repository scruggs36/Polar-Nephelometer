'''
Austen K. Scruggs
03-10-2020
Description: Hoping to be able to get some useful info from some examples
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyMieScatt as PMS
import pytmatrix.tmatrix_aux as tmatrix_aux
from pytmatrix import tmatrix, scatter
from pytmatrix.tmatrix import orientation
from pytmatrix.psd import PSDIntegrator, GammaPSD, BinnedPSD
from math import sqrt, log, pi


save_dir = '/home/austen/Desktop/Recent/'

def LogNormal(size, mu, gsd, N):
    if size ==0.0:
        return 0.0
    else:
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))



# setting Scatterer class attributes, in a loop to obtain a phase function #or_pdf=orientation.uniform_pdf()
def Tmatrix_PhaseFunction(m, diameter, gsd, wavelength, axis_ratio, theta_array, bins, N):
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
    bin_counts = [LogNormal(size=i, mu=diameter, gsd=gsd, N=N) for i in bins]
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
    print(len(ext_x_size_sl))
    print(len(bin_counts))
    bext_sl = np.sum(np.multiply(np.array(ext_x_size_sl), np.array(bin_counts)) * 1E8)
    bext_sr = np.sum(np.multiply(np.array(ext_x_size_sr), np.array(bin_counts)) * 1E8)
    bsca_sl = np.sum(np.multiply(np.array(sca_x_size_sl), np.array(bin_counts)) * 1E8)
    bsca_sr = np.sum(np.multiply(np.array(sca_x_size_sr), np.array(bin_counts)) * 1E8)
    # sl (parallel & horizontal polarization parameters)
    params_sl = {"SL": SL_sig, "SL Norm": SL_sig_norm, "SL diff csca": SL_diff_sca_x_section, "Bext": bext_sl, "Cext": cext_x_sl, "Bsca": bsca_sl,"Csca": csca_x_sl, "ssa": ssa_sl, "g": asym_sl, "ldr": ldr_h}
    params_sr = {"SR": SR_sig, "SR Norm": SR_sig_norm, "SR diff csca": SR_diff_sca_x_section, "Bext": bext_sr, "Cext": cext_x_sr, "Bsca": bsca_sr,"Csca": csca_x_sr, "ssa": ssa_sr, "g": asym_sr, "ldr": ldr_v}
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


# setting some constants
# diameter, radius, and  wavelength in nanometers
diameter_val = 903.0
gsd_val = 1.005
size_bins = np.arange(800.0, 1000.0, 1.0)
N_val = 1000
bin_counts = [LogNormal(size=i, mu=diameter_val, gsd=gsd_val, N=N_val) for i in size_bins]
wavelength_val = 663.0
m_val = complex(1.59, 0)
n_medium_val = 1.0
# 0.6 to less than 1.00 is a prolate top, greater than 1.00 to 1.66 is an oblate top, and 1.00 is spherical particle for ax_r
ax_r_val = 1.00
theta_scattered = np.arange(0.0, 180.5, 0.5)


# test our Tmatrix_PhaseFunction and various Mie theory Phase Function calculation methods, they all need to yield the same phase function
# they do! it works!!!!!!!!!!!!
# make a phase function! testing everything really
SL_params, SR_params, bins, counts = Tmatrix_PhaseFunction(m=m_val, diameter=diameter_val, gsd=gsd_val, wavelength=wavelength_val, axis_ratio=ax_r_val, theta_array=theta_scattered, bins=size_bins, N=N_val)
theta_mie, SL_mie, SR_mie, SU_mie = PMS.SF_SD(m_val, wavelength=wavelength_val, dp=size_bins, ndp=bin_counts, nMedium=n_medium_val, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
theta_mie = np.array(theta_mie) * (180.0/pi)
SL_mie_norm = SL_mie
SR_mie_norm = SR_mie
#SL_mie_norm = SL_mie / np.sum(SL_mie)
#SR_mie_norm = SR_mie / np.sum(SR_mie)


soln_mie, bins_mie, counts_mie = PMS.Mie_Lognormal(m=m_val, wavelength=wavelength_val, geoStdDev=gsd_val, geoMean=diameter_val, nMedium=n_medium_val, numberOfParticles=N_val, lower=800.0, upper=1000.0, numberOfBins=201, asDict=True, returnDistribution=True)
print('-------Mie Results-------')
print('Mie ext coefficient: ', soln_mie['Bext'])
print('Mie ext x-section: ', (soln_mie['Bext']/1E8)/np.sum(counts_mie))
print('Mie sca coefficient: ', soln_mie['Bsca'])
print('Mie sca x-section: ', (soln_mie['Bsca']/1E8)/np.sum(counts_mie))
print('Mie SL Sum: ',np.sum(SL_mie_norm))

'''
# make figures
t_font = 24
l_font = 18
f0, ax0 = plt.subplots(1, 2, figsize=(12, 6))
ax0[0].semilogy(theta_scattered, SL_tmat_norm, color='orange', ls='-', label='T-Matrix: SL\n axis ratio: ' + str('{:.2f}'.format(ax_r)))
ax0[0].semilogy(theta_mie, SL_mie_norm, color='blue', ls='--', label='Mie Theory: SL')
ax0[0].set_xlabel('\u0398(\u00b0)', fontsize=l_font)
ax0[0].set_ylabel('Normalized Intensity', fontsize=l_font)
ax0[0].set_title('Phase Function Resultant from\n Incident Horizontally Polarized Light', fontsize=t_font)
ax0[0].grid(True)
ax0[0].legend(loc=1, fontsize=l_font)
ax0[1].semilogy(theta_scattered, SR_tmat_norm, color='yellow', ls='-', label='T-Matrix: SR\n axis ratio: ' + str('{:.2f}'.format(ax_r)))
ax0[1].semilogy(theta_mie, SR_mie_norm, color='green', ls='--', label='Mie Theory: SR')
ax0[1].set_xlabel('\u0398(\u00b0)', fontsize=l_font)
ax0[1].set_ylabel('Normalized Intensity', fontsize=l_font)
ax0[1].set_title('Phase Function Resultant from\n Incident Vertically Polarized Light', fontsize=t_font)
ax0[1].grid(True)
ax0[1].legend(loc=1, fontsize=l_font)
plt.tight_layout()
plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_' + str('{:.2f}'.format(ax_r)) + '.pdf', format='pdf')
plt.savefig(save_dir + 'T-Matrix & Mie Theory axis_ratio_' + str('{:.2f}'.format(ax_r)) + '.png', format='png')
plt.show()
# setting the psd class attributes
#scatterer.psd.BinnedPSD(bin_edges=1024, bin_psd=1025,  D=900)
#scatterer.PSDIntegrator(num_points=1024, m_func =None, axis_ratio_func=None, geometries=(90.0, 90.0, 0.0, 180.0, 0.0, 0.0))

print('SL T-Matrix Sum: ', np.sum(SL_tmat_norm))
print('SL Mie Sum: ', np.sum(SL_mie_norm))
print('SR T-Matrix Sum: ', np.sum(SR_tmat_norm))
print('SR Mie Sum: ', np.sum(SR_mie_norm))
'''

