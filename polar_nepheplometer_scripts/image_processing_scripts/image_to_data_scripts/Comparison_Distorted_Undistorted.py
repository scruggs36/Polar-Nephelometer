'''
Austen K. Scruggs
05-02-2019
Description: A script to compare the undistorted and barrel distorted image data
'''


from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, argrelmin
from scipy.interpolate import interp1d, pchip_interpolate
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dist_directory = '/home/austen/Documents/Particle_Data/Warped/SD_Particle.txt'
undist_directory = '/home/austen/Documents/Particle_Data/Unwarped/SD_Particle.txt'
save_directory = '/home/austen/Documents'


dist_data = pd.read_csv(dist_directory, sep=',', header=0)
undist_data = pd.read_csv(undist_directory, sep=',', header=0)


Bright_CO2_PN = np.array(dist_data['Bright CO2 Columns'])
Bright_CO2_PF = np.array(dist_data['Bright CO2 Intensity'])
Bright_CO2_PN_UND = np.array(undist_data['Bright CO2 Columns'])
Bright_CO2_PF_UND = np.array(undist_data['Bright CO2 Intensity'])


correction = (Bright_CO2_PN_UND[np.argmin(Bright_CO2_PF_UND)] - Bright_CO2_PN[np.argmin(Bright_CO2_PF)])/2

Bright_CO2_PN_Corr = Bright_CO2_PN + correction


arr_pn = []
arr_pf = []
for counter, element in enumerate(Bright_CO2_PF):
    if element <= np.amax(Bright_CO2_PF_UND):
        print(element)
        arr_pn.append(Bright_CO2_PN[counter])
        arr_pf.append(element)


def rayleigh_scattering(a, b):
    return a * b * (1 + np.square(np.cos(np.linspace(0.0, 180.0, 1.0) * (180.0/pi))))


f0, ax0 = plt.subplots(figsize=(12, 6))
ax0.plot(Bright_CO2_PN_Corr, Bright_CO2_PF, linestyle='-', color='red', label='CO2 Scattering Distorted')
ax0.plot(Bright_CO2_PN_UND, Bright_CO2_PF_UND, linestyle='-', color='blue', label='CO2 Scattering Undistorted')
ax0.set_title('CO2 Rayleigh Scattering')
ax0.set_ylabel('Intensity (DN)')
ax0.set_xlabel('Profile Number (Column Number)')
ax0.grid(True)
ax0.legend(loc=1)
plt.tight_layout()
f0.savefig(save_directory + 'Dist_V_Undist.png', format='png')
plt.show()

'''
# 1 + cos^2(theta) fits
popt_CO2, pcov_CO2 = curve_fit(rayleigh_scattering, theta_CO2, SD_CO2)
popt_N2, pcov_N2 = curve_fit(rayleigh_scattering, theta_N2, SD_N2)

# plug into function for each angle
rayleigh_cos_N2 = np.array([rayleigh_scattering(rad, *popt_N2) for rad in rads_N2])
rayleigh_cos_CO2 = np.array([rayleigh_scattering(rad, *popt_CO2) for rad in rads_CO2])
rayleigh_ideal_cos = np.array([1 + (np.cos(rad) ** 2) for rad in rads_CO2])


# take the ratio of the data to the fit
ral_CO2 = np.array([rayleigh_scattering(rad, *popt_CO2) for rad in rads_CO2])
ratio_CO2 = np.array(SD_CO2) / ral_CO2
ral_N2 = np.array([rayleigh_scattering(rad, *popt_N2) for rad in rads_N2])
ratio_N2 = np.array(SD_N2) / ral_N2
ratio_CO2_N2 = np.array(SD_CO2_gfit_SG) / np.array(SD_N2_gfit_SG)
ratio_CO2_N2_normed = (np.array(SD_CO2_gfit_SG)/np.linalg.norm(np.array(SD_CO2_gfit_SG))) / (np.array(SD_N2_gfit_SG)/np.linalg.norm(np.array(SD_N2_gfit_SG)))
#print(ratio_CO2_N2)
ratio_CO2_min = ratio_CO2 / np.amin(ratio_CO2)
ratio_N2_min = ratio_N2 / np.amin(ratio_N2)
ratio_ideal_CO2 = np.array(SD_CO2) / rayleigh_ideal_cos
ratio_ideal_N2 = np.array(SD_N2) / rayleigh_ideal_cos
ideal_CO2_N2_ratio = ral_CO2 / ral_N2


# filters and pchips
ratio_CO2_min_savgol = savgol_filter(ratio_CO2_min, window_length=151, polyorder=2, deriv=0)
ratio_N2_min_savgol = savgol_filter(ratio_N2_min, window_length=151, polyorder=2, deriv=0)
ratio_CO2_min_pchip = pchip_interpolate(theta_CO2, ratio_CO2_min_savgol, cols_array, der=0, axis=0)
ratio_N2_min_pchip = pchip_interpolate(theta_N2, ratio_N2_min_savgol, cols_array, der=0, axis=0)




# we did not do a semilogy plot here, y data cannot have zeros in it! log of zero is not defined!!!
f8, ax8 = plt.subplots(figsize=(12, 6))
ax8.plot(theta_CO2, SD_CO2, linestyle='-', color='black', label='$CO_2$ Scattering - He Scattering')
ax8.plot(theta_CO2, SD_CO2_gfit_bkg_corr, linestyle='--', color='purple', label='$CO_2$ Scattering - He Scattering')
ax8.plot(theta_CO2, rayleigh_cos_CO2, linestyle='--', color='yellow', label='CO2 Scattering fit')
ax8.plot(theta_N2, SD_N2, linestyle='-', color='green', label='$N_2$ Scattering - He Scattering')
ax8.plot(theta_N2, SD_N2_gfit_bkg_corr, linestyle='--', color='lawngreen', label='$N_2$ Scattering - He Scattering')
ax8.plot(theta_N2, rayleigh_cos_N2, linestyle='--', color='orange', label='N2 Scattering fit')
ax8.plot(theta_N2, SD_He, linestyle='-', color='cyan', label='He Scattering - Background')
ax8.plot(theta_N2, SD_BKG, linestyle='-', color='red', label='Background')
ax8.set_title('Corrected Scattering Diagram')
ax8.set_ylabel('Intensity (DN)')
ax8.set_xlabel('Profile Number (Column Number)')
ax8.grid(True)
ax8.legend(loc=1)
plt.tight_layout()
f8.savefig(Path_Save + '/F8_Contributions_Corr.png', format='png')
plt.show()

polynomial_fit_comp_CO2 = np.poly1d(np.polyfit(theta_CO2, ratio_CO2, deg=4))
fit_comp_ratio_CO2 = np.array([polynomial_fit_comp_CO2(element) for element in theta_CO2])
CO2_Compression_Corrected_Ratio = ratio_CO2 / fit_comp_ratio_CO2
polynomial_fit_comp_N2 = np.poly1d(np.polyfit(theta_N2, ratio_N2, deg=4))
fit_comp_ratio_N2 = np.array([polynomial_fit_comp_N2(element) for element in theta_N2])
N2_Compression_Corrected_Ratio = ratio_N2 / fit_comp_ratio_N2
f9, ax9 = plt.subplots(1, 3, figsize=(20, 6))

ax9[0].plot(theta_CO2, rayleigh_ideal_cos, linestyle='-', color='red', label='$(1 + cos^2(\u0398)$')
ax9[0].grid(True)
ax9[0].set_xlabel('\u0398')
ax9[0].set_ylabel('Ratio')
ax9[0].set_title('1 + $cos^2(\u0398)$')
ax9[0].legend(loc=1)
ax9[1].plot(theta_CO2, ratio_CO2, linestyle='-', color='red', label='$CO_2$/$(ab(1 + cos^2(\u0398)$ \n a = ' + str('{:.2f}'.format(popt_CO2[0])) + ' b = ' + str('{:.2f}'.format(popt_CO2[1])))
ax9[1].plot(theta_CO2, fit_comp_ratio_CO2, linestyle='--', color='orange', label='fit $CO_2$/$(1 + cos^2(\u0398)$')
ax9[1].plot(theta_N2, ratio_N2, linestyle='-', color='blue', label='$N_2$/$(ab(1 + cos^2(\u0398)$ \n a = ' + str('{:.2f}'.format(popt_N2[0])) + ' b = ' + str('{:.2f}'.format(popt_N2[1])))
ax9[1].plot(theta_N2, fit_comp_ratio_N2, linestyle='--', color='cyan', label='fit $N_2$/$(1 + cos^2(\u0398)$')
ax9[1].plot(theta_CO2, ratio_CO2_N2, linestyle='-', color='green', label='$CO_2$/$N_2$')
ax9[1].plot(theta_CO2, ratio_CO2_N2_normed, linestyle='-', color='purple', label='$CO_2$/$N_2$ normed')
ax9[1].plot(theta_CO2, ideal_CO2_N2_ratio, linestyle='-', color='black', label='$CO_2 fit$/$N_2 fit$')
ax9[1].plot(theta_CO2, CO2_Compression_Corrected_Ratio, linestyle='--', color='lawngreen', label='Compression Corrected $CO_2$')
ax9[1].plot(theta_N2, N2_Compression_Corrected_Ratio, linestyle='--', color='yellow', label='Compression Corrected $N_2$')
ax9[1].grid(True)
ax9[1].set_xlabel('\u0398')
ax9[1].set_ylabel('Ratio')
ax9[1].set_title('Ratio Plot \n Rayleigh Scattering : ab(2 + $cos^2(\u0398)$')
ax9[1].legend(loc=1)
ax9[2].plot(theta_CO2, ratio_ideal_CO2, linestyle='-', color='red', label='$CO_2$/$(1 + cos^2(\u0398)$')
ax9[2].plot(theta_N2, ratio_ideal_N2, linestyle='-', color='blue', label='$N_2$/$(1 + cos^2(\u0398)$')
ax9[2].plot(theta_CO2, ratio_CO2_N2, linestyle='-', color='green', label='$CO_2$/$N_2$')
ax9[2].plot(theta_CO2, ideal_CO2_N2_ratio, linestyle='-', color='purple', label='$CO_2 fit$/$N_2 fit$')
ax9[2].grid(True)
ax9[2].set_xlabel('\u0398')
ax9[2].set_ylabel('Ratio')
ax9[2].set_title('Ratio Plot \n Rayleigh Scattering : 1 + $cos^2(\u0398)$')
ax9[2].legend(loc=1)
plt.tight_layout()
f9.savefig(Path_Save + '/F9_Ratio.png', format='png')
plt.show()

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

SD_CO2_Compression_Corrected = SD_CO2 / fit_comp_ratio_CO2
SD_N2_Compression_Corrected = SD_N2 / fit_comp_ratio_N2

f10, ax10 = plt.subplots(1, 2, figsize=(12, 6))
ax10[0].plot(theta_CO2, SD_CO2, linestyle='-', color='black', label='$CO_2$ Scattering - He Scattering')
ax10[0].plot(theta_CO2, SD_CO2_gfit_bkg_corr, linestyle='--', color='purple', label='$CO_2$ Scattering - He Scattering')
ax10[0].plot(theta_CO2, SD_CO2_Compression_Corrected, linestyle='-', color='blue', label='$CO_2$ Scattering - He Scattering Comp. Corrected')
ax10[0].plot(theta_CO2, rayleigh_cos_CO2, linestyle='--', color='cyan', label='CO2 Scattering fit')
ax10[0].set_title('$CO_2$ Corrected Scattering Diagram ')
ax10[0].set_ylabel('Intensity (DN)')
ax10[0].set_xlabel('\u0398')
ax10[0].grid(True)
ax10[0].legend(loc=1)
ax10[1].plot(theta_N2, SD_N2, linestyle='-', color='green', label='$N_2$ Scattering - He Scattering')
ax10[1].plot(theta_N2, SD_N2_gfit_bkg_corr, linestyle='--', color='lawngreen', label='$N_2$ Scattering - He Scattering Comp Corrected')
ax10[1].plot(theta_N2, SD_N2_Compression_Corrected, linestyle='-', color='pink', label='$CO_2$ Scattering - He Scattering')
ax10[1].plot(theta_N2, rayleigh_cos_N2, linestyle='--', color='orange', label='N2 Scattering fit')
#ax10[1].plot(theta_N2, SD_He, linestyle='-', color='cyan', label='He Scattering - Background')
#ax10[1].plot(theta_N2, SD_BKG, linestyle='-', color='red', label='Background')
ax10[1].set_title('$N_2$ Corrected Scattering Diagram')
ax10[1].set_ylabel('Intensity (DN)')
ax10[1].set_xlabel('\u0398')
ax10[1].grid(True)
ax10[1].legend(loc=1)
plt.tight_layout()
f10.savefig(Path_Save + '/F10_Intensity_Corr.png', format='png')
plt.show()
'''