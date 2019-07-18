'''
Austen K. Scruggs
02/12/2019
Description: Using our measured phase function to retrieve the real refractive index "n" by comparing to Mie Theory.

PSLs:
Mean    Mean Uncertainty     Size Dist Sigma
600nm     9nm                     10.0nm
800nm     14nm                    5.6nm
903nm     12nm                    4.1nm
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyMieScatt as ps
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from scipy.interpolate import pchip_interpolate

# save directory
save_directory = '/home/austen/Documents/'

# import data
data_directory = '/home/austen/Documents/01-23-2019_Analysis/'
data = pd.read_csv(data_directory + 'SD_Offline_800.txt')
data_sample_intensity = np.array(data['Sample Intensity']) - np.array(data['Nitrogen Intensity'])
data_nitrogen_intensity = np.array(data['Nitrogen Intensity'])
data_profiles = np.array(data['Columns'])

# smooth data
data_sample_intensity_savgol = savgol_filter(data_sample_intensity, window_length=151, polyorder=2, deriv=0)

# converting pixel number to angle
slope = 0.2045
# the intercept has the power to adjust your theta space by a constant, basically, sliding it to the left or to the right!
# i think that there are different degrees of saturation among the different size PSLs causing differences in the y intercept needed for accurate
# pixel to angle transformation
intercept = -41.9764
experiment_theta = (slope * data_profiles) + intercept

# import experimental data
#Data = pd.read_csv(Data_Directory, sep=',', header=0)
# Particle diameter, geometric mean of the particle diameter
d = 800
# particle size standard deviation
sigma_s = 5.6
# define Gaussian function
def Gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

# size distribution plot
size_axis = np.arange(d - (sigma_s * 3), d + (sigma_s * 3), 1)
Gaussian_Data = Gaussian(size_axis, mu=d, sigma=sigma_s)
#print(sp.integrate.simps(Gaussian(size_axis, mu=d, sigma=sigma_g), size_axis, dx=1))
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.plot(size_axis, Gaussian_Data, 'b-', label='Gaussian Dist. \u03bc=' + str(d) + ', \u03c3=' + str(sigma_s))
ax1.set_xlabel('particle diameter (nm)')
ax1.set_ylabel('Normalized $dN/Log_{10}(D)$')
ax1.set_title('Distributions Used for Mie Theory Calculations')
ax1.grid(True)
plt.legend(loc=1)
plt.savefig(save_directory + 'Mie_Distributions.pdf', format='pdf')
plt.savefig(save_directory + 'Mie_Distributions.png', format='png')
#plt.show()
plt.close()

'''
We will used the normalized distribution intensities at each particle diameter as weights 
when we sum all our scattering diagrams, here we just collect the distribution information in array of arrays
'''

Weights_Gaussian = []
size_array = []
weights_array = []
for counter, element in enumerate(range(len(Gaussian_Data))):
    Weights_Gaussian.append([size_axis[counter], Gaussian_Data[counter]])
    size_array.append(size_axis[counter])
    weights_array.append(Gaussian_Data[counter])

# wavelength in centimeters
w_c = 663E-7

# wavelength in microns
w_u = .663

# wavelength in nanometers
w_n = 663

# wavelength in angstroms
w_a = 6630

# UV-Visible spectrum in angstroms
uv_visible_spectrum_centimeters = np.arange(300E-7, 1050E-7, 10E-7)
#print(len(uv_visible_spectrum_centimeters))

# UV-Visible spectrum in microns
uv_visible_spectrum_microns = np.arange(.300, 1.060, .010)

# UV-Visible spectrum in nanometers
uv_visible_spectrum_nanometers = np.arange(300, 1060, 10)
#print(len(uv_visible_spectrum_nanometers))

# UV-Visible spectrum in angstroms
uv_visible_spectrum_angstroms = np.arange(3000, 10600, 100)



# complex refractive index, Cauchy parameters for PSL Matheson & Sanderson 1952, wavelength in microns from Greenslade
A0 = 1.5663
B0 = 0.00785
C0 = 0.000334

# complex refractive index, Cauchy parameters for PSL Bateman 1959 wavelength in centimeters
A1 = 1.5683
B1 = 10.087E-11

# defining cauchy equation functions
def cauchy_2term(wav, A_2term, B_2term):
    return A_2term + (B_2term / wav ** 2)


def cauchy_4term(wav, A_4term, B_4term, C_4term):
    return A_4term + (B_4term / wav ** 2) + (C_4term / (wav ** 4))


def cauchy_6term(wav, A_1_6term, A_2_6term, A_3_6term, A_4_6term, A_5_6term, A_6_6term):
    return np.sqrt(A_1_6term + (A_2_6term * wav ** 2) + (A_3_6term / wav ** 2) + (A_4_6term / (wav ** 4)) + (A_5_6term / (wav ** 6)) + (A_6_6term / (wav ** 8)))


# fit Nikalov data for cauchy coefficients, wavelength in microns
w_Nikalov_microns = np.array([0.436, 0.486, 0.546, 0.588, 0.633, 0.656, 0.703, 0.752, 0.804, 0.833, 0.879, 1.052])
w_Nikalov_nanometers = w_Nikalov_microns * 1000
n_RI_Nikalov = np.array([1.617, 1.606, 1.596, 1.592, 1.587, 1.586, 1.582, 1.579, 1.578, 1.577, 1.576, 1.572])
#print(w_Nikalov_microns)
popt_Nikalov, pcov_Nikalov = curve_fit(cauchy_6term, w_Nikalov_microns, n_RI_Nikalov, p0=[2.44675093, -1.011623E-3, 2.840749E-2, -3.761631E-4, 8.193491E-5, 9.055861E-4])
#print(popt_Nikalov)
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

# n of refractive index for PSL as cauchy equation at a specific wavelength
n_cauchy0 = A0 + (B0 / w_u ** 2) + (C0 / (w_u ** 4))
n_cauchy1 = A1 + (B1 / w_c ** 2)
n_cauchy2 = np.sqrt(A2 + (B2 * w_u ** 2) + (C2 / w_u ** 2) + (D2 / (w_u ** 4)) + (E2 / (w_u ** 6)) + (F2 / (w_u ** 8)))
n_cauchy3 = A3 + (B3 / w_u ** 2) + (C3 / (w_u ** 4))
n_cauchy4 = np.sqrt(A4 + (B4 * w_u ** 2) + (C4 / w_u ** 2) + (D4 / (w_u ** 4)) + (E4 / (w_u ** 6)) + (F4 / (w_u ** 8)))
n_cauchy5 = np.sqrt(A5 + (B5 * w_u ** 2) + (C5 / w_u ** 2) + (D5 / (w_u ** 4)) + (E5 / (w_u ** 6)) + (F5 / (w_u ** 8)))
n_cauchy6 = A6 + (B6 / w_u ** 2) + (C6 / (w_u ** 4))
n_cauchy7 = A7 + (B7 / w_n ** 2) + (C7 / (w_n ** 4))
n_cauchy8 = A8 + (B8 / w_u ** 2) + (C8 / (w_u ** 4))
n_sellmeier9 = np.sqrt(1 + ((B9 * w_n ** 2)/(w_n ** 2 - wav9 ** 2)))

# basic statistics
n_all_groups = [n_cauchy0, n_cauchy1, n_cauchy2, n_cauchy3, n_cauchy4, n_cauchy5, n_cauchy6, n_cauchy7, n_cauchy8, n_sellmeier9]
n_mean = np.mean(n_all_groups)
n_percentiles = np.percentile(n_all_groups, [0, 25, 50, 75, 100])


# n (real refractive index) over many wavelengths
n_cauchy_v_wav0 = [A0 + (B0 / element ** 2) + (C0 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav1 = [A1 + (B1 / element ** 2) for element in uv_visible_spectrum_centimeters]
n_cauchy_v_wav2 = [np.sqrt(A2 + (B2 * element ** 2) + (C2 / element ** 2) + (D2 / (element ** 4)) + (E2 / (element ** 6)) + (F2 / (element ** 8))) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav3 = [A3 + (B3 / element ** 2) + (C3 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav4 = [np.sqrt(A4 + (B4 * element ** 2) + (C4 / element ** 2) + (D4 / (element ** 4)) + (E4 / (element ** 6)) + (F4 / (element ** 8))) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav5 = [np.sqrt(A5 + (B5 * element ** 2) + (C5 / element ** 2) + (D5 / (element ** 4)) + (E5 / (element ** 6)) + (F5 / (element ** 8))) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav6 = [A6 + (B6 / element ** 2) + (C6 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_cauchy_v_wav7 = [A7 + (B7 / element ** 2) + (C7 / (element ** 4)) for element in uv_visible_spectrum_nanometers]
n_cauchy_v_wav8 = [A8 + (B8 / element ** 2) + (C8 / (element ** 4)) for element in uv_visible_spectrum_microns]
n_sellmeier_v_wav9 = [np.sqrt(1 + ((B9 * element ** 2)/(element ** 2 - wav9 ** 2))) for element in uv_visible_spectrum_nanometers]

fig2, ax2 = plt.subplots(2, 2, figsize=(20, 20))
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav0, label='Matheson et. al. 1952')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav1, label='Bateman et. al. 1959')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav2, label='Nikalov et. al. 2000 Fit')
ax2[0, 0].plot(w_Nikalov_nanometers, n_RI_Nikalov, marker='o', ms=3, ls='', label='Nikalov et. al. 2000 Meas.')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav3, label='Ma et. al. 2003')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav4, label='Sultanova et. al. 2003')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav5, label='Kasarova et. al. 2006')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav6, label='Miles et. al. 2010')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav7, label='Jones et. al. 2013')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_cauchy_v_wav8, label='Greenslade et. al. 2017')
ax2[0, 0].plot(uv_visible_spectrum_nanometers, n_sellmeier_v_wav9, label='Gienger et. al. 2017')
ax2[0, 0].set_title('Polystyrene Latex Sphere Real Refractive Index (n) \n as a Function of Wavelength')
ax2[0, 0].set_xlabel('Wavelength (nm)')
ax2[0, 0].set_ylabel('n')
ax2[0, 0].grid(True)
ax2[0, 0].legend(loc=1)
ax2[0, 1].plot(663, n_cauchy0, marker='o', ms=3, ls='', label='Matheson et. al. 1952')
ax2[0, 1].plot(663, n_cauchy1, marker='o', ms=3, ls='', label='Bateman et. al. 1959')
ax2[0, 1].plot(663, n_cauchy2, marker='o', ms=3, ls='', label='Nikalov et. al. 2000')
ax2[0, 1].plot(663, n_cauchy3, marker='o', ms=3, ls='', label='Ma et. al. 2003')
ax2[0, 1].plot(663, n_cauchy4, marker='o', ms=3, ls='', label='Sultanova et. al. 2003')
ax2[0, 1].plot(663, n_cauchy5, marker='o', ms=3, ls='', label='Kasarova et. al. 2006')
ax2[0, 1].plot(663, n_cauchy6, marker='o', ms=3, ls='', label='Miles et. al. 2010')
ax2[0, 1].plot(663, n_cauchy7, marker='o', ms=3, ls='', label='Jones et. al. 2013')
ax2[0, 1].plot(663, n_cauchy8, marker='o', ms=3, ls='', label='Greenslade et. al. 2017')
ax2[0, 1].plot(663, n_sellmeier9, marker='o', ms=3, ls='', label='Gienger et. al. 2017')
ax2[0, 1].set_title('All the Groups Values at 663nm')
ax2[0, 1].set_xlabel('Wavelength (nm)')
ax2[0, 1].set_ylabel('n')
ax2[0, 1].grid(True)
ax2[0, 1].legend(loc=1)
ax2[1, 0].boxplot(n_all_groups)
ax2[1, 0].set_xticklabels(['Wavelength 663nm'])
ax2[1, 0].set_title('Box Plot Statistics of All the Groups Values at 663nm')
ax2[1, 0].grid(True)

# k, imaginary part of RI
k0 = .0003j
k1 = .0003j
k2 = .0003j
k3 = .0003j
k4 = .0003j
k5 = .0003j
k6 = .0003j
k7 = .0003j
k8 = .0003j
k9 = .0003j

# create RI

m0 = n_cauchy0 + k0
print('Matheson 1952 RI: ', m0)
m1 = n_cauchy1 + k1
print('Bateman 1959 RI: ', m1)
m2 = n_cauchy2 + k2
print('Nikalov 2000 RI: ', m2)
m3 = n_cauchy3 + k3
print('Ma 2003 RI: ', m3)
m4 = n_cauchy4 + k4
print('Sultanova 2003 RI: ', m4)
m5 = n_cauchy5 + k5
print('Kasarova 2006 RI: ', m5)
m6 = n_cauchy6 + k6
print('Miles 2010 RI: ', m6)
m7 = n_cauchy7 + k7
print('Jones 2013 RI: ', m7)
m8 = n_cauchy8 + k8
print('Greenslade 2017 RI: ', m8)
m9 = n_sellmeier9 + k9
print('Gienger 2017 RI: ', m9)
m_list = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]
group_list = ['Matheson et. al. 1952', 'Bateman et. al. 1959', 'Nikalov et. al. 2000', 'Ma et. al. 2003', 'Sultanova et. al. 2003', 'Kasarova et. al. 2006', 'Miles et. al. 2010', 'Jones et. al. 2013', 'Greenslade et. al. 2017', 'Gienger et. al. 2017']
#

# initializing lists
scalar_array = []
for counter, m in enumerate(m_list):
    theta_2darray = []
    SL_2darray = []
    SR_2darray = []
    SU_2darray = []
    for size_bin in size_axis:
        theta, SL, SR, SU = ps.ScatteringFunction(m, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        theta_2darray.append(theta)
        # SL_2darray.append(SL)
        # SR_2darray.append(SR)
        SU_2darray.append(SU)
    theta_avg = np.average(theta_2darray, axis=0)
    #SL_avg = np.average(SL_2darray, axis=0, weights=weights_array)
    #SR_avg = np.average(SR_2darray, axis=0, weights=weights_array)
    SU_avg = np.average(SU_2darray, axis=0, weights=weights_array)
    SU_Pchip = pchip_interpolate(theta_avg, SU_avg, experiment_theta, der=0, axis=0)
    minima_calc = np.argmin(SU_Pchip)
    minima_meas = np.argmin(data_sample_intensity_savgol)
    scalar = SU_Pchip[minima_calc] / data_sample_intensity_savgol[minima_meas]
    scalar_array.append(scalar)
    ax2[1, 1].semilogy(experiment_theta, SU_Pchip, ls='-', label=str(group_list[counter]))
average_scalar = np.mean(scalar_array)
ax2[1, 1].semilogy(experiment_theta, data_sample_intensity_savgol * average_scalar, color='black', ls='-', label='Measurement')
ax2[1, 1].set_xlabel('\u03b8')
ax2[1, 1].set_ylabel('$|S^2|$')
ax2[1, 1].set_title('Computed Scattering Diagrams from Literature \n Cauchy Equations')
ax2[1, 1].grid(True)
ax2[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(save_directory + 'N_vs_Wav&SD.pdf', format='pdf')
plt.savefig(save_directory + 'N_vs_Wav&SD.png', format='png')
#plt.show()
plt.close()

# create real refractive index space
n_space = np.linspace(1.50, 1.70, 10)

df = pd.DataFrame()
df_theta = []
df_su = []
df_chi = []
df_pval = []
df_m = []
# retrieval based on X^2 minimization
for counter, n in enumerate(n_space):
    percent_progress = (counter/len(n_space)) * 100
    print('percent progress for refractive index retrieval: ', percent_progress)
    theta_2darray = []
    #SL_2darray = []
    #SR_2darray = []
    SU_2darray = []
    k = .0003j
    m_input = n + k
    for size_bin in size_axis:
        theta, SL, SR, SU = ps.ScatteringFunction(m_input, w_n, size_bin, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
        theta_2darray.append(theta)
        # SL_2darray.append(SL)
        # SR_2darray.append(SR)
        SU_2darray.append(SU)
    theta_avg = np.average(theta_2darray, axis=0)
    # SL_avg = np.average(SL_2darray, axis=0, weights=weights_array)
    # SR_avg = np.average(SR_2darray, axis=0, weights=weights_array)
    SU_avg = np.average(SU_2darray, axis=0, weights=weights_array)
    SU_pchip = pchip_interpolate(theta_avg, SU_avg, experiment_theta, der=0, axis=0)
    chi, pval = chisquare(f_obs=data_sample_intensity_savgol[250:-1] * average_scalar, f_exp=SU_pchip[250:-1], axis=0)
    df_theta.append(experiment_theta)
    df_su.append(SU_pchip)
    df_chi.append(chi)
    df_pval.append(pval)
    df_m.append(m_input)

# find Chi^2 global minimum
min_index = np.argmin(df_chi)

df['Theta'] = df_theta
df['SU'] = df_su
df['Refractive Index'] = df_m
df['Chi Min'] = df_chi
df['Pval'] = df_pval

fig3, ax3 = plt.subplots(1, 2, figsize=(20, 7))
ax3[0].semilogy(df_theta[min_index], df_su[min_index], color='red', ls='-', label='$|X|^2: $' + str(df_chi[min_index]) + ' RI: ' + str(df_m[min_index]))
ax3[0].semilogy(df_theta[min_index][250:-1], df_su[min_index][250:-1], color='orange', ls='-', label='$|X|^2$ Minimization: Mie Subset')
ax3[0].semilogy(experiment_theta, data_sample_intensity_savgol * average_scalar, color='black', ls='-', label='Measurement')
ax3[0].semilogy(experiment_theta[250:-1], data_sample_intensity_savgol[250:-1] * average_scalar, color='grey', ls='-', label='$|X|^2$ Minimization: Measurement Subset')
ax3[0].set_xlabel('\u03b8')
ax3[0].set_ylabel('$|S|^2$')
ax3[0].set_title('Best Minimization Result')
ax3[0].grid(True)
ax3[0].legend(loc=1)
ax3[1].plot(np.real(df_m), df_chi, color='blue', marker='o', ms='2', label='$|X|^2$ vs n')
ax3[1].plot(np.real(df_m)[min_index], df_chi[min_index], color='red', ls='',  marker='x', ms='4', label='$|X|^2$ Global Min')
ax3[1].set_xlabel('n')
ax3[1].set_ylabel('$|X|^2$')
ax3[1].set_title('Minimization Space')
ax3[1].grid(True)
ax3[1].legend(loc=1)
plt.tight_layout()
plt.savefig(save_directory + 'Chi_Best.pdf', format='pdf')
plt.savefig(save_directory + 'Chi_Best.png', format='png')
#plt.show()
plt.close()

