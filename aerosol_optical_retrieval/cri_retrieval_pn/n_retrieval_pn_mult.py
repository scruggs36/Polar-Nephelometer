'''
Austen K. Scruggs
07-19-2019
Description: Retrieve n for a given lognormal size distribution
'''

import PyMieScatt as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import chisquare
import datetime
from math import log, sqrt, pi


# generate artificial data, this is the creation of the log normal distribution

# write function for LogNormal distribution (same as Tami Bonds)
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))


def Retrieval_v0(path, number_density, geom_stdev, diameter, sizes, wavelength, m_known,  added_noise_array, n_space, k_space):
    pf_2darray = []
    pf_2darray_R = []
    pf_arr = []
    theta_2darray = []
    m_R_space = []
    n_R_space = []
    k_R_space = []
    pf_space = []
    chi_square_space = []
    chi_arr = []
    # create distribution data
    log_dist = np.array([LogNormal(element, diameter, geom_stdev, number_density) for element in sizes])
    print(log_dist.shape)

    for s in added_noise_array:
        pf_2darray = []
        pf_2darray_R = []
        chi_square_space = []
        now = datetime.datetime.now()
        data_path = path + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '_' + str(now.second) + '_' + 'sigma_' + str(s) + '/'
        try:
            os.makedirs(os.path.expanduser(data_path))
        except FileExistsError:
            print('File already exists')

        # plot distribution
        f0, ax0 = plt.subplots(figsize=(6, 12))
        ax0.plot(sizes, log_dist, 'r-', label='lognormal dist.: \u03bc=' + str(d) + ' $\u03c3_{g}$=' + str(sigma_g))
        ax0.set_xlabel('particle diameter (nm)')
        ax0.set_ylabel('dN/dD')
        ax0.set_title('Log Normal Distribution')
        ax0.grid(True)
        ax0.legend(loc=1)
        plt.savefig(data_path + 'size_distribution.pdf', format='pdf')
        # plt.show()

        for element in sizes:
            theta, SL, SR, SU = ps.ScatteringFunction(m_known, wavelength, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
            pf_2darray.append(SU)
        print(np.array(pf_2darray).shape)
        # calculate weighted average of phase function data based on weights determined from the lognormal distribution
        pf_average = np.average(pf_2darray, axis=0, weights=log_dist)
        # creating noise in phase functions
        noise_mu = np.mean(0)
        noise_sigma = s
        noise = np.random.normal(noise_mu, noise_sigma, len(pf_average))
        # Noise up the original signal
        pf_average = np.add(pf_average, np.multiply(pf_average, noise / 100))
        # plot phase functions that comprise the weighted average phase function
        f1, ax1 = plt.subplots(1, 2, figsize=(6, 12))
        for counter, element in enumerate(pf_2darray):
            ax1[0].semilogy(theta, element, label='P.F. @ size: ' + str(sizes[counter]))
        ax1[0].legend(loc=1, ncol=3)
        ax1[0].set_title('Phase Functions Used to Create Weighted Average Phase Function')
        ax1[0].set_xlabel('\u03b8 (\u0b00)')
        ax1[0].set_ylabel('Intensity')
        ax1[0].grid(True)
        ax1[1].semilogy(theta, pf_average, label='Weighted Average Phase Function')
        ax1[1].legend(loc=1)
        ax1[1].set_title('Weighted Average Phase Function')
        ax1[1].set_xlabel('\u03b8 (\u00b0)')
        ax1[1].set_ylabel('Intensity')
        ax1[1].grid(True)
        plt.savefig(data_path + 'generated_pfs.pdf', format='pdf')
        #plt.show()

        # i am right here, just changed names to n_space and k_space
        for n in n_space:
            for k in k_space:
                m_R = complex(n, k)
                print(m_R)
                m_R_space.append(m_R)
                n_R_space.append(n)
                k_R_space.append(k)
                for element in sizes:
                    theta_R, SL_R, SR_R, SU_R = ps.ScatteringFunction(m_R, wavelength, element, nMedium=1.0, minAngle=0, maxAngle=180, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization=None)
                    pf_2darray_R.append(SU_R)
                pf_average_R = np.average(pf_2darray_R, axis=0, weights=log_dist)
                # the std will be different when using real data, for now we set it to the difference between data and model at each point
                std = np.abs(pf_average - pf_average_R)
                # chi square = residual divided by standard deviation at each angle, we call it chi square as the is value has a chi square distribution, but it is not a chi square statistic
                chisq = np.sum((pf_average - pf_average_R)**2 / std)
                #chisq = np.sum((pf_average - pf_average_R) ** 2)
                pf_arr.append(pf_average_R)
                chi_arr.append(chisq)
                pf_2darray_R = []
            chi_square_space.append(chi_arr)
            pf_space.append(pf_arr)
            chi_arr = []
            pf_arr = []

        chi_square_space = np.transpose(chi_square_space)
        k_idx_min, n_idx_min = np.where(chi_square_space == np.min(chi_square_space))
        k_idx_min = int(k_idx_min)
        n_idx_min = int(n_idx_min)
        #print(np.array(chi_square_space).shape)
        X, Y = np.meshgrid(n_space, k_space)
        f2, ax2 = plt.subplots(1, 2, figsize=(12, 12))
        #print(X.shape)
        #print(X)
        ax2[0].set_title('Chi Square Minimization  \n Minimum Value @' + str(n_space[n_idx_min]) + ' + ' + str(k_space[k_idx_min]) + 'j')
        c1 = ax2[0].pcolormesh(X, Y, chi_square_space, cmap='rainbow')
        ax2[0].set_xlabel('n')
        ax2[0].set_ylabel('k')
        f2.colorbar(c1, ax=ax2[0])
        ax2[0].plot(n_space[n_idx_min], k_space[k_idx_min], color='black', marker='o', ms=16)
        ax2[1].semilogy(theta_R, pf_average, color='black', ls='-', label='Input Phase Function')
        ax2[1].semilogy(theta_R, pf_space[n_idx_min][k_idx_min], color='red', ls='--', label='Retrieved Phase Function')
        ax2[1].set_title('Input and Retrieved Phase Functions')
        ax2[1].set_xlabel('\u03b8 (\u00b0)')
        ax2[1].set_ylabel('Intensity')
        ax2[1].legend(loc=1)
        ax2[1].grid(True)
        plt.savefig(data_path + 'retrieval_pfs.pdf', format='pdf')

        X_array = []
        Y_array = []
        chi_square_array = []
        df1 = pd.DataFrame()
        for element in X:
            X_array.append(element)
        df1['n space'] = X_array
        for element in Y:
            Y_array.append(element)
        df1['k space'] = Y_array
        for element in chi_square_space:
            chi_square_array.append(element)
        df1['Chi Square Statistic'] = chi_square_array
        df1.to_csv(data_path + 'CHI_DATA_NOISE_@' + str(s) + '.txt')
        df2 = pd.DataFrame()
        df2['Theta'] = theta_R
        df2['Input Phase Function'] = pf_average
        df2.to_csv(data_path + 'PF_INPUT_NOISE_@' + str(s) + '.txt')
        #np.savetxt(path + 'PF_SPACE_@' + str(s) + '.txt', pf_space, delimiter=',')



py_path = 'Data/Mult_Noise_HiRes/'
# total particle concentration
concentration = 1000
# geometric standard deviation
sigma_g = 1.05
# Particle diameter, geometric mean of the particle diameter
d= 903
# wavelength
w_n = 663
# CRI
m = 1.59 + 0.0j
# size array
size_array = np.arange(700, 1110, 10)
# generate phase function data
#noise_sigma_array = [0.1]
noise_sigma_array = np.arange(0.1, 50.0, 5.0)
# testing retrieval with data generated from mie theory with added noise
cri_n_space = np.arange(1.5, 1.7, .001)
#print(len(cri_n_space))
cri_k_space = np.arange(0.0, 0.2, 0.001)
#print(len(cri_k_space))

Retrieval_v0(path=py_path, number_density=concentration, geom_stdev=sigma_g, diameter=d, sizes=size_array, wavelength=w_n, m_known=m,  added_noise_array=noise_sigma_array, n_space=cri_n_space, k_space=cri_k_space)





