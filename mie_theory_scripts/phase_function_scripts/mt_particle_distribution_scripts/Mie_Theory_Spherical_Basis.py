'''
Austen K. Scruggs
04/28/2020
Description: Performing Mie theory calculations at a single size distribution and wavelength
but changing CRI
'''

import numpy as np
import pandas as pd
import PyMieScatt as ps
import matplotlib.pyplot as plt
from scipy.integrate import simps
from math import sqrt, log, pi
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.interpolate import pchip_interpolate, interp1d
from sklearn.linear_model import LinearRegression

save_dir = '/home/austen/Desktop/'

# log normal distribution function, we might want it normalized, check if the equation is right...
def LogNormal(size, mu, gsd, N):
    #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
    return (N / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))

# size distribution parameters
geo_mean = 903.0
geo_sigma = 1.005
sigma = 4.1
N = 1000.0
# complex refractive index parameters
#n_array = [1.0] # used as a test value, all values must be in an array for the code to work
n_array = np.linspace(1.00, 2.00, 100E1, endpoint=True)
k_array = np.linspace(0.00, 1.00, 100E1, endpoint=True)
n_N2 = 1.00029739
# wavelength
w_n = 663.0


# pre-allocate all lists to populate
data = []
n_list = []
k_list = []
dbins_list = []
nbins_list = []
Bext_list = []
Bsca_list = []
Babs_list = []
bigG_list = []
Bpr_list = []
Bback_list = []
Bratio_list = []
Theta_list = []
SL_list = []
SR_list = []
SU_list = []
DLP_list = []
for counter, n in enumerate(n_array):
    completed = (counter / len(n_array)) * 100
    print('percent completed: ', completed, '%')
    for k in k_array:
        m = complex(n, k)
        # calculate Bext, Bsca, Babs, G, Bpr, Bback, Bratio
        returned = ps.Mie_Lognormal(m=m, wavelength=w_n, geoStdDev=geo_sigma, geoMean=geo_mean, numberOfParticles=N, nMedium=n_N2, numberOfBins=41, lower=geo_mean-(5.0*sigma), upper=geo_mean+(5.0*sigma), returnDistribution=True, asDict=True)
        optics_params = returned[0]
        dbins = returned[1]
        nbins = returned[2]
        #print(optics_params)
        # calculate phase function for the given distribution
        Theta, SL, SR, SU = ps.SF_SD(m=m, wavelength=w_n, dp=dbins, ndp=nbins, nMedium=n_N2, minAngle=0.0, maxAngle=180.0, angularResolution=0.5, space='theta', angleMeasure='degrees', normalization='t')
        #print(SL)
        DLP = (-1.0 * (SL - SR)) / (SL + SR)
        n_list.append(n)
        k_list.append(k)
        dbins_list.append(dbins)
        nbins_list.append(nbins)
        Bext_list.append(optics_params['Bext'])
        Bsca_list.append(optics_params['Bsca'])
        Babs_list.append(optics_params['Babs'])
        bigG_list.append(optics_params['bigG'])
        Bpr_list.append(optics_params['Bpr'])
        Bback_list.append(optics_params['Bback'])
        Bratio_list.append(optics_params['Bratio'])
        Theta_list.append(Theta)
        SL_list.append(SL)
        SR_list.append(SR)
        SU_list.append(SU)
        DLP_list.append(DLP)


# saving data of mixed arrays and single valued floats with numpy basically has to all be formatted into a string
data = np.transpose(np.array([n_list, k_list, dbins_list, nbins_list, Bext_list, Bsca_list, Babs_list, bigG_list, Bpr_list, Bback_list, Bratio_list, Theta_list, SL_list, SR_list, SU_list, DLP_list]))
print('data dimensions: ', data.shape)
headers = "n,k,Dbins,Nbins,Bext,Bsca,Babs,bigG,Bpr,Bback,Bratio,Theta,SL,SR,SU,DLP"
#fmt_array = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" only works if each element in the ndarray is the same format
np.savetxt(save_dir + 'numpy_data.txt', data, delimiter=',', fmt='%s', header=headers)

# pandas is better!
head = ["n", "k", "Dbins", "Nbins", "Bext", "Bsca", "Babs", "bigG", "Bpr", "Bback", "Bratio", "Theta", "SL", "SR", "SU", "DLP"]
DF = pd.DataFrame(data, columns=head)
DF.to_csv(save_dir + 'pandas_data.txt', sep=',')
#print(DF)

