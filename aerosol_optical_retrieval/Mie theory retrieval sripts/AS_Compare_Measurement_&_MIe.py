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
file_directory = '/home/austen/Desktop/Recent/'
save_directory = '/home/austen/Desktop/Recent/'


def LogNormal(size, mu, gsd):
    if size ==0.0:
        return 0.0
    else:
        # the one directly below doesnt integrate up to the number of particles, it integrates to waaay more than the number of particles
        #return (N / (sqrt(2 * pi) * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))
        return (1 / (sqrt(2 * pi) * size * log(gsd))) * np.exp(-1 * ((log(size) - log(mu)) ** 2) / (2 * log(gsd) ** 2))


comp = pd.read_csv(save_directory + 'AS_DF.txt', sep=',', header=0, index_col=0)

comp.set_index(['Size (nm)', 'Polarization (deg)'], inplace=True)

# here we were troubleshooting reading in the data, because of this we are adjusting data format in pandas for compiling data!
#print(comp)
#y0 = comp.loc[300, 0]['Riemann'].split()
y0 = comp.loc[300, 0]['Riemann']
y1 = y0.strip('[]')
y2 = y1.split()
print(np.array(y2).astype('float'))

# we are going to brute force reading in MT
MT_300_file = '/home/austen/Desktop/Recent/AS_MT_Quick/300nm/AS_MT_300.txt'
MT_400_file = '/home/austen/Desktop/Recent/AS_MT_Quick/400nm/AS_MT_400.txt'
MT_500_file = '/home/austen/Desktop/Recent/AS_MT_Quick/500nm/AS_MT_500.txt'
MT_600_file = '/home/austen/Desktop/Recent/AS_MT_Quick/600nm/AS_MT_600.txt'
MT_700_file = '/home/austen/Desktop/Recent/AS_MT_Quick/700nm/AS_MT_700.txt'
MT_800_file = '/home/austen/Desktop/Recent/AS_MT_Quick/800nm/AS_MT_800.txt'
MT_900_file = '/home/austen/Desktop/Recent/AS_MT_Quick/900nm/AS_MT_900.txt'

MT_300_DF = pd.read_csv(MT_300_file, sep=',', header=0)
MT_400_DF = pd.read_csv(MT_400_file, sep=',', header=0)
MT_500_DF = pd.read_csv(MT_500_file, sep=',', header=0)
MT_600_DF = pd.read_csv(MT_600_file, sep=',', header=0)
MT_700_DF = pd.read_csv(MT_700_file, sep=',', header=0)
MT_800_DF = pd.read_csv(MT_800_file, sep=',', header=0)
MT_900_DF = pd.read_csv(MT_900_file, sep=',', header=0)

f0 = plt.figure(constrained_layout=True, figsize=(12, 18))
spec = GridSpec(ncols=3, nrows=4, figure=f0)
# size 300nm data
f0_ax00 = f0.add_subplot(spec[0, 0])
f0_ax00.semilogy(np.array(comp.loc[300, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='300nm \u2225')
f0_ax00.semilogy(MT_300_DF['Theta'], np.array(MT_300_DF['SL'])/np.sum(np.array(MT_300_DF['SL'])), color='black', label='MT 300nm \u2225')
f0_ax00.set_xlabel('Theta')
f0_ax00.set_xlabel('Normalized Intensity')
f0_ax00.grid(True)
f0_ax00.legend(loc=1)
f0_ax01 = f0.add_subplot(spec[0, 1], sharex=f0_ax00)
f0_ax01.semilogy(np.array(comp.loc[300, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='300nm \u27f3')
f0_ax01.semilogy(MT_300_DF['Theta'], np.array(MT_300_DF['SU'])/np.sum(np.array(MT_300_DF['SU'])), color='black', label='MT 300nm \u27f3')
f0_ax01.set_xlabel('Theta')
f0_ax01.set_xlabel('Normalized Intensity')
f0_ax01.grid(True)
f0_ax01.legend(loc=1)
f0_ax02 = f0.add_subplot(spec[0, 2])
f0_ax02.semilogy(np.array(comp.loc[300, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[300, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[300, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='300nm \u22A5')
f0_ax02.semilogy(MT_300_DF['Theta'], np.array(MT_300_DF['SR'])/np.sum(np.array(MT_300_DF['SR'])), color='black', label='MT 300nm \u22A5')
f0_ax02.set_xlabel('Theta')
f0_ax02.set_xlabel('Normalized Intensity')
f0_ax02.grid(True)
f0_ax02.legend(loc=1)
# size 400nm data
f0_ax10 = f0.add_subplot(spec[1, 0])
f0_ax10.semilogy(np.array(comp.loc[400, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='400nm \u2225')
f0_ax10.semilogy(MT_400_DF['Theta'], np.array(MT_400_DF['SL'])/np.sum(np.array(MT_400_DF['SL'])), color='black', label='MT 400nm \u2225')
f0_ax10.set_xlabel('Theta')
f0_ax10.set_xlabel('Normalized Intensity')
f0_ax10.grid(True)
f0_ax10.legend(loc=1)
f0_ax11 = f0.add_subplot(spec[1, 1])
f0_ax11.semilogy(np.array(comp.loc[400, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='400nm \u27f3')
f0_ax11.semilogy(MT_400_DF['Theta'], np.array(MT_400_DF['SU'])/np.sum(np.array(MT_400_DF['SU'])), color='black', label='MT 400nm \u27f3')
f0_ax11.set_xlabel('Theta')
f0_ax11.set_xlabel('Normalized Intensity')
f0_ax11.grid(True)
f0_ax11.legend(loc=1)
f0_ax12 = f0.add_subplot(spec[1, 2])
f0_ax12.semilogy(np.array(comp.loc[400, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[400, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[400, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='400nm \u22A5')
f0_ax12.semilogy(MT_400_DF['Theta'], np.array(MT_400_DF['SR'])/np.sum(np.array(MT_400_DF['SR'])), color='black', label='MT 400nm \u22A5')
f0_ax12.set_xlabel('Theta')
f0_ax12.set_xlabel('Normalized Intensity')
f0_ax12.grid(True)
f0_ax12.legend(loc=1)

# size 500nm data
f0_ax20 = f0.add_subplot(spec[2, 0])
f0_ax20.semilogy(np.array(comp.loc[500, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='500nm \u2225')
f0_ax20.semilogy(MT_500_DF['Theta'], np.array(MT_500_DF['SL'])/np.sum(np.array(MT_500_DF['SL'])), color='black', label='MT 500nm \u2225')
f0_ax20.set_xlabel('Theta')
f0_ax20.set_xlabel('Normalized Intensity')
f0_ax20.grid(True)
f0_ax20.legend(loc=1)
f0_ax21 = f0.add_subplot(spec[2, 1])
f0_ax21.semilogy(np.array(comp.loc[500, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='500nm \u27f3')
f0_ax21.semilogy(MT_500_DF['Theta'], np.array(MT_500_DF['SU'])/np.sum(np.array(MT_500_DF['SU'])), color='black', label='MT 500nm \u27f3')
f0_ax21.set_xlabel('Theta')
f0_ax21.set_xlabel('Normalized Intensity')
f0_ax21.grid(True)
f0_ax21.legend(loc=1)
f0_ax22 = f0.add_subplot(spec[2, 2])
f0_ax22.semilogy(np.array(comp.loc[500, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[500, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[500, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='500nm \u22A5')
f0_ax22.semilogy(MT_500_DF['Theta'], np.array(MT_500_DF['SR'])/np.sum(np.array(MT_500_DF['SR'])), color='black', label='MT 500nm \u22A5')
f0_ax22.set_xlabel('Theta')
f0_ax22.set_xlabel('Normalized Intensity')
f0_ax22.grid(True)
f0_ax22.legend(loc=1)

# size 600nm data
f0_ax30 = f0.add_subplot(spec[3, 0])
f0_ax30.semilogy(np.array(comp.loc[600, 0]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 0]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 0]['Riemann'].strip('[]').split()).astype('float')), color='red', label='600nm \u2225')
f0_ax30.semilogy(MT_600_DF['Theta'], np.array(MT_600_DF['SL'])/np.sum(np.array(MT_600_DF['SL'])), color='black', label='MT 600nm \u2225')
f0_ax30.set_xlabel('Theta')
f0_ax30.set_xlabel('Normalized Intensity')
f0_ax30.grid(True)
f0_ax30.legend(loc=1)
f0_ax31 = f0.add_subplot(spec[3, 1])
f0_ax31.semilogy(np.array(comp.loc[600, 90]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 90]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 90]['Riemann'].strip('[]').split()).astype('float')), color='green', label='600nm \u27f3')
f0_ax31.semilogy(MT_600_DF['Theta'], np.array(MT_600_DF['SU'])/np.sum(np.array(MT_600_DF['SU'])), color='black', label='MT 600nm \u27f3')
f0_ax31.set_xlabel('Theta')
f0_ax31.set_xlabel('Normalized Intensity')
f0_ax31.grid(True)
f0_ax31.legend(loc=1)
f0_ax32 = f0.add_subplot(spec[3, 2])
f0_ax32.semilogy(np.array(comp.loc[600, 180]['Angles (deg)'].strip('[]').split()).astype('float'), np.array(comp.loc[600, 180]['Riemann'].strip('[]').split()).astype('float') / np.sum(np.array(comp.loc[600, 180]['Riemann'].strip('[]').split()).astype('float')), color='blue', label='600nm \u22A5')
f0_ax32.semilogy(MT_600_DF['Theta'], np.array(MT_600_DF['SR'])/np.sum(np.array(MT_600_DF['SR'])), color='black', label='MT 600nm \u22A5')
f0_ax32.set_xlabel('Theta')
f0_ax32.set_xlabel('Normalized Intensity')
f0_ax32.grid(True)
f0_ax32.legend(loc=1)
#plt.suptitle('Data Collected Using Impactor', y=1.25)
plt.savefig(save_directory + 'Impactor_Data.png', format='png')
plt.savefig(save_directory + 'Impactor_Data.pdf', format='pdf')
plt.show()

plt.semilogy()