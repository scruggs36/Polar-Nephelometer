'''
Austen K. Scruggs
01-29-2019
Description: Plotting the dependence of the Mie calculated scattering diagram
 on the imaginary refractive index
'''

import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from itertools import cycle


data_path = '/home/austen/Documents/01-23-2019_Analysis/K_Sensitivity/k_900'
save_path = '/home/austen/Documents/'
file_list = os.listdir(data_path)


# create figure and subplot
f0, ax0 = plt.subplots(1, 3, figsize=(15, 10))
# line style colors to cycle through
cycol = cycle('bgrcmk')
for counter, file in enumerate(file_list):
    # associate file names with a value for the RI k as a float
    kval = file.split('_')[3]
    kval2 = kval.split('.')[1]
    kval3 = float('.' + kval2)
    # import data
    data = pd.read_csv(data_path + '/' + file, sep=',', header=0)
    ax0[0].semilogy(data['Theta Matheson 1957'], data['SL Matheson 1957'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
    ax0[1].semilogy(data['Theta Matheson 1957'], data['SR Matheson 1957'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
    ax0[2].semilogy(data['Theta Matheson 1957'], data['SU Matheson 1957'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
# subplot titles and labels
ax0[0].set_title('SL')
ax0[0].set_xlabel('\u03b8 (\u00b0)')
ax0[0].set_ylabel('|$S^2$|')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].set_title('SR')
ax0[1].set_xlabel('\u03b8 (\u00b0)')
ax0[1].set_ylabel('|$S^2$|')
ax0[1].grid(True)
ax0[1].legend(loc=1)
ax0[2].set_title('SU')
ax0[2].set_xlabel('\u03b8 (\u00b0)')
ax0[2].set_ylabel('|$S^2$|')
ax0[2].grid(True)
ax0[2].legend(loc=1)
f0.suptitle('Calculated 900nm PSL Scattering Diagram \n Using Matheson et. al. Cauchy Coefficients', fontsize=16)
#plt.tight_layout()
plt.savefig(save_path + 'k_sensitivity_Matheson.pdf', format='pdf')
plt.show()


# create figure and subplot
f1, ax1 = plt.subplots(1, 3, figsize=(15, 10))
# line style colors to cycle through
cycol = cycle('bgrcmk')
for counter, file in enumerate(file_list):
    # associate file names with a value for the RI k as a float
    kval = file.split('_')[3]
    kval2 = kval.split('.')[1]
    kval3 = float('.' + kval2)
    # import data
    data = pd.read_csv(data_path + '/' + file, sep=',', header=0)
    ax1[0].semilogy(data['Theta Ma 2003'], data['SL Ma 2003'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
    ax1[1].semilogy(data['Theta Ma 2003'], data['SR Ma 2003'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
    ax1[2].semilogy(data['Theta Ma 2003'], data['SU Ma 2003'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
# subplot titles and labels
ax1[0].set_title('SL')
ax1[0].set_xlabel('\u03b8 (\u00b0)')
ax1[0].set_ylabel('|$S^2$|')
ax1[0].grid(True)
ax1[0].legend(loc=1)
ax1[1].set_title('SR')
ax1[1].set_xlabel('\u03b8 (\u00b0)')
ax1[1].set_ylabel('|$S^2$|')
ax1[1].grid(True)
ax1[1].legend(loc=1)
ax1[2].set_title('SU')
ax1[2].set_xlabel('\u03b8 (\u00b0)')
ax1[2].set_ylabel('|$S^2$|')
ax1[2].grid(True)
ax1[2].legend(loc=1)
f1.suptitle('Calculated 900nm PSL Scattering Diagram \n Using Ma et. al. Cauchy Coefficients', fontsize=16)
#plt.tight_layout()
plt.savefig(save_path + 'k_sensitivity_Ma.pdf', format='pdf')
plt.show()


# create figure and subplot
f2, ax2 = plt.subplots(1, 3, figsize=(15, 10))
# line style colors to cycle through
cycol = cycle('bgrcmk')
for counter, file in enumerate(file_list):
    # associate file names with a value for the RI k as a float
    kval = file.split('_')[3]
    kval2 = kval.split('.')[1]
    kval3 = float('.' + kval2)
    # import data
    data = pd.read_csv(data_path + '/' + file, sep=',', header=0)
    ax2[0].semilogy(data['Theta Greenslade 2017'], data['SL Greenslade 2017'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
    ax2[1].semilogy(data['Theta Greenslade 2017'], data['SR Greenslade 2017'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
    ax2[2].semilogy(data['Theta Greenslade 2017'], data['SU Greenslade 2017'], ls='-', color=plt.cm.Set1(counter), label='k = ' + str(kval3))
# subplot titles and labels
ax2[0].set_title('SL')
ax2[0].set_xlabel('\u03b8 (\u00b0)')
ax2[0].set_ylabel('|$S^2$|')
ax2[0].grid(True)
ax2[0].legend(loc=1)
ax2[1].set_title('SR')
ax2[1].set_xlabel('\u03b8 (\u00b0)')
ax2[1].set_ylabel('|$S^2$|')
ax2[1].grid(True)
ax2[1].legend(loc=1)
ax2[2].set_title('SU')
ax2[2].set_xlabel('\u03b8 (\u00b0)')
ax2[2].set_ylabel('|$S^2$|')
ax2[2].grid(True)
ax2[2].legend(loc=1)
f2.suptitle('Calculated 900nm PSL Scattering Diagram \n Using Greenslade et. al. Cauchy Coefficients', fontsize=16)
#plt.tight_layout()
plt.savefig(save_path + 'k_sensitivity_Greenslade.pdf', format='pdf')
plt.show()
