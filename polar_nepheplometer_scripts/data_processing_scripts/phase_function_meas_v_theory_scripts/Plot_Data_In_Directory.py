'''
Author: Austen K. Scruggs
Date: 06-20-2018
Description: This is kind of a general tool to plot text files in directories
and use their file names as labels, I use it to plot scattering diagams with different
exposure times or even scattering diagrams of different particles and sizes, this script
acts on text files generated from Labview_Data_to_Txt.py, which averages data from summary
files from all images collected and produces an averaged summary text file, thus, this
script is merely a plotting tool

'''

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import pchip_interpolate
from math import pi

Path_Particle = '/home/austen/Documents/02-13-2019_Analysis/900nm_PSL/SDs'
Path_Save = '/home/austen/Documents/'
Path_Mie = '/home/austen/Documents/01-23-2019_Analysis/900/Theory/PSL900nm_MieTheory.txt'

# import Mie Data
Mie_Data = pd.read_csv(Path_Mie, sep=',', header=0)
Mie_Theta_Array = [Mie_Data['Theta Matheson 1952'], Mie_Data['Theta Bateman 1959'], Mie_Data['Theta Nikalov 2000'], Mie_Data['Theta Ma 2003'], Mie_Data['Theta Sultanova 2003'], Mie_Data['Theta Kasarova 2006'], Mie_Data['Theta Miles 2010'], Mie_Data['Theta Jones 2013'],Mie_Data['Theta Greenslade 2017'], Mie_Data['Theta Gienger 2017']]
Mie_Array = [Mie_Data['SU Matheson 1952'], Mie_Data['SU Bateman 1959'], Mie_Data['SU Nikalov 2000'], Mie_Data['SU Ma 2003'], Mie_Data['SU Sultanova 2003'], Mie_Data['SU Kasarova 2006'], Mie_Data['SU Miles 2010'], Mie_Data['SU Jones 2013'],Mie_Data['SU Greenslade 2017'], Mie_Data['SU Gienger 2017']]
Groups = ['Matheson 1952', 'Bateman 1959', 'Nikalov 2000', 'Ma 2003', 'Sultanova 2003', 'Kasarova 2006', 'Miles 2010', 'Jones 2013', 'Greenslade 2017', 'Gienger 2017']
#  list all file names within a directory
file_list1 = os.listdir(Path_Particle)
# number of files in directory
num_files = len(file_list1)


file_exposures = []
for file in file_list1:
    print(file)
    name, exposure, ft = file.split('_')
    file_exposures.append(exposure)

slope = 0.2045
intercept = -41.9764
scalar_array = []
# create subplot
f0, ax0 = plt.subplots(1, 2, figsize=(20, 10))
for counter, element in enumerate(file_list1):
    Data = pd.read_csv(Path_Particle + '/' + str(element), sep=',')
    SD = np.array(Data['Sample Intensity']) - np.asarray(Data['Nitrogen Intensity'])
    Z = SD
    X = np.array(Data['Columns'])
    theta = np.array([(element * slope) + intercept for element in X])
    f3, ax3 = plt.subplots(1, 2, figsize=(20, 7))
    for counter1, group in enumerate(Mie_Array):
        scalar = np.amin(np.array(group)) / np.amin(Z)
        scalar_array.append(scalar)
        Mie_Theta = np.array(Mie_Theta_Array[counter1])
        Mie_Pchip = pchip_interpolate(Mie_Theta, np.array(group), theta, der=0, axis=0)
        ratio_array = np.divide(Z, Mie_Pchip)
        ax3[0].plot(theta, ratio_array, ls='-', label=str(file_exposures[counter]) + ' Exp. / ' + Groups[counter1])
        ax3[1].plot(theta, np.divide(Z, ratio_array), ls='-', label=str(file_exposures[counter]) + ' Exp. / ' + Groups[counter1])
    ax3[1].semilogy(np.array(Mie_Data['Theta Matheson 1952']), np.array(Mie_Data['SU Matheson 1952']), ls='--', label='Matheson et. al. 1952')
    ax3[1].semilogy(np.array(Mie_Data['Theta Bateman 1959']), np.array(Mie_Data['SU Bateman 1959']), ls='--', label='Bateman et. al. 1959')
    ax3[1].semilogy(np.array(Mie_Data['Theta Nikalov 2000']), np.array(Mie_Data['SU Nikalov 2000']), ls='--', label='Nikalov et. al. 2000')
    ax3[1].semilogy(np.array(Mie_Data['Theta Ma 2003']), np.array(Mie_Data['SU Ma 2003']), ls='--', label='Ma et. al. 2003')
    ax3[1].semilogy(np.array(Mie_Data['Theta Sultanova 2003']), np.array(Mie_Data['SU Sultanova 2003']), ls='--', label='Sultanova et. al. 2003')
    ax3[1].semilogy(np.array(Mie_Data['Theta Kasarova 2006']), np.array(Mie_Data['SU Kasarova 2006']), ls='--', label='Kasarova et. al. 2006')
    ax3[1].semilogy(np.array(Mie_Data['Theta Miles 2010']), np.array(Mie_Data['SU Miles 2010']), ls='--', label='Miles et. al. 2010')
    ax3[1].semilogy(np.array(Mie_Data['Theta Jones 2013']), np.array(Mie_Data['SU Jones 2013']), ls='--', label='Jones et. al. 2013')
    ax3[1].semilogy(np.array(Mie_Data['Theta Greenslade 2017']), np.array(Mie_Data['SU Greenslade 2017']), ls='--', label='Greensalde et. al. 2017')
    ax3[1].semilogy(np.array(Mie_Data['Theta Gienger 2017']), np.array(Mie_Data['SU Gienger 2017']), ls='--', label='Grienger et. al. 2017')
    ax3[0].set_title('900nm PSL Ratio of Phase Functions (Experiment/Mie) \n at Various Exposure Times')
    ax3[0].set_xlabel('\u0398')
    ax3[0].set_ylabel('Ratio')
    ax3[0].grid(True)
    ax3[0].legend(bbox_to_anchor=(1, 0.5), loc='center left', fancybox=True)
    ax3[1].set_title('900nm PSL Ratio Corrected of Phase Functions \n at Various Exposure Times')
    ax3[1].set_xlabel('\u0398')
    ax3[1].set_ylabel('Ratio')
    ax3[1].grid(True)
    ax3[1].legend(bbox_to_anchor=(1, 0.5), loc='center left', fancybox=True)
    plt.tight_layout()
    plt.savefig(Path_Save + 'Ratio_' + file_list1[counter] + '.pdf', format='pdf')
    plt.savefig(Path_Save + 'Ratio_' + file_list1[counter] + '.png', format='png')
    plt.close(f3)
    scalar_avg = np.average(scalar_array)
    scalar_array = []
    ax0[0].semilogy(X, Z, label=str(file_exposures[counter]))
    ax0[1].semilogy(theta, Z * scalar_avg, label=str(file_exposures[counter]) + ' Avg. Int. Scalar: ' + str('{:.3e}'.format(scalar_avg)))


ax0[1].semilogy(np.array(Mie_Data['Theta Matheson 1952']), np.array(Mie_Data['SU Matheson 1952']), ls='--', label='Matheson et. al. 1952')
ax0[1].semilogy(np.array(Mie_Data['Theta Bateman 1959']), np.array(Mie_Data['SU Bateman 1959']), ls='--', label='Bateman et. al. 1959')
ax0[1].semilogy(np.array(Mie_Data['Theta Nikalov 2000']), np.array(Mie_Data['SU Nikalov 2000']), ls='--', label='Nikalov et. al. 2000')
ax0[1].semilogy(np.array(Mie_Data['Theta Ma 2003']), np.array(Mie_Data['SU Ma 2003']), ls='--', label='Ma et. al. 2003')
ax0[1].semilogy(np.array(Mie_Data['Theta Sultanova 2003']), np.array(Mie_Data['SU Sultanova 2003']), ls='--', label='Sultanova et. al. 2003')
ax0[1].semilogy(np.array(Mie_Data['Theta Kasarova 2006']), np.array(Mie_Data['SU Kasarova 2006']), ls='--', label='Kasarova et. al. 2006')
ax0[1].semilogy(np.array(Mie_Data['Theta Miles 2010']), np.array(Mie_Data['SU Miles 2010']), ls='--', label='Miles et. al. 2010')
ax0[1].semilogy(np.array(Mie_Data['Theta Jones 2013']), np.array(Mie_Data['SU Jones 2013']), ls='--', label='Jones et. al. 2013')
ax0[1].semilogy(np.array(Mie_Data['Theta Greenslade 2017']), np.array(Mie_Data['SU Greenslade 2017']), ls='--', label='Greensalde et. al. 2017')
ax0[1].semilogy(np.array(Mie_Data['Theta Gienger 2017']), np.array(Mie_Data['SU Gienger 2017']), ls='--', label='Grienger et. al. 2017')
ax0[0].set_title('900nm PSL Phase Functions at Various Exposure Times')
ax0[0].set_xlabel('Profile Number')
ax0[0].set_ylabel('Intensity (DN)')
ax0[0].grid(True)
ax0[0].legend(bbox_to_anchor=(1, 0.5), loc='center left', fancybox=True)
ax0[1].set_title('900nm PSL Phase Functions at Various Exposure Times')
ax0[1].set_xlabel('\u0398')
ax0[1].set_ylabel('Intensity (DN)')
ax0[1].grid(True)
ax0[1].legend(bbox_to_anchor=(1, 0.5), loc='center left', fancybox=True)
plt.savefig(Path_Save + '900nmPSL_Exposures.pdf', format='pdf')
plt.savefig(Path_Save + '900nmPSL_Exposures.png', format='png')
plt.show()


# create 1 + cos^2(theta) function
def func(x, a, b):
    return b + (a * (np.cos(x)**2))


# create subplot
f1, ax1 = plt.subplots(1, 2, figsize=(20, 7))
for counter, element in enumerate(file_list1):
    Data = pd.read_csv(Path_Particle + '/' + str(element), sep=',')
    SD = np.asarray(Data['Nitrogen Intensity'])
    Z = SD
    X = np.array(Data['Columns'])
    theta = np.array([(element * slope) + intercept for element in X])
    radians = theta * (pi/180.0)
    popt, pcov = curve_fit(func, radians[100:-100], Z[100:-100])
    ax1[0].plot(X, Z, label=str(file_exposures[counter]))
    ax1[1].plot(theta, Z, ls='-', label=str(file_exposures[counter]))
    ax1[1].plot(theta[100:-100], func(radians, *popt)[100:-100], ls='--', label=str(file_exposures[counter]))


ax1[0].set_title('N2 Phase Functions as a Function of Exposure Time')
ax1[0].set_xlabel('Profile Number')
ax1[0].set_ylabel('Intensity (DN)')
ax1[0].grid(True)
ax1[0].legend(bbox_to_anchor=(1, 0.5), loc='center left')
ax1[1].set_title('N2 Phase Functions and $1 + Cos^2(\u0398)$ Fits \n as a Function of Exposure Time')
ax1[1].set_xlabel('\u0398')
ax1[1].set_ylabel('Intensity (DN)')
ax1[1].set_ylim((300, 600))
ax1[1].grid(True)
ax1[1].legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.tight_layout()
plt.savefig(Path_Save + 'N2_Exposures.pdf', format='pdf')
plt.savefig(Path_Save + 'N2_Exposures.png', format='png')
plt.show()

