'''
Austen K. Scruggs
10-23-2019
Description: Looped plotting rayleigh scattering data at different polarizations
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

H_path = '/home/austen/Desktop/Rayleigh_Analysis/T3/lamda_0.5/PFs'
V_path = '/home/austen/Desktop/Rayleigh_Analysis/T3/lamda_0/PFs'

f0, ax0 = plt.subplots(1, 3, figsize=(20, 12))

H_files = os.listdir(H_path)
for file in H_files:
    df = pd.read_csv(H_path + '/' + str(file), sep=',', header=0)
    pf = df['CO2 Intensity gfit corr'] #- df['He Intensity gfit corr']
    theta = df['CO2 Theta']
    ax0[0].plot(theta, pf, label=str(file))
    ax0[2].plot(theta, pf, label=str(file))

V_files = os.listdir(V_path)
for file in V_files:
    df = pd.read_csv(V_path + '/' + str(file), sep=',', header=0)
    pf = df['CO2 Intensity gfit corr'] #- df['He Intensity gfit corr']
    theta = df['CO2 Theta']
    ax0[1].plot(theta, pf, label=str(file))
    ax0[2].plot(theta, pf, label=str(file))

ax0[0].set_title('\u03bb = 0.5')
ax0[0].set_ylabel('Intensity')
ax0[0].set_xlabel('\u03b8')
ax0[0].grid(True)
ax0[0].legend(loc=1)
ax0[1].set_title('\u03bb = 0.0')
ax0[1].set_ylabel('Intensity')
ax0[1].set_xlabel('\u03b8')
ax0[1].grid(True)
ax0[1].legend(loc=1)
ax0[2].set_title('Rayleigh Scattering at Various Retardances')
ax0[2].set_ylabel('Intensity')
ax0[2].set_xlabel('\u03b8')
ax0[2].grid(True)
ax0[2].legend(loc=1)
plt.tight_layout()
plt.show()