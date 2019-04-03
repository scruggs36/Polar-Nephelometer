import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

#compare phase functions
#PF_Path_1 = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/09-26-2017/Air/PF_Air_Minus_He.txt'
PF_Path_2 = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/12-06-2017/N2/T2/PF_N2_SUB_2.txt'
#PF_Path_2 = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/10-03-2017/10mW_30sExposure/Nigrosin/PF_Nigrosin_Stagnant.txt'
#PF_Path_3 = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/09-21-2017/Particle/phase_function_Particle.txt'
#PF_1 = pd.read_csv(PF_Path_1, delimiter=',', header=0)
PF_2 = pd.read_csv(PF_Path_2, delimiter=',', header=0)
#PF_3 = pd.read_csv(PF_Path_3, delimiter=',', header=0)

def Rayleigh_Scattering_Fitting_EQ(theta, C):
    n = 1.000298
    wav = 663E-9
    return C * ((1 + (np.cos(theta)) ** 2) * (((2 * np.pi)/(wav)) ** 4) * ((n ** 2 - 1)/(n ** 2 + 2)) ** 2)


def Cosine_Squared(theta, a):
    return a * (0 + np.cos(theta) ** 2)


# for loops using list comprehensions
Intensity_Norm = [x/np.amax(PF_2['Top Riemann Sum']) for x in PF_2['Top Riemann Sum']]
Array_Length_1 = len(PF_2['Top Profile Number'])
Degrees_Array = np.linspace(0,90,Array_Length_1)
#Intensity_Norm = PF_2['Top Riemann Sum']
#Profiles_to_Radians = [x * (np.pi/180.0) for x in PF_2['Top Profile Number']]
Profiles_to_Radians = [x * (np.pi/180.0) for x in Degrees_Array]
#print(Intensity_Norm)
#print(Profiles_to_Radians)
popt, pcov = curve_fit(Cosine_Squared, Profiles_to_Radians, Intensity_Norm, maxfev=800)


f0, ax0 = plt.subplots()
#ax0.plot(PF_2['Top Profile Number'], PF_2['Top Riemann Sum'], 'r--', label='Room Air - He')
ax0.plot(Degrees_Array, Intensity_Norm, 'b-', label='Nitrogen Scattering Meas.')
ax0.plot(Degrees_Array, Cosine_Squared(Profiles_to_Radians, *popt), 'r-', label='Nitrogen Rayleigh Fit ' + 'a: ' + str(popt))
#ax0.plot(PF_Particle['Top Profile Number'], PF_Particle['Top Riemann Sum'], 'g--', label='Particle')
ax0.set_title('Nitrogen Normalized Integrated Scattering Intensity\n and $\mathregular{\cos^2(\Theta)}$ Rayleigh Fit as a Function of Scattering Angle', size=26)
#ax0.set_title('Nigrosin Normalized Integrated Scattering\n Intensity as a Function of Scattering Angle', size=26)
ax0.set_xlabel('Scattering Angle (Degrees)', size=26)
ax0.set_ylabel('Normalized Integrated Intensity', size=26)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc=1, prop={'size': 8})
plt.show()
#CGS units rayleigh scattering for a single particle

# Save Phase Function
PF_Path = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/Data/12-06-2017/N2/T2/PF_N2_Data.txt'
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Radians'] = Profiles_to_Radians
PhaseFunctionDF['Degrees'] = Degrees_Array
PhaseFunctionDF['Normalized Intensity'] = Intensity_Norm
#PhaseFunctionDF.to_csv(PF_Path)

#'''
# Plot Phase Function from csv
Data_Path = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/Data/12-05-2017/N2/300s Background Subtracted/PF_N2_Data - Copy.txt'
Data = pd.read_csv(Data_Path, delimiter=',', header=0)
Array_Length_2 = len(Data['Normalized Intensity'])
Degrees_Array_2 = np.linspace(10, 90,Array_Length_2)
Profiles_to_Radians_2 = [x * (np.pi/180.0) for x in Degrees_Array_2]
popt1, pcov1 = curve_fit(Cosine_Squared, Profiles_to_Radians_2, Data['Normalized Intensity'], maxfev=800)


f1, ax1 = plt.subplots()
#ax1.plot(PF_2['Top Profile Number'], PF_2['Top Riemann Sum'], 'r--', label='Room Air - He')
ax1.plot(Degrees_Array_2, Data['Normalized Intensity'], 'b*', label='Nitrogen Scattering Meas.')
ax1.plot(Degrees_Array_2, Cosine_Squared(Profiles_to_Radians_2, *popt1), 'r-', label='Nitrogen Rayleigh Fit ' + 'a: ' + str(popt))
#ax1.plot(PF_Particle['Top Profile Number'], PF_Particle['Top Riemann Sum'], 'g--', label='Particle')
ax1.set_title('Nitrogen Normalized Integrated Scattering Intensity\n and $\mathregular{\cos^2(\Theta)}$ Rayleigh Fit as a Function of Scattering Angle', size=26)
#ax1.set_title('Nigrosin Normalized Integrated Scattering\n Intensity as a Function of Scattering Angle', size=26)
ax1.set_xlabel('Scattering Angle (Degrees)', size=26)
ax1.set_ylabel('Normalized Integrated Intensity', size=26)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(loc=1, prop={'size': 8})
plt.show()
#'''

