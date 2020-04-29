'''
Austen K. Scruggs
09-13-2018
Description: This is a Mie Theory script that a computes scattering diagram
taking into account identical particles in a given gaussian size distribution
'''

import numpy as np
import PyMieScatt as PMS
from math import pi, sqrt, log10
import scipy as sp
import pandas as PD
import matplotlib.pyplot as plt

Save_Directory = '/home/austen/Documents/'
'''
First we will calculate the particle size distribution and plot it
'''
# geometric standard deviation
sigma_g = 1.05
# Particle diameter, geometric mean of the particle diameter
d= 150

# write function for LogNormal distribution (same as Tami Bonds)
def LogNormal(diam, mu, gsd):
    return np.exp(-1 * ((np.log10(diam/mu)) ** 2) / (2 * log10(gsd) ** 2)) / (sqrt(2 * pi) * log10(gsd))


size_axis_lognorm = np.arange(1, 1000, 10)
Log_Normal_Data = LogNormal(size_axis_lognorm, mu=d, gsd=sigma_g) / np.linalg.norm(LogNormal(size_axis_lognorm, mu=d, gsd=sigma_g))

#print(sp.integrate.simps(Gaussian(size_axis, mu=d, sigma=sigma_g), size_axis, dx=1))
f, ax = plt.subplots(figsize=(6, 6))
ax.plot(size_axis_lognorm, Log_Normal_Data, 'b-', label='LogNormal $\u03c3_{g}$=' + str(sigma_g))
ax.set_xlabel('particle diameter (nm)')
ax.set_ylabel('Normalized $dN/Log_{10}(D)$')
ax.set_title('Distributions Used for Mie Theory Calculations')
ax.grid(True)
plt.legend(loc=1)
plt.savefig(Save_Directory + 'Mie_LogNormal.pdf', format='pdf')
plt.show()
'''
We will used the normalized distribution intensities at each particle diameter as weights 
when we sum all our scattering diagrams, here we just collect the distribution information in array of arrays
'''

Weights_LogNormal = []
for counter, element in enumerate(range(len(Log_Normal_Data))):
    Weights_LogNormal.append([size_axis_lognorm[counter], Log_Normal_Data[counter]])

'''
Here we are conducting the Mie theory, and we are going to loop through the sizes
'''
# wavelength
wavelength = 663.0
# refractive index
m_medium = 1.000277
# Cauchy parameters for PSL
A = 1.5725
B = 0.0031080
C = 0.00034779
# n of refractive index for PSL as cauchy equation
n_cauchy = A + (B / wavelength ** 2) + (C / (wavelength ** 4))
m_particle = complex(n_cauchy, 0.0005)
# particle wavenumber calculation
k = (pi * m_medium) / wavelength
# size parameter calculation
X = k * d
# scattering angles
theta = np.arange(0, 181, 1)
rads = [x * (pi / 180.0) for x in theta]
mu = np.cos(rads)

theta_0, SL_0, SR_0, SU_0 = ps.ScatteringFunction(m0, w_n, element, nMedium=1.0, minAngle=0, maxAngle=180,
                                                  angularResolution=0.5, space='theta', angleMeasure='degrees',
                                                  normalization=None)

'''
# use PyMieScatt function to calculate scattering amplitude matrix elements
pd = []
S11 = []
S12 = []
S33 = []
S34 = []
SAM_Matrix_Array = []
SAM_Array_Of_Matrix_Arrays = []
S11_Weighted = []
S12_Weighted = []
S33_Weighted = []
S34_Weighted = []
incident_stokes_vector_perp = [1, -1, 0, 0]
resultant_stokes_vector_array = []
I_vec = []
Q_vec = []
U_vec = []
V_vec = []
I_vec_2darray = []
Q_vec_2darray = []
U_vec_2darray = []
V_vec_2darray = []
for element0 in Weights_LogNormal:
    for element1 in mu:
        Matrix_Elements = PMS.MatrixElements(m=m_particle, wavelength=wavelength, diameter=element0[0], mu=element1, nMedium=m_medium)
        # the 4 x 4 scattering amplitude matrix calculation I believe assumes S3 and S4 of the 2 x 2 scattering amplitude matrix are zero due to the particles being spherical (symmetry arguments)
        SAM = element0[1] * np.matrix([[Matrix_Elements[0], Matrix_Elements[1], 0, 0], [Matrix_Elements[1], Matrix_Elements[0], 0, 0], [0, 0, Matrix_Elements[2], Matrix_Elements[3]], [0, 0, Matrix_Elements[3], Matrix_Elements[2]]])
        resultant_stokes_vector = np.asarray(SAM.dot(np.transpose(incident_stokes_vector_perp))).flatten()
        SAM_Matrix_Array.append(SAM)
        resultant_stokes_vector_array.append(resultant_stokes_vector)
        I_vec.append(resultant_stokes_vector[0])
        Q_vec.append(resultant_stokes_vector[1])
        U_vec.append(resultant_stokes_vector[2])
        V_vec.append(resultant_stokes_vector[3])
        S11.append(element0[1] * Matrix_Elements[0])
        S12.append(element0[1] * Matrix_Elements[1])
        S33.append(element0[1] * Matrix_Elements[2])
        S34.append(element0[1] * Matrix_Elements[3])
    SAM_Array_Of_Matrix_Arrays.append(SAM_Matrix_Array)
    S11_Weighted.append(np.asarray(S11))
    S12_Weighted.append(np.asarray(S12))
    S33_Weighted.append(np.asarray(S33))
    S34_Weighted.append(np.asarray(S34))
    I_vec_2darray.append(I_vec)
    Q_vec_2darray.append(Q_vec)
    U_vec_2darray.append(U_vec)
    V_vec_2darray.append(V_vec)
    SAM_Matrix_Array = []
    S11 = []
    S12 = []
    S33 = []
    S34 = []
    I_vec = []
    Q_vec = []
    U_vec = []
    V_vec = []


#print(np.asarray(S11_Weighted).shape)
S11_Weighted_Sum = np.sum(np.asarray(S11_Weighted), axis=0)
S12_Weighted_Sum = np.sum(np.asarray(S12_Weighted), axis=0)
S33_Weighted_Sum = np.sum(np.asarray(S33_Weighted), axis=0)
S34_Weighted_Sum = np.sum(np.asarray(S34_Weighted), axis=0)
I_vec_Weighted_Sum = np.sum(np.asarray(I_vec_2darray), axis=0)
Q_vec_Weighted_Sum = np.sum(np.asarray(Q_vec_2darray), axis=0)
U_vec_Weighted_Sum = np.sum(np.asarray(U_vec_2darray), axis=0)
V_vec_Weighted_Sum = np.sum(np.asarray(V_vec_2darray), axis=0)



#Take stokes vectors into account, this is the case where the incident stokes vector is vertically polarized, 
# or perpendicularly polarized to the scattering plane

#incident_stokes_vector_perp = [1, -1, 0, 0]
# transmitted intensity is given by the expression below, equation 2.87 in bohren and huffman
# here we are calculating the output stokes vector at each of the scattering angles (0-180 degrees) after the sample



#Plot the vertical, or perpendicular to plane polarized incident light scattering case

f0, ax0 = plt.subplots(2, 2, figsize=(10, 5))
ax0[0, 0].plot(theta, I_vec_Weighted_Sum, 'r-', label = 'I vs \u03B8')
ax0[0, 0].set_title('Intensity as a Function of Scattering Angle \n '+ str(incident_stokes_vector_perp) + ' on the Aerosol Sample')
ax0[0, 0].set_ylabel('I')
ax0[0, 0].set_yscale('symlog')
ax0[0, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[0, 0].grid(True)
ax0[1, 0].plot(theta, Q_vec_Weighted_Sum, 'b-', label = 'Q vs \u03B8')
ax0[1, 0].set_title('Stokes Element Q as a Function of Scattering Angle \n '+ str(incident_stokes_vector_perp) + ' on the Aerosol Sample')
ax0[1, 0].set_ylabel('Q')
ax0[1, 0].set_yscale('symlog')
ax0[1, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[1, 0].grid(True)
ax0[0, 1].plot(theta, U_vec_Weighted_Sum, 'g-', label = 'U vs \u03B8')
ax0[0, 1].set_title('Stokes Element U as a Function of Scattering Angle \n '+ str(incident_stokes_vector_perp) + ' on the Aerosol Sample')
ax0[0, 1].set_ylabel('U')
ax0[0, 1].set_yscale('symlog')
ax0[0, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[0, 1].grid(True)
ax0[1, 1].plot(theta, V_vec_Weighted_Sum, 'y-', label = 'V vs \u03B8')
ax0[1, 1].set_title('Stokes Element V as a Function of Scattering Angle \n '+ str(incident_stokes_vector_perp) + ' on the Aerosol Sample')
ax0[1, 1].set_ylabel('V')
ax0[1, 1].set_yscale('symlog')
ax0[1, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Stokes_Params_V_Incident.pdf', format='pdf')
plt.show()

f3, ax3 = plt.subplots(2, 2, figsize=(10, 5))
ax3[0, 0].plot(theta, S11_Weighted_Sum, 'r-')
ax3[0, 0].set_title('$S_{11}$ Weighted by Distribution \n Vertically Polarized 663nm Light')
ax3[0, 0].set_xlabel('Scattering Angle (\u00b0)')
ax3[0, 0].set_ylabel('$S_{11}$')
ax3[0, 0].set_yscale('log')
ax3[0, 0].grid(True)
ax3[0, 1].plot(theta, S12_Weighted_Sum)
ax3[0, 1].set_title('$S_{12}$ Weighted by Distribution \n Vertically Polarized 663nm Light')
ax3[0, 1].set_xlabel('Scattering Angle (\u00b0)')
ax3[0, 1].set_ylabel('$S_{12}$')
ax3[0, 1].set_yscale('log')
ax3[0, 1].grid(True)
ax3[1, 0].plot(theta, S33_Weighted_Sum)
ax3[1, 0].set_title('$S_{33}$ Weighted by Distribution \n Vertically Polarized 663nm Light')
ax3[1, 0].set_xlabel('Scattering Angle (\u00b0)')
ax3[1, 0].set_ylabel('$S_{33}$')
ax3[1, 0].set_yscale('log')
ax3[1, 0].grid(True)
ax3[1, 1].plot(theta, S34_Weighted_Sum)
ax3[1, 1].set_title('$S_{34}$ Weighted by Distribution \n Vertically Polarized 663nm Light')
ax3[1, 1].set_xlabel('Scattering Angle (\u00b0)')
ax3[1, 1].set_ylabel('$S_{34}$')
ax3[1, 1].set_yscale('log')
ax3[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Matrix_Elements_Perp.pdf', format='pdf')
plt.show()

#Take stokes vectors into account, this is the case where the incident stokes vector is horizontally polarized,
#or polarized parallel (inside) the scattering plane

S11_2 = []
S12_2 = []
S33_2 = []
S34_2 = []
SAM_Matrix_Array_2 = []
SAM_Array_Of_Matrix_Arrays_2 = []
S11_Weighted_2 = []
S12_Weighted_2 = []
S33_Weighted_2 = []
S34_Weighted_2 = []
incident_stokes_vector_par = [1, 1, 0, 0]
resultant_stokes_vector_array2 = []
I_vec2 = []
Q_vec2 = []
U_vec2 = []
V_vec2 = []
I_vec2_2darray = []
Q_vec2_2darray = []
U_vec2_2darray = []
V_vec2_2darray = []
for element0 in Weights_LogNormal:
    for element1 in mu:
        Matrix_Elements2 = PMS.MatrixElements(m=m_particle, wavelength=wavelength, diameter=element0[0], mu=element1, nMedium=m_medium)
        SAM2 = element0[1] * np.matrix([[Matrix_Elements2[0], Matrix_Elements2[1], 0, 0], [Matrix_Elements2[1], Matrix_Elements2[0], 0, 0], [0, 0, Matrix_Elements2[2], Matrix_Elements2[3]], [0, 0, Matrix_Elements2[3], Matrix_Elements2[2]]])
        SAM_Matrix_Array_2.append(SAM2)
        resultant_stokes_vector2 = np.asarray(SAM2.dot(np.transpose(incident_stokes_vector_par))).flatten()
        resultant_stokes_vector_array2.append(resultant_stokes_vector2)
        I_vec2.append(resultant_stokes_vector2[0])
        Q_vec2.append(resultant_stokes_vector2[1])
        U_vec2.append(resultant_stokes_vector2[2])
        V_vec2.append(resultant_stokes_vector2[3])
        S11_2.append(element0[1] * Matrix_Elements2[0])
        S12_2.append(element0[1] * Matrix_Elements2[1])
        S33_2.append(element0[1] * Matrix_Elements2[2])
        S34_2.append(element0[1] * Matrix_Elements2[3])
    SAM_Array_Of_Matrix_Arrays_2.append(SAM_Matrix_Array_2)
    S11_Weighted_2.append(np.asarray(S11_2))
    S12_Weighted_2.append(np.asarray(S12_2))
    S33_Weighted_2.append(np.asarray(S33_2))
    S34_Weighted_2.append(np.asarray(S34_2))
    I_vec2_2darray.append(I_vec2)
    Q_vec2_2darray.append(Q_vec2)
    U_vec2_2darray.append(U_vec2)
    V_vec2_2darray.append(V_vec2)
    SAM_Matrix_Array_2 = []
    S11_2 = []
    S12_2 = []
    S33_2 = []
    S34_2 = []
    I_vec2 = []
    Q_vec2 = []
    U_vec2 = []
    V_vec2 = []


S11_Weighted_Sum_2 = np.sum(np.asarray(S11_Weighted), axis=0)
S12_Weighted_Sum_2 = np.sum(np.asarray(S12_Weighted), axis=0)
S33_Weighted_Sum_2 = np.sum(np.asarray(S33_Weighted), axis=0)
S34_Weighted_Sum_2 = np.sum(np.asarray(S34_Weighted), axis=0)
I_vec_Weighted_Sum_2 = np.sum(np.asarray(I_vec2_2darray), axis=0)
Q_vec_Weighted_Sum_2 = np.sum(np.asarray(Q_vec2_2darray), axis=0)
U_vec_Weighted_Sum_2 = np.sum(np.asarray(U_vec2_2darray), axis=0)
V_vec_Weighted_Sum_2 = np.sum(np.asarray(V_vec2_2darray), axis=0)


#Plot the horizontal, or in plane polarized incident light scattering case


f1, ax1 = plt.subplots(2, 2, figsize=(10, 5))
ax1[0, 0].plot(theta, I_vec_Weighted_Sum_2, 'r-', label = 'I vs \u03B8')
ax1[0, 0].set_title('Intensity as a Function of Scattering Angle \n ' + str(incident_stokes_vector_par) + ' on the Aerosol Sample')
ax1[0, 0].set_ylabel('I')
ax1[0, 0].set_yscale('symlog')
ax1[0, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[0, 0].grid(True)
ax1[1, 0].plot(theta, Q_vec_Weighted_Sum_2, 'b-', label = 'Q vs \u03B8')
ax1[1, 0].set_title('Stokes Element Q as a Function of Scattering Angle \n ' + str(incident_stokes_vector_par) + ' on the Aerosol Sample')
ax1[1, 0].set_ylabel('Q')
ax1[1, 0].set_yscale('symlog')
ax1[1, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[1, 0].grid(True)
ax1[0, 1].plot(theta, U_vec_Weighted_Sum_2, 'g-', label = 'U vs \u03B8')
ax1[0, 1].set_title('Stokes Element U as a Function of Scattering Angle \n ' + str(incident_stokes_vector_par) + ' on the Aerosol Sample')
ax1[0, 1].set_ylabel('U')
ax1[0, 1].set_yscale('symlog')
ax1[0, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[0, 1].grid(True)
ax1[1, 1].plot(theta, V_vec_Weighted_Sum_2, 'y-', label = 'V vs \u03B8')
ax1[1, 1].set_title('Stokes Element V as a Function of Scattering Angle \n ' + str(incident_stokes_vector_par) + ' on the Aerosol Sample')
ax1[1, 1].set_ylabel('V')
ax1[1, 1].set_yscale('symlog')
ax1[1, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Stokes_Params_H_Incident.pdf', format='pdf')
plt.show()

f4, ax4 = plt.subplots(2, 2, figsize=(10, 5))
ax4[0, 0].plot(theta, S11_Weighted_Sum_2, 'r-')
ax4[0, 0].set_title('$S_{11}$ Weighted by Distribution \n Horizontally Polarized 663nm Light')
ax4[0, 0].set_xlabel('Scattering Angle (\u00b0)')
ax4[0, 0].set_ylabel('$S_{11}$')
ax4[0, 0].set_yscale('log')
ax4[0, 0].grid(True)
ax4[0, 1].plot(theta, S12_Weighted_Sum_2)
ax4[0, 1].set_title('$S_{12}$ Weighted by Distribution \n Horizontally Polarized 663nm Light')
ax4[0, 1].set_xlabel('Scattering Angle (\u00b0)')
ax4[0, 1].set_ylabel('$S_{12}$')
ax4[0, 1].set_yscale('log')
ax4[0, 1].grid(True)
ax4[1, 0].plot(theta, S33_Weighted_Sum_2)
ax4[1, 0].set_title('$S_{33}$ Weighted by Distribution \n Horizontally Polarized 663nm Light')
ax4[1, 0].set_xlabel('Scattering Angle (\u00b0)')
ax4[1, 0].set_ylabel('$S_{33}$')
ax4[1, 0].set_yscale('log')
ax4[1, 0].grid(True)
ax4[1, 1].plot(theta, S34_Weighted_Sum_2)
ax4[1, 1].set_title('$S_{34}$ Weighted by Distribution \n Horizontally Polarized 663nm Light')
ax4[1, 1].set_xlabel('Scattering Angle (\u00b0)')
ax4[1, 1].set_ylabel('$S_{34}$')
ax4[1, 1].set_yscale('log')
ax4[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Matrix_Elements_Par.pdf', format='pdf')
plt.show()


#Here is the calculation of the degree of linear polarization, which takes both cases of incident parallel and perpendicular
#light scattering to calculate

I_scatot = np.asarray(I_vec_Weighted_Sum_2) + np.asarray(I_vec_Weighted_Sum)
I_scadiff = np.asarray(I_vec_Weighted_Sum_2) - np.asarray(I_vec_Weighted_Sum)
# Light Scattering Difference in vertical and horizontal light polarization to Sum Ratio
DLP = -1.0 * (I_scadiff/I_scatot)
# I_vec = vertical polarization, I_vec2 = horizontal polarization


#Plot I_scatot and I_scatdiff, these two cases were used to determine the DLP, which is also plotted

f2, ax2 = plt.subplots(1, 3, figsize=(10, 5))
ax2[0].plot(theta, I_scatot, 'r-')
ax2[0].set_title('Sum of Output Intensities from Incident \nVertical and Horizontal Polarized \nElectric Field $S_{11}$ as a function of \u03B8')
ax2[0].set_xlabel('\u03B8 (\u00B0)')
ax2[0].set_ylabel('Intensity')
ax2[0].set_yscale('log')
ax2[0].grid(True)
ax2[1].plot(theta, I_scadiff, 'b-')
ax2[1].set_title('Difference Between Intensities Output \nfrom Incident Horizontal and Vertical \nPolarizations $S_{12}$ as a function of \u03B8')
ax2[1].set_xlabel('\u03B8 (\u00B0)')
ax2[1].set_ylabel('Intensity')
ax2[1].set_yscale('log')
ax2[1].grid(True)
ax2[2].plot(theta, DLP, 'g-')
ax2[2].set_title('Degree of Linear Polarization \nas a function of \u03B8')
ax2[2].set_xlabel('\u03B8 (\u00B0)')
ax2[2].set_ylabel('Intensity')
ax2[2].set_yticks(np.arange(-1.0, 0.6, 0.2))
ax2[2].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'DLP.pdf', format='pdf')
plt.show()


Data_Summary = PD.DataFrame()
Data_Summary['Theta'] = theta
Data_Summary['S11'] = S11_Weighted_Sum
Data_Summary['S12'] = S12_Weighted_Sum
Data_Summary['S33'] = S33_Weighted_Sum
Data_Summary['S34'] = S34_Weighted_Sum
Data_Summary['DLP'] = -1.0 * np.divide(S12_Weighted_Sum, S11_Weighted_Sum)
Data_Summary.to_csv(Save_Directory + 'Theory_Summary2.txt', sep=',')
'''