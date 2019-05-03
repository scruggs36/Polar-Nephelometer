'''
Austen K. Scruggs
09-13-2018
Description: This is a Mie Theory script that a computes scattering diagram
taking into account identical particles in a given gaussian size distribution
'''

import numpy as np
import PyMieScatt as PMS
from math import cos, sin, atan, pi, radians, sqrt, log10, exp
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

Save_Directory = '/home/austen/Documents/'
'''
First we will calculate the particle size distribution and plot it
'''
# geometric standard deviation
sigma_g = 1.05
# Particle diameter, geometric mean of the particle diameter
d= 903
# write function for LogNormal distribution (same as Tami Bonds)
def LogNormal(diam, mu, gsd):
    return np.exp(-1 * ((np.log10(diam/mu)) ** 2) / (2 * np.log10(gsd) ** 2)) / (sqrt(2 * pi) * np.log10(gsd))
def Gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
# plot
size_axis = np.arange(d-d + 1, d*3, 1)
print(sp.integrate.simps(Gaussian(size_axis, mu=d, sigma=sigma_g), size_axis, dx=1))
f, ax = plt.subplots(figsize=(6, 6))
#ax.plot(size_axis, [LogNormal(element, mu=d, gsd=sigma_g) for element in size_axis])
ax.plot(size_axis, LogNormal(size_axis, mu=d, gsd=sigma_g) / np.linalg.norm(LogNormal(size_axis, mu=d, gsd=sigma_g)), 'r-', label='Tami Bond LogNormal $\u03c3_{g}$=1.05' )
ax.plot(size_axis, Gaussian(size_axis, mu=d, sigma=4.1), 'b-', label='Gaussian Dist. \u03bc=903nm \u03c3=4.1')
ax.set_xlabel('particle diameter (nm)')
ax.set_ylabel('Normalized $dN/Log_{10}(D)$')
ax.set_title('Distributions Used for Mie Theory Calculations')
ax.grid(True)
plt.legend(loc=1)
plt.savefig(Save_Directory + 'Mie_Distributions.pdf', format='pdf')
plt.show()

'''
Here we are conducting the Mie theory, and we are going to loop through the sizes
'''
# wavelength
wavelength = 662
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

# use PyMieScatt function to calculate scattering amplitude matrix elements
S11 = []
S12 = []
S33 = []
S34 = []
SAM_Matrix_Array = []
for element in mu:
    Matrix_Elements = PMS.MatrixElements(m_particle, wavelength, d, element, m_medium)
    # the 4 x 4 scattering amplitude matrix calculation I believe assumes S3 and S4 of the 2 x 2 scattering amplitude matrix are zero due to the particles being spherical (symmetry arguments)
    SAM = np.matrix([[Matrix_Elements[0], Matrix_Elements[1], 0, 0], [Matrix_Elements[1], Matrix_Elements[0], 0, 0], [0, 0, Matrix_Elements[2], Matrix_Elements[3]], [0, 0, Matrix_Elements[3], Matrix_Elements[2]]])
    SAM_Matrix_Array.append(SAM)
    S11.append(Matrix_Elements[0])
    S12.append(Matrix_Elements[1])
    S33.append(Matrix_Elements[2])
    S34.append(Matrix_Elements[3])


'''
Take stokes vectors into account, this is the case where the incident stokes vector is vertically polarized, 
or perpendicularly polarized to the scattering plane
'''
incident_stokes_vector_perp = [1, -1, 0, 0]
# transmitted intensity is given by the expression below, equation 2.87 in bohren and huffman
# here we are calculating the output stokes vector at each of the scattering angles (0-180 degrees) after the sample

output_stokes_vector_array = []
for Matrix in SAM_Matrix_Array:
    output_stokes_vector = Matrix.dot(np.transpose(incident_stokes_vector_perp))
    output_stokes_vector_array.append(output_stokes_vector)

I_vec = []
Q_vec = []
U_vec = []
V_vec = []

for element in output_stokes_vector_array:
    I_vec.append(np.asarray(element).flatten()[0])
    Q_vec.append(np.asarray(element).flatten()[1])
    U_vec.append(np.asarray(element).flatten()[2])
    V_vec.append(np.asarray(element).flatten()[3])

'''
Plot the vertical, or perpendicular to plane polarized incident light scattering case
'''
f0, ax0 = plt.subplots(2, 2, figsize=(10, 5))
ax0[0, 0].plot(theta, I_vec, 'r-', label = 'I vs \u03B8')
ax0[0, 0].set_title('Intensity as a Function of Scattering Angle for the Incident Stokes Vector \n'+ str(incident_stokes_vector_perp) + ' on the Aerosol Sample')
ax0[0, 0].set_ylabel('I')
ax0[0, 0].set_yscale('symlog')
ax0[0, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[0, 0].grid(True)
ax0[1, 0].plot(theta, Q_vec, 'b-', label = 'Q vs \u03B8')
ax0[1, 0].set_title('Stokes Element Q as a Function of Scattering Angle')
ax0[1, 0].set_ylabel('Q')
ax0[1, 0].set_yscale('symlog')
ax0[1, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[1, 0].grid(True)
ax0[0, 1].plot(theta, U_vec, 'g-', label = 'U vs \u03B8')
ax0[0, 1].set_title('Stokes Element U as a Function of Scattering Angle')
ax0[0, 1].set_ylabel('U')
ax0[0, 1].set_yscale('symlog')
ax0[0, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[0, 1].grid(True)
ax0[1, 1].plot(theta, V_vec, 'y-', label = 'V vs \u03B8')
ax0[1, 1].set_title('Stokes Element V as a Function of Scattering Angle')
ax0[1, 1].set_ylabel('V')
ax0[1, 1].set_yscale('symlog')
ax0[1, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax0[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Incident_Vertically_Polarized_Light_Scattering.pdf', format='pdf')
plt.show()

'''
Take stokes vectors into account, this is the case where the incident stokes vector is horizontally polarized,
or polarized parallel (inside) the scattering plane
'''
incident_stokes_vector_par = [1, 1, 0, 0]
output_stokes_vector_array2 = []
for Matrix in SAM_Matrix_Array:
    output_stokes_vector2 = Matrix.dot(np.transpose(incident_stokes_vector_par))
    output_stokes_vector_array2.append(output_stokes_vector2)


I_vec2 = []
Q_vec2 = []
U_vec2 = []
V_vec2 = []

for element in output_stokes_vector_array2:
    I_vec2.append(np.asarray(element).flatten()[0])
    Q_vec2.append(np.asarray(element).flatten()[1])
    U_vec2.append(np.asarray(element).flatten()[2])
    V_vec2.append(np.asarray(element).flatten()[3])

'''
Plot the horizontal, or in plane polarized incident light scattering case
'''

f1, ax1 = plt.subplots(2, 2, figsize=(10, 5))
ax1[0, 0].plot(theta, I_vec2, 'r-', label = 'I vs \u03B8')
ax1[0, 0].set_title('Intensity as a Function of Scattering Angle for the Incident Stokes Vector \n' + str(incident_stokes_vector_par) + ' on the Aerosol Sample')
ax1[0, 0].set_ylabel('I')
ax1[0, 0].set_yscale('symlog')
ax1[0, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[0, 0].grid(True)
ax1[1, 0].plot(theta, Q_vec2, 'b-', label = 'Q vs \u03B8')
ax1[1, 0].set_title('Stokes Element Q as a Function of Scattering Angle')
ax1[1, 0].set_ylabel('Q')
ax1[1, 0].set_yscale('symlog')
ax1[1, 0].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[1, 0].grid(True)
ax1[0, 1].plot(theta, U_vec2, 'g-', label = 'U vs \u03B8')
ax1[0, 1].set_title('Stokes Element U as a Function of Scattering Angle')
ax1[0, 1].set_ylabel('U')
ax1[0, 1].set_yscale('symlog')
ax1[0, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[0, 1].grid(True)
ax1[1, 1].plot(theta, V_vec2, 'y-', label = 'V vs \u03B8')
ax1[1, 1].set_title('Stokes Element V as a Function of Scattering Angle')
ax1[1, 1].set_ylabel('V')
ax1[1, 1].set_yscale('symlog')
ax1[1, 1].set_xlabel('Scattering Angle \u03B8 (\u00B0)')
ax1[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Incident_Horizontally_Polarized_Light_Scattering.pdf', format='pdf')
plt.show()

'''
Here is the calculation of the degree of linear polarization, which takes both cases of incident parallel and perpendicular
light scattering to calculate
'''
I_scatot = np.asarray(I_vec2) + np.asarray(I_vec)
I_scadiff = np.asarray(I_vec2) - np.asarray(I_vec)
# Light Scattering Difference in vertical and horizontal light polarization to Sum Ratio
DLP = -1.0 * (I_scadiff/I_scatot)
# I_vec = vertical polarization, I_vec2 = horizontal polarization

'''
Plot I_scatot and I_scatdiff, these two cases were used to determine the DLP, which is also plotted
'''
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