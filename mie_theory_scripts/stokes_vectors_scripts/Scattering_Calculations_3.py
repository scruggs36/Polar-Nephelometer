import numpy as np
import PyMieScatt as PMS
from math import cos, sin, atan, pi, radians, sqrt
import pandas as pd
import matplotlib.pyplot as plt

# wavelength
wavelength = 662
# Particle diameter
d= 600
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
#print(mu)
# use PyMieScatt function to calculate scattering amplitude matrix elements
S11 = []
S12 = []
S33 = []
S34 = []
SAM_Matrix_Array = []
for element in mu:
    Matrix_Elements = PMS.MatrixElements(m_particle, wavelength, d, element, m_medium)
    # the 4 x 4 scattering amplitude matrix calculation I believe assumes S3 and S4 of the 2 x 2 scattering amplitude matrix is zero due to the particles being spherical (symmetry arguments)
    SAM = np.matrix([[Matrix_Elements[0], Matrix_Elements[1], 0, 0], [Matrix_Elements[1], Matrix_Elements[0], 0, 0], [0, 0, Matrix_Elements[2], Matrix_Elements[3]], [0, 0, Matrix_Elements[3], Matrix_Elements[2]]])
    SAM_Matrix_Array.append(SAM)
    S11.append(Matrix_Elements[0])
    S12.append(Matrix_Elements[1])
    S33.append(Matrix_Elements[2])
    S34.append(Matrix_Elements[3])

# stokes ellipsometric parameter inputs:
# these inputs establish what the incident stokes vector is!
# (circularly polarized both major and minor axes are 1 / sqrt(2), change the sign
# of major axis for left or right circular polarization
semimajor_axis = 0.00001
semiminor_axis = 1

# magnitude of the semimajor and minor axes
c = sqrt((semimajor_axis ** 2) + (semiminor_axis ** 2))

# ellipticity parameter eta
eta = atan(semiminor_axis/ semimajor_axis)

# scattering plane rotation angle, gamma
azimuth = np.radians(0.0)


# stokes vector parameters, had to round to 'n' digits past the decimal for trig functions
# due to an issue with sin(pi) not equaling zero but an extremely small number
n = 3

# incident stokes vector params:
I = round(c ** 2, n)
Q = round(c ** 2, n) * round(cos(2 * eta), n) * round(cos(2 * azimuth), n)
U = round(c ** 2, n) * round(cos(2 * eta), n) * round(sin(2 * azimuth), n)
V = round(c ** 2, n) * round(sin(2 * eta), n)


# the incident stokes vector
incident_stokes_vector = [I, Q, U, V]
print('incident stokes vector: ', incident_stokes_vector)
#print(incident_stokes_vector)

# optics that transform the incident stokes vector, the ideal polarizer case!, and the ideal retarder (a waveplate)
# ideal linear polarizer
# xi is the smallest angle between the parallel electric field and the transmission axis of the polarizer
# if your passing vertically polarized light, the angle xi must be 90 degrees, such that the parallel electric field is perpendicular to the vertically polarized light which is being transmitted
xi = np.radians(90)
ideal_linear_polarizer_matrix = np.round((1/2) * np.matrix([[1, cos(2 * xi), sin(2 * xi), 0], [cos(2 * xi), cos(2 * xi) ** 2, cos(2 * xi) * sin(2 * xi), 0], [sin(2 * xi), sin(2 * xi) * cos(2 * xi), sin(2 * xi) ** 2, 0], [0, 0, 0, 0]]), n)
# here we are calculating the resultant stokes vector after the linear polarizer
output_stokes_vector_polarizer = np.round(ideal_linear_polarizer_matrix.dot(np.transpose(incident_stokes_vector)), n)
print('output stokes vector subsequent polarizer: ', output_stokes_vector_polarizer)
# transmitted intensity is given by the expression below, equation 2.87 in bohren and huffman
# here we are calculating the output stokes vector at each of the scattering angles (0-180 degrees) after the sample

output_stokes_vector_array = []
for Matrix in SAM_Matrix_Array:
    output_stokes_vector = Matrix.dot(np.transpose(output_stokes_vector_polarizer))
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


f0, ax0 = plt.subplots(2, 2, figsize=(10, 5))
ax0[0, 0].plot(theta, I_vec, 'r-', label = 'I vs \u03B8')
ax0[0, 0].set_title('Intensity as a Function of Scattering Angle for the Incident Stokes Vector \n'+ str(output_stokes_vector_polarizer) + ' on the Aerosol Sample')
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
plt.show()


# optics that transform the incident stokes vector, the ideal retarder (a waveplate)
# beta is the angle between the parallel electrical polarization and the new electric field e_1
# see figure 2.17 in bohren and huffman, this is the rotation of the waveplate in the mount!
beta = np.radians(45)
# half waveplate delta = 180, quarter waveplate delta = 90
delta = np.radians(180)

C = cos(2 * beta)
S = sin(2 * beta)
# measure of retardance delta1 - delta2 page 55 bohren and huffman, this is a property of the waveplate! probably isn't variable
ideal_linear_retarder_matrix = np.round(np.matrix([[1, 0, 0, 0], [0, (C **2) + (S ** 2) * (cos(delta)), S * C * (1 - cos(delta)), -1 * S * sin(delta)], [0, S * C * (1 - cos(delta)), (S ** 2) + (C ** 2) * cos(delta), C * sin(delta)], [0, S * sin(delta), -1 * C * sin(delta), cos(delta)]]), 3)
# the vector below is the transformation of the stokes vector after the linear retarder
output_stokes_vector_waveplate = ideal_linear_retarder_matrix.dot(np.transpose(incident_stokes_vector))
print('output stokes vector subsequent waveplate: ', output_stokes_vector_waveplate)
# the vector below is the transformation of the linear retarder transformed stokes vector after passing through a sample
output_stokes_vector_array2 = []

for Matrix in SAM_Matrix_Array:
    output_stokes_vector2 = Matrix.dot(np.transpose(output_stokes_vector_waveplate))
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


f1, ax1 = plt.subplots(2, 2, figsize=(10, 5))
ax1[0, 0].plot(theta, I_vec2, 'r-', label = 'I vs \u03B8')
ax1[0, 0].set_title('Intensity as a Function of Scattering Angle for the Incident Stokes Vector \n' + str(output_stokes_vector_waveplate) + ' on the Aerosol Sample')
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
plt.show()

I_scatot = np.asarray(I_vec2) + np.asarray(I_vec)
I_scadiff = np.asarray(I_vec2) - np.asarray(I_vec)
# Light Scattering Difference in vertical and horizontal light polarization to Sum Ratio
DLP = -1.0 * (I_scadiff/I_scatot)
# I_vec = vertical polarization, I_vec2 = horizontal polarization
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
plt.show()