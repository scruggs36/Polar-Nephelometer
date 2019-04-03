from math import cos, sin, atan, pi, radians, sqrt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import PyMieScatt as PMS

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
#print(incident_stokes_vector)

# optics that transform the incident stokes vector, the ideal polarizer case!, and the ideal retarder (a waveplate)
# ideal linear polarizer
# xi is the smallest angle between the parallel electric field and the transmission axis of the polarizer
I_pol = []
Q_pol = []
U_pol = []
V_pol = []
xi = np.radians(np.arange(0, 181, 1))
for angle in xi:
    ideal_linear_polarizer_matrix = np.round((1/2) * np.matrix([[1, cos(2 * angle), sin(2 * angle), 0], [cos(2 * angle), cos(2 * angle) ** 2, cos(2 * angle) * sin(2 * angle), 0], [sin(2 * angle), sin(2 * angle) * cos(2 * angle), sin(2 * angle) ** 2, 0], [0, 0, 0, 0]]), n)
    #print(ideal_linear_polarizer_matrix)
    output_stokes_vector = np.round(ideal_linear_polarizer_matrix.dot(np.transpose(incident_stokes_vector)), n)
    I_pol.append(output_stokes_vector[0])
    Q_pol.append(output_stokes_vector[1])
    U_pol.append(output_stokes_vector[2])
    V_pol.append(output_stokes_vector[3])
    # transmitted intensity is given by the expression below, equation 2.87 in bohren and huffman
    I_trans = (1 / 2) * (incident_stokes_vector[0] + (incident_stokes_vector[1] * cos(2 * angle)) + (incident_stokes_vector[2] * sin(2 * angle)))
    # maximum intensity is where xi = azimuthal angle
    I_max = (1 / 2) * (incident_stokes_vector[0] + (incident_stokes_vector[1] * cos(2 * angle)) + (incident_stokes_vector[2] * sin(2 * angle)))
    # minumum intensity is where xi = azimuthal angle + pi / 2
    I_min = (1 / 2) * (incident_stokes_vector[0] - (incident_stokes_vector[1] * cos(2 * angle)) - (incident_stokes_vector[2] * sin(2 * angle)))
    # lets see if calculations make sense by comparing incident beam and output beam for the case of the linear polarizer
    '''
    print('case of the linear polarizer:')
    print('incident stokes vector: ', incident_stokes_vector)
    print('parameters: \u03B3 = ' + str(azimuth * (180/pi)) + ' degrees and ' + '\u03BE = ' + str(xi * 180/pi) + ' degrees')
    print('output stokes vector after ideal linear polarizer: ', output_stokes_vector)
    print('note: \u03B3 governs the rotation angle of the scattering plane relative to the xz plane (rotation about the z axis) given the direction of propogation is along the z-axis.')
    print('note: \u03BE governs the transmission axis of the linear polarizer, 0\u00B0 is horizontal polarization 90\u00B0 is vertical polarization')
    #print('transmitted intensity: ', I_trans)
    print('\n')
    # maximum intensity is where the angle between the parallel electric field and the transmission axis (direction of propagation) is
    # equal to the angle governing the rotation of the scattering plane
    #print('maximum intensity where \u03BE = \u03B3: ', I_max)
    # minimum intensity is where the thee smallest angle between the parallel electric field and the transmission axis is
    # equal to the angle governing the rotation of the scattering plane + 90 degrees
    #print('minimum intensity where \u03BE = \u03B3 + \u03C0/2: ', I_min)
    '''
# plotting
f0, ax0 = plt.subplots(2, 2)
ax0[0, 0].plot(xi * (180/pi), I_pol, 'r-')
ax0[0, 0].set_xlabel('\u03B2 (°)')
ax0[0, 0].set_ylabel('Normalized Intensity')
ax0[0, 0].set_title('Intensity Versus Linear Polarizer Rotation')
ax0[0, 0].grid()
ax0[0, 1].plot(xi * (180/pi),Q_pol, 'b-')
ax0[0, 1].set_xlabel('\u03B2 (°)')
ax0[0, 1].set_ylabel('Normalized Q')
ax0[0, 1].set_title('Normalized Q Versus Linear Polarizer Rotation')
ax0[0, 1].grid()
ax0[1, 0].plot(xi * (180/pi),U_pol, 'g-')
ax0[1, 0].set_xlabel('\u03B2 (°)')
ax0[1, 0].set_ylabel('Normalized U')
ax0[1, 0].set_title('Normalized U Versus Linear Polarizer Rotation')
ax0[1, 0].grid()
ax0[1, 1].plot(xi * (180/pi), V_pol, 'y-')
ax0[1, 1].set_xlabel('\u03B2 (°)')
ax0[1, 1].set_ylabel('Normalized V')
ax0[1, 1].set_title('Normalized V Versus Linear Polarizer Rotation')
ax0[1, 1].grid()
plt.tight_layout()
plt.show()


# optics that transform the incident stokes vector, the ideal retarder (a waveplate)
# beta is the angle between the parallel electrical polarization and the new electric field e_1
# see figure 2.17 in bohren and huffman, this is the rotation of the waveplate in the mount!
beta = np.radians(np.arange(0, 181, 1))
delta = np.radians(180)
I_array = []
Q_array = []
U_array = []
V_array = []
for i in beta:
    C = cos(2 * i)
    S = sin(2 * i)
    # measure of retardance delta1 - delta2 page 55 bohren and huffman, this is a property of the waveplate! probably isn't variable
    ideal_linear_retarder_matrix = np.round(np.matrix([[1, 0, 0, 0], [0, (C **2) + (S ** 2) * (cos(delta)), S * C * (1 - cos(delta)), -1 * S * sin(delta)], [0, S * C * (1 - cos(delta)), (S ** 2) + (C ** 2) * cos(delta), C * sin(delta)], [0, S * sin(delta), -1 * C * sin(delta), cos(delta)]]), 3)
    #print(ideal_linear_retarder_matrix)
    output_stokes_vector2 = ideal_linear_retarder_matrix.dot(np.transpose(incident_stokes_vector))
    # okay, now lets see what happens to the incident stokes vector for the case of the linear retarder
    I_array.append(output_stokes_vector2[0])
    Q_array.append(output_stokes_vector2[1])
    U_array.append(output_stokes_vector2[2])
    V_array.append(output_stokes_vector2[3])
# plotting
f1, ax1 = plt.subplots(2, 2)
ax1[0, 0].plot(beta * (180/pi), I_array, 'r-')
ax1[0, 0].set_xlabel('\u03B2 (°)')
ax1[0, 0].set_ylabel('Normalized Intensity')
ax1[0, 0].set_title('Intensity Versus Half Waveplate Rotation')
ax1[0, 0].grid()
ax1[0, 1].plot(beta * (180/pi),Q_array, 'b-')
ax1[0, 1].set_xlabel('\u03B2 (°)')
ax1[0, 1].set_ylabel('Normalized Q')
ax1[0, 1].set_title('Normalized Q Versus Half Waveplate Rotation')
ax1[0, 1].grid()
ax1[1, 0].plot(beta * (180/pi),U_array, 'g-')
ax1[1, 0].set_xlabel('\u03B2 (°)')
ax1[1, 0].set_ylabel('Normalized U')
ax1[1, 0].set_title('Normalized U Versus Half Waveplate Rotation')
ax1[1, 0].grid()
ax1[1, 1].plot(beta * (180/pi), V_array, 'y-')
ax1[1, 1].set_xlabel('\u03B2 (°)')
ax1[1, 1].set_ylabel('Normalized V')
ax1[1, 1].set_title('Normalized V Versus Half Waveplate Rotation')
ax1[1, 1].grid()
plt.tight_layout()
plt.show()



#('case of the ideal linear retarder (waveplate)')
#print('incident stokes vector: ', incident_stokes_vector)
#print('parameters: \u03B2 = ' + str(beta * (180/pi)) + ' degrees and ' + '\u03B4 = ' + str(delta * 180/pi) + ' degrees')
#print('output stokes vector after ideal linear retarder: ', output_stokes_vector2)
# \u208'input number here' prints subscript number!
#print('note: \u03B2 is the angle between the vector representing the parallel electric field and the vector representing the axis of the linear retarder (e\u2081),\n for example a λ/4 waveplate with its axes oriented at 45° to linear polarization produces circular polarization.')
#print('note: \u03B4 is the retardance, which i believe is the phase delay (phase difference) between electric vector e\u2081 and e\u2082,\n a waveplate divides an incident electric vector into two linearly polarized mutually orthogonal components called e\u2081 and e\u2082 and introduces a phase difference between them.\n (described page 54 of bohren and huffman)')
#'''
