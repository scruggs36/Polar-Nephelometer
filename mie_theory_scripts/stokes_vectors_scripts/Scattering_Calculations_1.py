import matplotlib.pyplot as plt
from numpy import *
from math import pi
from bhmie import bhmie

wavelength = 663.0E-9
diameter = 701.0E-9
medium_ref_index = 1.000277
k = (pi * medium_ref_index)/wavelength
SizeParam = k * diameter
print(SizeParam)
CompRefIndex = complex(1.59, 0.0005)
# nang must be an integer, its the number of angles you want between 0 and pi/2
nang = 91
rads = linspace(0, pi/2, nang)
angles = rads*(180/pi)
angles2 = linspace(0,180,181)
print(size(angles2))
s1, s2, qext, qsca, qback, gsca = bhmie(SizeParam, CompRefIndex, nang)

plt.plot(angles2, abs(s2)**2, '-r')
plt.yscale('log')
plt.title('Scattering Diagram of Light Polarized Parallel \n to the Scattering Plane (S2)')
plt.xlabel('$\Theta$')
plt.ylabel('Scattered Irradiance')
plt.grid(True, which='major', color='lightgray', linestyle='-')
plt.grid(True, which='minor', color='lightgray', linestyle='-')
plt.show()


plt.plot(angles2, abs(s1)**2, '-r')
plt.yscale('log')
plt.title('Scattering Diagram of Light Polarized Perpendicular \n to the Scattering Plane (S1)')
plt.xlabel('$\Theta$')
plt.ylabel('Scattered Irradiance')
plt.grid(True, which='major', color='lightgray', linestyle='-')
plt.grid(True, which='minor', color='lightgray', linestyle='-')
plt.show()
'''
plt.plot()
print(s1)
print(s2)
print(qext)
print(qsca)
print(qback)
print(gsca)
'''
