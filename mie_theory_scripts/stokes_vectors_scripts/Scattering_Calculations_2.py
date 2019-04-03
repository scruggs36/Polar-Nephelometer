from numpy import *
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from bhmie import bhmie
# Inputs
save_dir = '/home/austen/Documents/'
wavelengths = [663]
#diameters = linspace(100, 900, 9)
diameters = [600]
# nang must be an integer, its the number of angles you want between 0 and pi/2
nang = 91
rads = linspace(0, pi/2, nang)
angles = rads*(180/pi)
# angles2 is for plots
angles2 = linspace(0,180,181)
# preallocating arrays
size_array = []
wavelength_array = []
SizeParam_diameters_array = []
s1_2darray = []
s2_2darray = []
qext_diameter_array = []
qback_diameter_array = []
qsca_diameter_array = []
gsca_diameter_array = []
SizeParam_array = []
s1_3darray = []
s2_3darray = []
qext_array = []
qback_array = []
qsca_array = []
gsca_array = []
legend_array = [str(x) + 'nm' for x in diameters]


for counter1, i in enumerate(wavelengths):
    for counter2, j in enumerate(diameters):
        # Nitrogen Medium Refractive Index
        medium_ref_index = 1.000277
        # Cauchy parameters for PSL
        A = 1.5725
        B = 0.0031080
        C = 0.00034779
        # n of refractive index for PSL as cauchy equation
        n_cauchy = A + (B / i ** 2) + (C / (i ** 4))
        CompRefIndex = complex(1.59, 0.0005)
        k = (pi * medium_ref_index) / i
        SizeParam = k * j
        SizeParam_diameters_array.append(SizeParam)
        s1, s2, qext, qsca, qback, gsca = bhmie(SizeParam, CompRefIndex, nang)
        print(s1)
        print(s2)
        s1_2darray.append(abs(s1) ** 2)
        s2_2darray.append(abs(s2) ** 2)
        qext_diameter_array.append(qext)
        qsca_diameter_array.append(gsca)
        qback_diameter_array.append(qback)
        gsca_diameter_array.append(gsca)
        # note that after we exit this inner most loop counter2 resets itself back to zero!
        if counter2 == len(diameters) - 1:
            SizeParam_array.append(SizeParam_diameters_array)
            s1_3darray.append(s1_2darray)
            s2_3darray.append(s2_2darray)
            qext_array.append(qext_diameter_array)
            qsca_array.append(qsca_diameter_array)
            qback_array.append(qback_diameter_array)
            gsca_array.append(gsca_diameter_array)
            SizeParam_diameters_array = []
            s1_2darray = []
            s2_2darray = []
            qext_diameter_array = []
            qback_diameter_array = []
            qsca_diameter_array = []
            gsca_diameter_array = []
            continue

#print(asarray(s1_3darray).shape)
#'''
# compute S1
S1_DF = pd.DataFrame()
plt.figure(figsize=(10, 6))
for counter1, z, in enumerate(wavelengths):
    for counter2, l in enumerate(s1_3darray[counter1]):
        S1_DF['particle size'] = full(len(angles2),fill_value=diameters[counter2])
        S1_DF['wavelength'] = full(len(angles2), fill_value=z)
        S1_DF['angles'] = angles2
        S1_DF['S1^2'] = l
        S1_DF.to_csv(save_dir + 'S1_Wavelength' + str(z)+ 'nm_' + 'Size' + str(diameters[counter2]) + 'nm.txt', sep=',')
        plt.plot(angles2, l)
    plt.yscale('log')
    plt.title('Scattering Diagram  for ' + str(z) + 'nm Light Polarized Perpendicular\n to the Scattering Plane (S1 Scattering Diagram)')
    plt.xlabel('$\Theta$')
    plt.ylabel('Scattered Irradiance')
    plt.grid(True, which='major', color='lightgray', linestyle='-')
    plt.grid(True, which='minor', color='lightgray', linestyle='-')
    plt.legend(legend_array, loc=7, bbox_to_anchor=(1.20, 0.5))
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_dir + 'S1_Wavelength' + str(z) + 'nm.pdf')
    # plt.clf() clears the current figure! this is how i am resetting figures at the end to save the next wavelength
    plt.clf()

# reset these arrays to empty
size_array = []
wavelength_array = []

# compute S2
S2_DF = pd.DataFrame()
plt.figure(figsize=(10, 6))
for counter1, m in enumerate(wavelengths):
    for counter2, n in enumerate(s2_3darray[counter1]):
        size_array.append(diameters[counter2])
        wavelength_array.append(m)
        S2_DF['particle size'] = full(len(angles2), fill_value=diameters[counter2])
        S2_DF['wavelength'] = full(len(angles2), fill_value=m)
        S2_DF['angles'] = angles2
        S2_DF['S2^2'] = n
        S2_DF.to_csv(save_dir + 'S2_Wavelength' + str(z)+ 'nm_' + 'Size' + str(diameters[counter2]) + 'nm.txt', sep=',')
        plt.plot(angles2, n)
    plt.yscale('log')
    plt.title('Scattering Diagram for ' + str(m) + 'nm Light Polarized Parallel\n to the Scattering Plane (S2 Scattering Diagram)')
    plt.xlabel('$\Theta$')
    plt.ylabel('Scattered Irradiance')
    plt.grid(True, which='major', color='lightgray', linestyle='-')
    plt.grid(True, which='minor', color='lightgray', linestyle='-')
    plt.legend(legend_array, loc=7, bbox_to_anchor=(1.20, 0.5))
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_dir + 'S2_Wavelength' + str(m) + 'nm.pdf')
    # plt.clf() clears the current figure! this is how i am resetting figures at the end to save the next wavelength
    plt.clf()

# reset these arrays to empty
size_array = []
wavelength_array = []
#'''
'''
Qext_DF = pd.DataFrame()
Qext_DF['partcle sizes'] = diameters
for counter1, o in enumerate(wavelengths):
    Qext_DF['size parameter at $\lambda$ ' + str(o) + 'nm'] = SizeParam_array[counter1]
    Qext_DF['gsca at $\lambda$ ' + str(o) + 'nm'] = qext_array[counter1]
    plt.plot(SizeParam_array[counter1], qext_array[counter1], label='Wavelength Fixed at ' + str(o) + 'nm')
plt.title('Extinction Efficiency as a Function of Size\n Parameter at Various Wavelengths')
plt.xlabel('Size Parameter')
plt.ylabel('$Q_{ext}$')
plt.grid(True, which='major', color='lightgray', linestyle='-')
plt.grid(True, which='minor', color='lightgray', linestyle='-')
plt.legend(loc=0)
#plt.show()
plt.savefig(save_dir + 'Qext.pdf', format='pdf')
plt.clf()
Qext_DF.to_csv(save_dir + 'Qext_DF.txt', sep=',')


Qsca_DF = pd.DataFrame()
Qsca_DF['partcle sizes'] = diameters
for counter1, p in enumerate(wavelengths):
    Qsca_DF['size parameter at $\lambda$ ' + str(p) + 'nm'] = SizeParam_array[counter1]
    Qsca_DF['gsca at $\lambda$ ' + str(p) + 'nm'] = qsca_array[counter1]
    plt.plot(SizeParam_array[counter1], qsca_array[counter1], label='Wavelength Fixed at ' + str(p) + 'nm')
plt.title('Scattering Efficiency as a Function of Size\n Parameter at Various Wavelengths')
plt.xlabel('Size Parameter')
plt.ylabel('$Q_{sca}$')
plt.grid(True, which='major', color='lightgray', linestyle='-')
plt.grid(True, which='minor', color='lightgray', linestyle='-')
plt.legend(loc=0)
#plt.show()
plt.savefig(save_dir + 'Qsca.pdf', format='pdf')
plt.clf()
Qsca_DF.to_csv(save_dir + 'Qsca_DF.txt', sep=',')


Qback_DF = pd.DataFrame()
Qback_DF['partcle sizes'] = diameters
for counter1, q in enumerate(wavelengths):
    Qback_DF['size parameter at $\lambda$ ' + str(q) + 'nm'] = SizeParam_array[counter1]
    Qback_DF['Qback at $\lambda$ ' + str(q) + 'nm'] = qback_array[counter1]
    plt.plot(SizeParam_array[counter1], qback_array[counter1], label='Wavelength Fixed at ' + str(q) + 'nm')
plt.title('Back Scattering Efficiency as a Function of Size\n Parameter at Various Wavelengths')
plt.xlabel('Size Parameter')
plt.ylabel('$Q_{back}$')
plt.grid(True, which='major', color='lightgray', linestyle='-')
plt.grid(True, which='minor', color='lightgray', linestyle='-')
plt.legend(loc=0)
#plt.show()
plt.savefig(save_dir + 'Qback.pdf', format='pdf')
plt.clf()
Qback_DF.to_csv(save_dir + 'Qback_DF.txt', sep=',')


gsca_DF = pd.DataFrame()
gsca_DF['partcle sizes'] = diameters
for counter1, r in enumerate(wavelengths):
    gsca_DF['size parameter at $\lambda$ ' + str(r) + 'nm'] = SizeParam_array[counter1]
    gsca_DF['gsca at $\lambda$ ' + str(r) + 'nm'] = gsca_array[counter1]
    plt.plot(SizeParam_array[counter1], gsca_array[counter1], label='Wavelength Fixed at ' + str(r) + 'nm')
plt.title('Asymmetry Parameter as a Function of Size\n Parameter at Various Wavelengths')
plt.xlabel('Size Parameter')
plt.ylabel('$g_{sca}$')
plt.grid(True, which='major', color='lightgray', linestyle='-')
plt.grid(True, which='minor', color='lightgray', linestyle='-')
plt.legend(loc=0)
#plt.show()
plt.savefig(save_dir + 'gsca.pdf', format='pdf')
plt.clf()
gsca_DF.to_csv(save_dir + 'gsca_DF.txt', sep=',')
'''
