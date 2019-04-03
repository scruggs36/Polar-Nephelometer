import scipy.constants as sc
import numpy as np
from math import pi


Lasernm = 632.8E-9
LaserPwr = .001



def FindWavenumber(Lambda):
    wavnum= ((((sc.h * sc.c)/Lambda)/(1.60218E-19))*8065.54)
    return wavnum


def RI(A, B, C, wavnum):
    NV = ((A + (B/(C - wavnum**2.0)))/(10E8)) + 1.0
    return NV


def Sigma(N, nv, Fk, wavnum):
    sigma = ((24 * (pi**3) * (wavnum**4))/N**2) * ((((nv**2) - 1)/((nv**2) + 2))**2) * Fk
    return sigma


RI_He = [2283, 1.8102E13, 1.5342E10]
He_Fk = 1
RI_O2 = [20564.8, 2.480899E13, 4.09E9]
O2_Fk = 1.09 + (1.385E-11 * FindWavenumber(Lasernm)**2) + (1.448E-20 * FindWavenumber(Lasernm)**2)
#RI_CO2 = ()
#CO2_Fk =

if FindWavenumber(Lasernm) < 21360:
    RI_N2 = [6498.2, 307.4335E13, 14.4E9]
    N2_Fk = (1.034 + 3.17E-12 * FindWavenumber(Lasernm))
else:
    RI_N2 = [5677.465, 318.81874E12, 14.4E9]
    N2_Fk = (1.034 + 3.17E-12 * FindWavenumber(Lasernm))

#Quick Calculation
#Correct refractive index for N2 is 1.00023 for some reason we have 1.00223, so this needs to change
wavenumb = FindWavenumber(Lasernm)
N2_Ref_Ind = RI(RI_N2[0], RI_N2[1], RI_N2[2], wavenumb)
N2_Sigma = Sigma(2.546899E19, N2_Ref_Ind, N2_Fk, wavenumb)
Mike_Sigma = 1.2577E-15 * 632.8**-4.1814

#Check to see if cross sections make sense by calculating the extinction in Mm^-1
Mike_Sigma*2.546899E19*1E8
N2_Sigma*2.546899E19*1E8

#Calculate attenuation with simple beers law type equation
SpotSizecm = 0.5
NephCellLengthcm = 47 * 2.54
BeamVolume = pi * ((SpotSizecm/2)**2) * (NephCellLengthcm * 2)
N_Gas = 2.546899E19 * BeamVolume
Na = 6.022E23
N2_Epsilon = (Na/np.log(10)) * N2_Sigma
N2_Attenuation = np.exp(-1.0*(N2_Epsilon * (N_Gas/6.022E23) * (NephCellLengthcm * 2)))
N2_Percent_Attenuation = 100 - (N2_Attenuation * 100)
N2_Percent_Attenuation





