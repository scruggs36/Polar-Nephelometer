'''
Austen K. Scruggs
03-10-2020
Description: Hoping to be able to get some useful info from some examples
'''


import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import pchip_interpolate
from pytmatrix.tmatrix import Scatterer
from pytmatrix.tmatrix import orientation
from pytmatrix.psd import PSDIntegrator, GammaPSD, BinnedPSD
import pytmatrix.tmatrix_aux as tmatrix_aux

scatterer = Scatterer(radius=.900, wavelength=663.0, m=complex(1.59, 0) , axis_ratio=1.0, orient=orientation.orient_averaged_fixed, or_pdf=tmatrix.orientation.uniform_pdf(), psd=tmatrix.Scatterer.psd, psd_integrator=tmatrix.Scatterer.psd.PSDIntegrator).get_S()
print(scatterer)
''''
#building a NLLS fitting function but we ain't here yet
def NLLS_tmatrix(x, wavelength, meas_int, meas_theta, ):
    tmat_theta = np.linspace(start=0.0, stop=180.0, num=181, endpoint=True)
    for element in tmat_theta
        scatterer = tmatrix.Scatterer(radius=x[0], wavelength=wavelength, m=x[1], axis_ratio=x[2], thet=element)

        sca_intensity(scatterer, h_pol=True)
    tmat_int_pchip = pchip_interpolate(tmat_theta, tmat_int, meas_theta)
    residuals = meas_int - tmat_int_pchip
'''