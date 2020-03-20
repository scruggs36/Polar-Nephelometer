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

# setting some constants
diameter = 900
radius = diameter / 2.0
wavelength = 663
m = complex(1.59, 0)

# setting Scatterer class attributes
scatterer = Scatterer(radius=radius, wavelength=wavelength, m=m, axis_ratio=1.0, or_pdf=orientation.uniform_pdf())


# setting the psd class attributes
scatterer.psd.BinnedPSD(bin_edges=1024, bin_psd=1025,  D=900)
scatterer.PSDIntegrator(num_points=1024, m_func =None, axis_ratio_func=None, geometries=(90.0, 90.0, 0.0, 180.0, 0.0, 0.0))

#
print(scatterer.get_S())

