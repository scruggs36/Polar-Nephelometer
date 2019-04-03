import numpy as np
from math import inf
import matplotlib.pyplot as plt

# lens parameters:
# focal length in meters
f = 14E-3 #4.8E-3
# f-number of the lens
N = 2.8 #[1.8, 2, 2.8, 4, 5.6, 8, 11, 16]
# circle of confusion in meters (set to pixel size on ccd) for 35mm format c = .025 mm
# so taking a ratio of the longest image format axes c should be 0.283 of the 0.025 which is roughly .007 for 2/3 format
c = .001
# Hyper focus distance is a distance beyond which all objects can be brought
# into an "acceptable" focus. As the hyperfocal distance is the focus distance
#  giving the maximum depth of field, it is the most desirable distance to set
# the focus of a fixed-focus camera.
H = f + (f**2)/(N*c)
print('hyper focus distance in meters: ',H)
# distance at which camera is focused in meters
s = 0.015
# the below equations are valid when the s >> f
# depth of field near limit
if s >= H:
    DN = H/2
    print('depth of field near limit in meters: ', DN)
    DF = inf
    print('depth of field far limit in meters: ', DF)
    print('depth of field', DF - DN)
if s < H:
    DN = (H*s)/(H + s)
    print ('depth of field near limit in meters: ', DN)
    # depth of field far limit
    DF = (H*s)/(H - s)
    print ('depth of field far limit in meters: ', DF)
    # when the subject distance is H, then DN = H/2 and DF = infinity
    print('depth of field' ,DF - DN)

'''
For Fujinon FE185C057HA-1 2/3" fish eye lens:
hyper focus distance in meters:  0.33241224489795923
depth of field near limit in meters:  0.009707954369360242
depth of field far limit in meters:  0.01031016191718044
depth of field 0.0006022075478201976
'''

'''
For the Ricoh lens we have:
hyper focus distance in meters:  1.8333714285714282
depth of field near limit in meters:  0.000999454854134542
depth of field far limit in meters:  0.001000545740882229
depth of field 1.0908867476870296e-06
'''