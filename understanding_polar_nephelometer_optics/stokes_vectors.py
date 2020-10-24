'''
Austen K. Scruggs
02/23/2020
Description: Optics understand the influence of optics on stokes vectors using python
'''
import numpy as np
from math import pi

# for LCVR
def rotator(two_x_theta_degrees):
    radians = (two_x_theta_degrees * pi)/180.0
    # its actually 2 * radians in each of the sine and cosine inputs, but we decided to have it work like the LCVR
    M = np.asmatrix(np.array([[1, 0, 0, 0] ,[0, np.cos(radians), np.sin(radians), 0] ,[0, -1*np.sin(radians), np.cos(radians), 0] ,[0, 0, 0, 1]]))
    return M

#waveplates are linear retarders! with defined phase shift!
def linear_retarders(quarter_wave_plate, half_wave_plate, fast_axis_vertical):
    if (quarter_wave_plate==True) and (half_wave_plate==False) and (fast_axis_vertical==True):
        M = np.asmatrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0] ,[0, 0, 0, -1], [0, 0, 1, 0]]))
    if (quarter_wave_plate==True) and (half_wave_plate==False) and (fast_axis_vertical==False):
        M = np.asmatrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]))
    if (quarter_wave_plate==False) and (half_wave_plate==True):
        M = np.asmatrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]))
    return M


def linear_polarizer():
    M = np.asmatrix(np.array([[],[],[],[]]))
    return M


def spatial_filter(wavelength, laser_beam_diameter_1_over_e_squared, aspheric_lens_focal_length):
    diffraction_limited_spot_size = (wavelength * aspheric_lens_focal_length) / (laser_beam_diameter_1_over_e_squared/2.0)
    pinhole_size = diffraction_limited_spot_size + (diffraction_limited_spot_size * 0.3)
    return [diffraction_limited_spot_size, pinhole_size]


# Q=1: s polarized, parallel to the scattering plane (perpendicular to the plane of incidence)
# Q=-1: p polarized perpendicularly to the scattering plane (parallel to the plane of incidence) (this is coming out the laser)
I = 1
Q = -1
U = 0
V = 0



# polar nephelometer polarization manipulation LCVR followed by a quarter waveplate at zero degrees (fast axis is vertical)
# our current setup produces left hand circularly polarized light
input_stokes_vector = np.asmatrix(np.array([[I], [Q], [U], [V]]))
transformation_1 = np.dot(rotator(two_x_theta_degrees=90), input_stokes_vector)
print('transformation 1:\n', transformation_1)
transformation_2 = np.dot(linear_retarders(quarter_wave_plate=True, half_wave_plate=False, fast_axis_vertical=True), transformation_1)
print('transformation_2:\n', transformation_2)

# spatial filter calculations
# choosing the correct optics and pinhole for spatial filter,

# units in meters
wavelength_red = 663E-9
wavelength_green = 532E-9
wavelength_blue = 405E-9
# units in millimeters
laser_beam_diameter = 0.9
# units in millimeters
lens_focal_length = 50.0

red_diffraction_limited_spot_size, red_pinhole_size = spatial_filter(wavelength=wavelength_red, laser_beam_diameter_1_over_e_squared=laser_beam_diameter, aspheric_lens_focal_length=lens_focal_length)
green_diffraction_limited_spot_size, green_pinhole_size = spatial_filter(wavelength=wavelength_green, laser_beam_diameter_1_over_e_squared=laser_beam_diameter, aspheric_lens_focal_length=lens_focal_length)
blue_diffraction_limited_spot_size, blue_pinhole_size = spatial_filter(wavelength=wavelength_blue, laser_beam_diameter_1_over_e_squared=laser_beam_diameter, aspheric_lens_focal_length=lens_focal_length)

print('red wavelength pinhole size: ', red_pinhole_size)
print('green wavelength pinhole size: ', green_pinhole_size)
print('blue wavelength pinhole size: ', blue_pinhole_size)