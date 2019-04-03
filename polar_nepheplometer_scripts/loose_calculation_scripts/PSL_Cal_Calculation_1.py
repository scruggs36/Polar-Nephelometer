from Neph_Functions import *
from scipy.interpolate import PchipInterpolator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Path_Perpendicular_MieTheory_600nm = '/home/austen/Documents/Compare_Data/Mie_Theory/S1_Wavelength663nm_Size600nm.txt'
MieTheory_PSL600nm_S1_Perpendicular = '/home/austen/Documents/Compare_Data/Mie_Theory/S1_Wavelength663nm_Size600nm.txt'
MieTheory_PSL701nm_S2_Parallel = '/home/austen/Documents/Compare_Data/Mie_Theory/S2_Wavelength663nm_Size701nm.txt'
MieTheory_PSL600nm_S2_Parallel = '/home/austen/Documents/Compare_Data/Mie_Theory/S2_Wavelength663nm_Size600nm.txt'
EXP_PSL600nm_S1_Path_Perpendicular = '/home/austen/Documents/Compare_Data/Perpendicular_to_Scattering_Plane/PSL_600nm/SD_PSL600nm_60s_06_04_2018_Perpendicular.txt'
EXP_PSL701nm_S2_Path_Parallel = '/home/austen/Documents/Compare_Data/Parallel_to_Scattering_Plane/PSL_701nm/SD_PSL701nm_20s_05_23_2018_Parallel.txt'
EXP_PSL600nm_S2_Path_Parallel = '/home/austen/Documents/Compare_Data/Parallel_to_Scattering_Plane/PSL_600nm/SD_PSL600nm_30s_05_24_2018_Parallel.txt'
# This is me trying to roughly approximate the angular range and resolution of the polar nephelometer
# import mie theory 600nm data
MT_600nm_S1 = pd.read_csv(MieTheory_PSL600nm_S1_Perpendicular, sep=',')
intensities = np.asarray(MT_600nm_S1['S1^2'])
angles = np.asarray(MT_600nm_S1['angles'])
# had to run a pchip interpolator to increase resolution of Mie Theory SD and give same number of points as
# in the experiment measurement of the SD
intensities_pchip_math_function = PchipInterpolator(angles, intensities, axis=0)
angles_pchip = np.linspace(0, 180, num=1050-250, endpoint=True)
intensities_pchip = intensities_pchip_math_function(angles_pchip)

# divide the mie theory data into three sections
idx_a, val_a = find_nearest(angles_pchip, 75)
print(idx_a, val_a)

angles_sect_a = angles_pchip[0:idx_a]
intensities_sect_a = intensities_pchip[0:idx_a]
index_max_a = np.argmax(intensities_sect_a)
max_a = intensities_sect_a[index_max_a]

idx_b, val_b = find_nearest(angles_pchip, 150)
print(idx_b, val_b)

angles_sect_b = angles[idx_a:idx_b]
intensities_sect_b = intensities_pchip[idx_a:idx_b]
index_max_b = np.argmax(intensities_sect_b)
intensity_max_b = intensities_sect_b[index_max_b]
print(index_max_b, intensity_max_b)


# find nearest value in section a that closest matches the max in section b
# the "goodness" of this method is dependent on the resolution of the theory
idx_c, val_c = find_nearest(intensities_sect_a, intensity_max_b)
print(idx_c, val_c)
idx_d, val_d = find_nearest(intensities_sect_b, intensity_max_b)
print(idx_d, val_d)
print(intensities_sect_a[idx_c], intensities_sect_b[idx_d])

# make a fourth section from the Mie Theory data that shows where the intensities match in the
# mie theory data over a section of angles
angles_sect_d = angles_pchip[idx_c:idx_d + idx_a]
intensities_sect_d = intensities_pchip[idx_c:idx_d + idx_a]
plt.plot(angles_sect_d, intensities_sect_d)
plt.yscale('log')
plt.title('Section of the Scattering Diagram Calculated via Mie Theory\n Where the Intensities of the Endpoints of the Curve Roughly Match ')
plt.ylabel('Intensity')
plt.xlabel('Scattering Angle $\Theta$')
plt.xticks(np.arange(angles_pchip[idx_c], angles_pchip[idx_d + idx_a], 2))
plt.grid()
plt.tight_layout()
plt.show()
#'''
#'''
# we need to do the same thing to the experimental data, find where
# two points that are the same intensity in the scattering diagram like we did for the mie theory

EXP_PSL600nm_S1 = pd.read_csv(EXP_PSL600nm_S1_Path_Perpendicular, sep=',')
EXP_PSL600nm_S1_intensities = np.asarray(EXP_PSL600nm_S1['Intensity'])
EXP_PSL600nm_S1_profile_num = np.asarray(EXP_PSL600nm_S1['Columns'])

# divide the mie theory data into three sections
idx_a2, val_a2 = find_nearest(EXP_PSL600nm_S1_profile_num, 590)
print(idx_a2, val_a2)

profile_num_sect_a2 = EXP_PSL600nm_S1_profile_num[0:idx_a2]
intensities_sect_a2 = EXP_PSL600nm_S1_intensities[0:idx_a2]
index_max_a2 = np.argmax(intensities_sect_a2)
max_a2 = intensities_sect_a2[index_max_a2]

idx_b2, val_b2 = find_nearest(EXP_PSL600nm_S1_profile_num, 920)
print(idx_b2, val_b2)

profile_num_sect_b2 = EXP_PSL600nm_S1_profile_num[idx_a2:idx_b2]
intensities_sect_b2 = EXP_PSL600nm_S1_intensities[idx_a2:idx_b2]
index_max_b2 = np.argmax(intensities_sect_b2)
intensity_max_b2 = intensities_sect_b2[index_max_b2]
print(index_max_b2, intensity_max_b2)


# find nearest value in section a that closest matches the max in section b
# the "goodness" of this method is dependent on the resolution of the theory
idx_c2, val_c2 = find_nearest(intensities_sect_a2, intensity_max_b2)
print(idx_c2, val_c2)
idx_d2, val_d2 = find_nearest(intensities_sect_b2, intensity_max_b2)
print(idx_d2, val_d2)
print(intensities_sect_a2[idx_c2], intensities_sect_b2[idx_d2])

# make a fourth section from the Mie Theory data that shows where the intensities match in the
# mie theory data over a section of angles
profile_num_sect_d2 = EXP_PSL600nm_S1_profile_num[idx_c2:idx_d2 + idx_a2]
intensities_sect_d2 = EXP_PSL600nm_S1_intensities[idx_c2:idx_d2 + idx_a2]
plt.plot(profile_num_sect_d2, intensities_sect_d2)
plt.yscale('log')
plt.title('Section of the Scattering Diagram Calculated via Mie Theory\n Where the Intensities of the Endpoints of the Curve Roughly Match ')
plt.ylabel('Intensity')
plt.xlabel('Profile Number')
plt.xticks(np.arange(EXP_PSL600nm_S1_profile_num[idx_c2], EXP_PSL600nm_S1_profile_num[idx_d2 + idx_a2], 10))
plt.grid()
plt.tight_layout()
plt.show()
#'''
#'''
#angles_sect_d2 = np.linspace(angles_sect_d[0], angles_sect_d[len(angles_sect_d)-1], len(profile_num_sect_d2))
m, b = np.polyfit(profile_num_sect_d2[0:210-12], angles_sect_d, 1)
fit = np.polyfit(profile_num_sect_d2[0:210-12], angles_sect_d, 1)
fit_fn = np.poly1d(fit)
fit_string = 'y = ' + str('{:.3e}'.format(m)) + 'x + ' + str('{:.3e}'.format(b))
plt.plot(profile_num_sect_d2[0:210-12], angles_sect_d, 'bo', label='Data')
plt.plot(profile_num_sect_d2[0:210-12], fit_fn(profile_num_sect_d2[0:210-12]), 'r-', label='Fit')
plt.title('Pixel to Angle Transformation over the Scattering Angles 62.2\u00B0 to 106.2\u00B0')
plt.ylabel('Scattering Angle $\Theta$')
plt.xlabel('Profile Number')
plt.legend(loc=1)
plt.text(650, 80, s=fit_string, fontsize=12)
plt.grid()
plt.show()

#transform 600nm parallel profile numbers to angles
EXP_PSL600nm_S2 = pd.read_csv(EXP_PSL600nm_S2_Path_Parallel, sep=',')
MT_PSL600nm_S2 = pd.read_csv(MieTheory_PSL600nm_S2_Parallel, sep=',')

EXP_PSL600nm_S2_profile_num = EXP_PSL600nm_S2['Columns']
EXP_PSL600nm_S2_intensities = EXP_PSL600nm_S2['Intensity']

EXP_PSL600nm_S2_inty_norm_factor = np.linalg.norm(EXP_PSL600nm_S2_intensities, ord=np.inf)
EXP_PSL600nm_S2_intensities_norm = EXP_PSL600nm_S2_intensities/EXP_PSL600nm_S2_inty_norm_factor
angles_EXP_PSL600nm_S2_Parallel = [int(round((m * x) + b)) for x in EXP_PSL600nm_S2_profile_num]

MT_PSL600nm_S2_inty_norm_factor = np.linalg.norm(MT_PSL600nm_S2['S2^2'], ord=np.inf)

PSL600nm_S2_Data = pd.DataFrame()
PSL600nm_S2_Data['Profile Numbers'] = EXP_PSL600nm_S2_profile_num
PSL600nm_S2_Data['Angles'] = angles_EXP_PSL600nm_S2_Parallel
PSL600nm_S2_Data['Intensities'] = EXP_PSL600nm_S2_intensities
PSL600nm_S2_Data['Normalized Intensities'] = EXP_PSL600nm_S2_intensities_norm
PSL600nm_S2_Data.to_csv('/home/austen/Documents/Compare_Data/PSL600nm_S2_Data.txt', sep=',')

'''
plt.plot(angles_EXP_PSL600nm_S2_Parallel, EXP_PSL600nm_S2_intensities, 'b-', label='600nm PSL Experiment Data')
plt.plot(MT_PSL600nm_S2['angles'], MT_PSL600nm_S2['S2^2'], 'r-', label='600nm PSL Mie Theory')
plt.title('600nm PSL Particle Scattering Diagram of 663nm Light\n Polarized Parallel to the Scattering Plane (S2)')
plt.xlabel('Scattering Angle $\Theta$')
plt.ylabel('Intensity (DN)')
plt.yscale('log')
plt.legend(loc=1)
plt.grid()
plt.show()
'''

plt.plot(angles_EXP_PSL600nm_S2_Parallel, EXP_PSL600nm_S2_intensities_norm, 'b-', label='600nm PSL Experiment Data')
plt.plot(MT_PSL600nm_S2['angles'], MT_PSL600nm_S2['S2^2']/MT_PSL600nm_S2_inty_norm_factor, 'r-', label='600nm PSL Mie Theory')
plt.title('600nm PSL Particle Scattering Diagram of 663nm Light\n Polarized Parallel to the Scattering Plane (S2)')
plt.xlabel('Scattering Angle $\Theta$')
plt.ylabel('Intensity (DN)')
plt.yscale('log')
plt.legend(loc=1)
plt.grid()
plt.show()

#transform 701nm parallel profile numbers to angles
EXP_PSL701nm_S2 = pd.read_csv(EXP_PSL701nm_S2_Path_Parallel, sep=',')
MT_PSL701nm_S2 = pd.read_csv(MieTheory_PSL701nm_S2_Parallel, sep=',')

EXP_PSL701nm_S2_profile_num = EXP_PSL701nm_S2['Columns']
EXP_PSL701nm_S2_intensities = EXP_PSL701nm_S2['Intensity']

EXP_PSL701nm_S2_inty_norm_factor = np.linalg.norm(EXP_PSL701nm_S2_intensities, ord=np.inf)
EXP_PSL701nm_S2_intensities_norm = EXP_PSL701nm_S2_intensities/EXP_PSL701nm_S2_inty_norm_factor
angles_EXP_PSL701nm_S2_Parallel = [int(round((m * x) + b)) for x in EXP_PSL701nm_S2_profile_num]

MT_PSL701nm_S2_inty_norm_factor = np.linalg.norm(MT_PSL701nm_S2['S2^2'], ord=np.inf)

PSL701nm_S2_Data = pd.DataFrame()
PSL701nm_S2_Data['Profile Numbers'] = EXP_PSL701nm_S2_profile_num
PSL701nm_S2_Data['Angles'] = angles_EXP_PSL701nm_S2_Parallel
PSL701nm_S2_Data['Intensities'] = EXP_PSL701nm_S2_intensities
PSL701nm_S2_Data['Normalized Intensities'] = EXP_PSL701nm_S2_intensities_norm
PSL701nm_S2_Data.to_csv('/home/austen/Documents/Compare_Data/PSL701nm_S2_Data.txt', sep=',')

'''
plt.plot(angles_EXP_PSL701nm_S2_Parallel, EXP_PSL701nm_S2_intensities, 'b-', label='701nm PSL Experiment Data ')
plt.plot(MT_PSL701nm_S2['angles'], MT_PSL701nm_S2['S2^2'], 'r-', label='701nm PSL Mie Theory')
plt.title('701nm PSL Particle Scattering Diagram of 663nm Light\n Polarized Parallel to the Scattering Plane (S2)')
plt.xlabel('Scattering Angle $\Theta$')
plt.ylabel('Intensity (DN)')
plt.yscale('log')
plt.legend(loc=1)
plt.grid()
plt.show()
'''

plt.plot(angles_EXP_PSL701nm_S2_Parallel, EXP_PSL701nm_S2_intensities_norm, 'b-', label='701nm PSL Experiment Data ')
plt.plot(MT_PSL701nm_S2['angles'], MT_PSL701nm_S2['S2^2']/MT_PSL701nm_S2_inty_norm_factor, 'r-', label='701nm PSL Mie Theory')
plt.title('701nm PSL Particle Scattering Diagram of 663nm Light\n Polarized Parallel to the Scattering Plane (S2)')
plt.xlabel('Scattering Angle $\Theta$')
plt.ylabel('Intensity (DN)')
plt.yscale('log')
plt.legend(loc=1)
plt.grid()
plt.show()
