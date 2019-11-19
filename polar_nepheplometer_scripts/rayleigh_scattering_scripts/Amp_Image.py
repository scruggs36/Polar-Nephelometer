'''
Austen K. Scruggs
11/01/2019
Description:
Amplify image signal and background to reveal problems in the background
'''

import matplotlib.pyplot as plt
import pandas as pd

Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/2019/2019-10-25/N2/150s/N2_150s_0lamda_0_AVG_.txt'
im = pd.read_csv(Path_N2_Dir)

fcal, axcal = plt.subplots(figsize=(12, 12))
im_fcala = axcal.pcolormesh(im, cmap='gray')
divider_cala = make_axes_locatable(axcal)
cax_cala = divider_cala.append_axes("right", size="5%", pad=0.05)
fcal.colorbar(im_fcala, cax=cax_cala)
axcal.set_title('Carbon Dioxide Rayleigh Scattering')