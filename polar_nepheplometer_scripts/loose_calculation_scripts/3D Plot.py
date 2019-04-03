'''
Austen K. Scruggs
12-18-2018
Description 3D Plot example using real data
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Save_Directory = '/home/austen/Documents/'

Exposure_Time = [1, 3, 6, 9, 12, 15]
Slopes = [0.2085, 0.2076, 0.2052, 0.2074, .2179, 0.2232]
Intercepts = [-45.9408, -45.9737, -44.3922, -45.972, -52.0609, -54.8141]


fig0, ax0 = plt.subplots(1, 2, figsize=(10, 6))
ax0[0].plot(Exposure_Time, Intercepts, color='blue', ls='-', label='Intercepts vs Exposure Time')
ax0[0].set_xlabel('Exposure Time')
ax0[0].set_ylabel('Y Intercept (\u00b0)')
ax0[0].set_title('Calibration Intercept vs. Exposure Time')
ax0[0].legend(loc=1)
ax0[0].grid(True)
ax0[1].plot(Exposure_Time, Slopes, color='green', ls='-', label='Slopes vs Exposure Time')
ax0[1].set_xlabel('Exposure Time')
ax0[1].set_ylabel('Slope (\u00b0/PN)')
ax0[1].set_title('Calibration Slope vs. Exposure Time')
ax0[1].legend(loc=1)
ax0[1].grid(True)
plt.tight_layout()
plt.savefig(Save_Directory + 'Cal Params vs Exposure Time.pdf', format='pdf')
plt.show()

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
ax1.plot_surface(Exposure_Time, Intercepts, Slopes, label='Time vs. Slopes vs. Intercepts', cmap='inferno')
ax1.set_xlabel('Exposure Time')
ax1.set_ylabel('Intercepts')
ax1.set_zlabel('Slopes')
ax1.set_title('900nm PSL Calibration Slopes and Intercepts as a Function of Exposure Time')
ax1.grid(True)
plt.savefig(Save_Directory + 'Exposure Time vs Intercepts vs Slopes.pdf', format='pdf')
plt.show()

