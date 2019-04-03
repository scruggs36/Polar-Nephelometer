from Neph_Functions import Set_Beam_Edges
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd
import numpy as np



# directory navigation i.e. path to image
Path = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/06-26-2018/HW_Analysis/150_Degrees/im_avg_N2_30s_HW_150Degrees.png'

# scipy.misc package mi.imread reads the BMP as an image format
im = cv2.imread(Path, 0)

# Show Images
plt.imshow(im, cmap='gray')
plt.show()

# Calling function I made early on in the project
Y_coords, X_coords, column_profiles, Intensity_Matrix, Scattering_Diagram = Set_Beam_Edges(Image=im, top_point1=(400, 250), top_point2=(400, 1050), bot_point1=(500, 250), bot_point2=(500, 1050), num=1050-250)
#print(column_profiles)
#print(Scattering_Diagram)


f, ax = plt.subplots()
ax.plot(column_profiles, Scattering_Diagram, 'b-', label='Scattering Diagram')
ax.set_title('ROI Scattering Diagram as \n a Function of Profile Number', fontsize=36)
ax.set_ylabel('Intensity', fontsize=18)
ax.set_xlabel('Profile Number', fontsize=18)
ax.legend(loc=1)
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()

SD = pd.DataFrame()
SD['Columns'] = column_profiles
SD['Intensity'] = Scattering_Diagram
SD.to_csv('/home/austen/Documents/SD_N2.txt')
