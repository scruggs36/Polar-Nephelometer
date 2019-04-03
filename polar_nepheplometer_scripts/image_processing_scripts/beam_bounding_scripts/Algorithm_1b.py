from Neph_Functions import Loop_Image_Average, Set_Beam_Edges
import matplotlib.pyplot as plt
#import cv2
#import math
import pandas as pd
#import numpy as np



# directory navigation i.e. path to image
Path_ParticleDir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/08-13-2018/PSL_600nm/45s/Images'
Path_MediumImage = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/08-13-2018/N2/Data_Analysis/im_avg_N2_45s.png'
Path_SD = '/home/austen/Documents/PSL600nm_SD_45s.txt'

# averaging images
im_Avg = Loop_Image_Average(Path_ParticleDir, Path_MediumImage)
plt.imshow(im_Avg, cmap='gray')
plt.show()

top = 480
bot = 650
left = 240
right = 1060

# Calling function I made early on in the project
Y_coords, X_coords, column_profiles, Intensity_Matrix, Scattering_Diagram = Set_Beam_Edges(Image=im_Avg, top_point1=(top, left), top_point2=(top, right), bot_point1=(bot, left), bot_point2=(bot, right), num=right-left)
#print(column_profiles)
#print(Scattering_Diagram)


f, ax = plt.subplots()
#ax.plot(column_profiles, Scattering_Diagram[::-1], 'b-', label='Scattering Diagram')
ax.plot(column_profiles, Scattering_Diagram, 'b-', label='Scattering Diagram')
ax.set_title('ROI Scattering Diagram as \n a Function of Profile Number', fontsize=36)
ax.set_ylabel('Intensity', fontsize=18)
ax.set_xlabel('Profile Number', fontsize=18)
ax.legend(loc=1)
plt.grid()
plt.tight_layout()
plt.yscale('log')
plt.show()

SD = pd.DataFrame()
SD['Columns'] = column_profiles
#SD['Intensity'] = Scattering_Diagram[::-1]
SD['Intensity'] = Scattering_Diagram
SD.to_csv(Path_SD)
