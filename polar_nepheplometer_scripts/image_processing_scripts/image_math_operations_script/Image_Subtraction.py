from Neph_Functions import Image_Subtract
import matplotlib.pyplot as plt
import cv2
import numpy as np

# note try to only use cv2 and numpy as it allows for use of numpy analysis/ indexing

#subtract two images setting images as variables
Image_1_Path = '/home/austen/media/winshare/Groups/Smith_G/austen/Projects/Nephelometry/Polar Nephelometer/Data/05-11-2018/N2_Analysis/im_avg_N2_10s.png'
Image_2_Path = '/home/austen/media/winshare/Groups/Smith_G/austen/Projects/Nephelometry/Polar Nephelometer/Data/05-11-2018/He_Analysis/im_avg_He_10s.png'
Image_1 = cv2.imread(Image_1_Path, 0)
#plt.imshow(Image_1, cmap='gray')
#plt.show()
Image_2 = cv2.imread(Image_2_Path, 0)
#plt.imshow(Image_2, cmap='gray')
#plt.show()

#calling the function and showing the resultant subtracted image
Diff_Image = Image_Subtract(Image_1, Image_2)
plt.imshow(Diff_Image, cmap='gray')
plt.show()

#save particle scattering image (particle scattering = total scattering - rayleigh scattering (n2 or He))
cv2.imwrite('/home/austen/Documents/im_avg_N2subHe.png', Diff_Image)
