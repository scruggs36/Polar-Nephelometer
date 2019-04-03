'''
Author: Austen K. Scruggs
Date: 05-01-2018
Description: This script is used to average Images together,
however, it is now obsolete and must not be used anymore, the for this
is that labview creates summary files which are to be evaluated! even for
Nitrogen and Helium scattering!
'''

import cv2
import os
# so within pyplot Tkinter was changed to tkinter, so I had to sudo apt-get install python3-tk
import matplotlib.pyplot as plt

# network file path
Path = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/08-13-2018/N2/20s/Images'
Path_Save = '/home/austen/Documents'
FN_1 = '/im_avg_N2_20s_coord.png'
FN_2 = '/im_avg_N2_20s.png'

# list files in directory
file_list = os.listdir(Path)
#print(file_list)
# number of files in directory
num_files = len(file_list)
# loop average of images (cannot sum due to the fact that everything is 8 bit, averaging mitigates overflow)
# the 0 in imread means its grayscale, a 1 would mean color, and a -1 would mean unchanged
for counter, fn in enumerate(file_list):
    if counter == 0 and fn != 'Thumbs.db':
        A = cv2.imread(Path+'/'+str(fn), 0)
        A = A.astype('float')
    if counter > 0 and fn != 'Thumbs.db':
        print(fn)
        B = cv2.imread(Path+'/'+str(fn), 0)
        C = A.astype('float') + B.astype('float')
        A = C
im_summed = A/num_files
im_summed = im_summed.astype('uint8')
plt.imshow(im_summed, cmap='gray')
plt.savefig(Path_Save + FN_1)
plt.colorbar()
#plt.show()
# writes image to new file
cv2.imwrite(Path_Save + FN_2, im_summed)