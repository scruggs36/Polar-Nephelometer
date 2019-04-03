import PIL
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# network file path
Path_ImDir = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/04-16-2018/N2'
Path_Save = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/04-16-2018/Analysis_N2/Sum'
FN_1 = '/im_sum_N2_60s_coord.png'
FN_2 = '/im_sum_N2_60s.png'

# list files in directory
file_list = os.listdir(Path_ImDir)
# number of files in directory
num_files = len(file_list)
# loop average of images (cannot sum due to the fact that everything is 8 bit, averaging mitigates overflow)
# the 0 in imread means its grayscale, a 1 would mean color, and a -1 would mean unchanged
for counter, fn in enumerate(file_list):
    if counter == 0 and fn != 'Thumbs.db':
        A = cv2.imread(Path_ImDir+'/'+str(fn), 0)
        A = A.astype('int')
    if counter > 0 and fn != 'Thumbs.db':
        print(fn)
        B = cv2.imread(Path_ImDir + '/' + str(fn), 0)
        B = B.astype('int')
        C = A.astype('int') + B.astype('int')
        A = C

# creating different types of the summed array
im_summed = A
#im_summed_8bit = im_summed.astype('uint8')
im_summed_int = im_summed.astype('int')

# plot summed image
plt.imshow(im_summed_int, cmap='inferno')
plt.colorbar()
plt.show()
#plt.savefig(Path_Save + FN_1)

# plot histogram of summed image
histogram, bin_edges = np.histogram(im_summed_int, bins=np.arange(0, im_summed_int.flatten().max(), 10))
plt.hist(histogram, bin_edges)
plt.title('Image Histogram')
plt.xlabel('Summed Digital Number Bins')
plt.ylabel('$Log_{10}(Counts)$')
plt.yscale('log')
plt.xscale('linear')
plt.grid(True)
plt.show()

# writes image to new file
cv2.imwrite(Path_Save + FN_2, im_summed_int)




