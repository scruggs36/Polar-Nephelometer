'''
Austen K. Scruggs
10-24-2018
Description: Undistorts images due to the use of a fish eye lens.
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
'''

# imported packages
from calibration import lens_calibration
import cv2
assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import glob

# directory holding calibration images
Cal_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Lens Calibration/10-24-2018/Calibration Images/bmp images'
Image_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Lens Calibration/10-24-2018/Calibration Images/bmp images'
Save_Dir = '/home/austen/Documents'

# lens calibration
N_OK, dim, k, d = lens_calibration(Cal_Dir)

# You should replace these 3 lines with the output in calibration step
DIM=dim
K=np.array(k)
D=np.array(d)

def undistort(img_path, save_path):
    file_list = os.listdir(img_path)
    print(file_list)
    fnum = len(file_list)
    images = glob.glob(img_path + '/*.BMP')
    for counter, fname in enumerate(images):
        print(fname)
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        cv2.imwrite(save_path + '/undistorted_' + str(file_list[counter]), undistorted_img)
        plt.imshow(undistorted_img)
        #plt.savefig(save_path + '/undistorted_matplotlib_' + str(file_list[counter]) + '_.pdf', format='pdf')
        plt.show()

    if __name__ == '__main__':
        for p in sys.argv[1:]:
            undistort(p)


undistort(Image_Dir, Save_Dir)
