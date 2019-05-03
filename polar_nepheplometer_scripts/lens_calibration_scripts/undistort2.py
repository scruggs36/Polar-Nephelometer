'''
Austen K. Scruggs
10-25-2018
Description: Undistorts images due to the use of a fish eye lens. in undistort.py we noticed
that the dimensions of the undistorted images were different than those of the original image
this is an attempt to explore why that is, and if we can get everything to be the original image dimensions.
I beleve this will have to do with the balance parameter, a balance = 0 cuts out pixels and keeps the best part of the image.
a balance = 1 keeps all the pixels but you see black hills at the edges of the image.

I DID NOT USE THIS!!!! WE PULLED THIS FROM PART 2 OF https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
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
Cal_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Lens Calibration/10-24-2018/Calibration Images'
Image_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Lens Calibration/10-24-2018/Calibration Images'
Save_Dir = '/home/austen/Documents'

# lens calibration
N_OK, dim, k, d = lens_calibration(Cal_Dir)

# You should replace these 3 lines with the output in calibration step
DIM=dim
K=np.array(k)
D=np.array(d)


def undistort(img_path, balance=0.0, dim2=None, dim3=None):

    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

    if not dim2:
        dim2 = dim1

    if not dim3:
        dim3 = dim1

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    for p in sys.argv[1:]:
        undistort(p)