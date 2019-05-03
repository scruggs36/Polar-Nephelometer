'''
Austen K. Scruggs
10-23-2018
Description: This script analyzes checkerboard calibration images taken using a fisheye lens and finds parameters K and D.
Parameters K and D are used to undistort other images taken with the fisheye lens.
https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
'''

# imported packages
import cv2
print('opencv version: ', cv2.__version__)
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

# directory where the calibrated images are
Cal_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Lens Calibration/10-24-2018/Calibration Images/jpg images'
Save_Dir = '/home/austen/Documents/Lens_Calibration_Corrections'

def lens_calibration(image_directory):
    # checkerboard printed page dimensions
    CHECKERBOARD = (6, 9)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    # this can be changed to BMP or jpg, whatever the image format is!

    images = glob.glob(image_directory + '/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

         # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs, calibration_flags, (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return [N_OK, _img_shape[::-1], K, D]


# lens calibration
N_OK, dim, k, d = lens_calibration(Cal_Dir)

# You should replace these 3 lines with the output in calibration step
DIM_0=dim
K_0=np.array(k)
D_0=np.array(d)

def undistort(img_path, save_path, DIM, K, D):
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


undistort(Cal_Dir, Save_Dir, DIM_0, K_0, D_0)
