from Neph_Functions import Image_Subtract, Image_Heatmaps, Profiles, Curve_Fit_Profiles, gaussian, quadratic
import pandas as pd
from scipy.optimize import curve_fit
from skimage.exposure import histogram
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import math


# Initial boundaries on the image
col_boundary_1 = 0
col_boundary_2 = 800
row_boundary_1 = 400
row_boundary_2 = 600
numrows = row_boundary_2 - row_boundary_1
numcols = 100 #col_boundary_2 - col_boundary_1

# Looping evaluation
# network file path
Path_bkg = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/data/03-14-2018/N2/5s/Analysis/im_avg_N2_5s.png'
Path = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/data/03-14-2018/PSL_701nm/5s/Images'
Path_PD = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/data/03-08-2018/PSL_701nm/Analysis'
if not os.path.exists(Path_PD):
    os.makedirs(Path_PD)
# list files in directory
file_list = os.listdir(Path)
# number of files in directory
num_files = len(file_list)
print(num_files)
# loop average of images (cannot sum due to the fact that everything is 8 bit, averaging mitigates overflow)
# the L in Image.fromarray means its grayscale
B = cv2.imread(Path_bkg, 0)
plt.imshow(B, cmap='gray')
plt.show()
#B = B.astype('int')
#B8 = B.astype('uint8')
for counter, fn in enumerate(file_list):
    if counter >= 0 and fn != 'Thumbs.db':
        print(str(fn))
        A = cv2.imread(Path + '/' + str(fn), 0)
        #A = A.astype('int')
        C = Image_Subtract(A, B)
        DI = C.astype('uint8')

        # creates directory for difference images and saves them to it
        if not os.path.exists(Path_PD + '/Images_Bkg_Sub'):
            os.makedirs(Path_PD + '/Images_Bkg_Sub')
        cv2.imwrite(Path_PD + '/Images_Bkg_Sub/' + 'diff_' + str(counter) + '.png', DI)

        # creates directory for heatmaps and saves them to it
        if not os.path.exists(Path_PD + '/Heatmaps'):
            os.makedirs(Path_PD + '/Heatmaps')
        Image_Heatmaps(B, DI, 255, Path_PD + '/Heatmaps/HM_' + str(counter) + '.png')
        rows_DF, cols_DF, int_DF = Profiles(DI, col_boundary_1, col_boundary_2, numrows, numcols, row_boundary_1, row_boundary_2)
        fitxdata1, fitydata1, fyd1, ft, bt, mt, st, bkgt, aug_array, aug_array_norm, tt_smooth, tb_smooth = Curve_Fit_Profiles(gaussian, quadratic, rows_DF, int_DF, cols_DF)

        # here we are plotting the fit data
        f1, ax1 = plt.subplots(1, 2)
        ax1[0].imshow(DI, cmap='gray')
        ax1[0].plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
        ax1[0].plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
        ax1[0].plot(cols_DF, mt, 'rX', ms=0.4)
        ax1[1].set_title('Fits to Top Profiles')
        ax1[1].set_xlabel('Y Coordinate')
        ax1[1].set_ylabel('Intensity')

        # creates directory for plots and saves them to it
        for counter1, row in enumerate(fitxdata1):
            ax1[1].plot(fitxdata1[counter1], fyd1[counter1])
        if not os.path.exists(Path_PD + '/Fits'):
            os.makedirs(Path_PD + '/Fits')
        plt.savefig(Path_PD + '/Fits/' + 'Fits_' + str(counter) + '.png')
        plt.close()

        # plotting data
        f1a, ax1a = plt.subplots()
        ax1a.imshow(DI, cmap='gray')
        ax1a.plot(cols_DF, tt_smooth, color='lime', marker='.', ms=0.2)
        ax1a.plot(cols_DF, tb_smooth, color='orange', marker='.', ms=0.2)
        ax1a.plot(cols_DF, mt, 'rX', ms=0.4)

        # creates directory for plots and saves them to it
        if not os.path.exists(Path_PD + '/Bounds'):
            os.makedirs(Path_PD + '/Bounds')
        plt.savefig(Path_PD + '/Bounds/' + 'Bounds_' + str(counter) + '.png')
        plt.close()

        # plotting data
        f2, ax2 = plt.subplots()
        ax2.plot(cols_DF, aug_array_norm, 'r-', label='Phase Function')
        ax2.set_title('Integrated Intensity as a Function of Vertical Profile Number')
        ax2.set_xlabel('Column Profile Number')
        ax2.set_ylabel('Integrated Intensity')
        plt.legend(loc=1)

        # creates directory for plots and saves them to it
        if not os.path.exists(Path_PD + '/PF'):
            os.makedirs(Path_PD + '/PF')
        if not os.path.exists(Path_PD + '/PF/PF_Normalized'):
            os.makedirs(Path_PD + '/PF/PF_Normalized')
        plt.savefig(Path_PD + '/PF/PF_Normalized/' + 'PF_Norm_' + str(counter) + '.png')
        plt.close()

        # plotting data
        f3, ax3 = plt.subplots()
        ax3.plot(cols_DF, aug_array, 'b-', label='Phase Function')
        ax3.set_title('Integrated Intensity as a Function of Vertical Profile Number')
        ax3.set_xlabel('Column Profile Number')
        ax3.set_ylabel('Integrated Intensity')
        plt.legend(loc=1)

        # creates directory for plots and saves them to it
        if not os.path.exists(Path_PD + '/PF/PF_Not_Normalized'):
            os.makedirs(Path_PD + '/PF/PF_Not_Normalized')
        plt.savefig(Path_PD + '/PF/PF_Not_Normalized/' + 'PF_NN_' + str(counter) + '.png')
        plt.close()

        # creates phase function data fram
        PhaseFunctionDF = pd.DataFrame()
        PhaseFunctionDF['Profile Number'] = cols_DF
        PhaseFunctionDF['Intensity'] = aug_array
        PhaseFunctionDF['Normalized Intensity'] = aug_array_norm

        # creates directory for PF data and saves them to it
        if not os.path.exists(Path_PD + '/PF/PF_Data'):
            os.makedirs(Path_PD + '/PF/PF_Data')
        PhaseFunctionDF.to_csv(Path_PD + '/PF/PF_Data/' + 'PF_' + str(counter) + '.txt')


