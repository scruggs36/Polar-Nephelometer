import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc as mi
import cv2
import numpy as np
import pandas as pd
from scipy.integrate import simps as simpson
from scipy import interpolate
from scipy.optimize import curve_fit
from skimage import feature
from skimage import exposure


#reading images into python
#Path_BKG = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/11-20-2017/Image15s_bkg.BMP'
#Path_N2 = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/10-03-2017/10mW_30sExposure/Nigrosin/Image30s_Nigrosin_Stagnant.BMP'
#Path_HE = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/10-02-2017/bkg/Image60s_bkg_Air.BMP'
Path_PARTICLES = '/home/austen/PycharmProjects/Polar-Nephelometer/experiment/Data/04-04-2018/PSL_600nm/Analysis_30s/Average/im_avg_PSL600nm_N2sub.png'
#im_BKG = mi.imread(Path_BKG)
#im_N2 = mi.imread(Path_N2)
#im_HE = mi.imread(Path_HE)
im_PARTICLES = cv2.imread(Path_PARTICLES, 0)
#cv2.imshow('preview', im_PARTICLES)
#cv2.waitKey()

'''
#Background Heatmap
histBKG, bins_centerBKG = exposure.histogram(im_BKG, nbins=256)
f0, (ax0, ax1) = plt.subplots(ncols=2)
heatmap = ax0.pcolormesh(np.flip(im_BKG, 0), cmap=cm.afmhot, vmax=255)
ax0.set_title('Heatmap')
ax0.set_ylabel('Y Pixels')
ax0.set_xlabel('X Pixels')
f0.colorbar(heatmap, ax=ax0)
ax1.bar(bins_centerBKG, histBKG)
ax1.set_title('Histogram: Image Pixel Distribution Amongst Intensity Bins')
ax1.set_ylabel('Counts')
ax1.set_xlabel('Bit Number (0-255)')
ax1.set_xlim(0,20)
#plt.tight_layout()
plt.show()
'''
width, height = im_PARTICLES.shape
print([width, height])
f1, ax = plt.subplots(1, 2)
ax[0].imshow(im_PARTICLES, cmap='gray')
ax[0].set_ylabel('Y Pixels')
ax[0].set_xlabel('X Pixels')
ax[0].set_title('Polar Nephelometer Image')

def Profiles(Image):
    TopRowsDF = []
    TopColsDF = []
    TopIntDF = []
    rows1 = np.linspace(300, 600, 300)
    rows1int = rows1.astype(int)
    columns = np.arange(0,1280, 1)
    #columns = np.linspace(0, 1279, 1280)
    columns1int = columns.astype(np.int)
    for i in columns1int:
        z1 = Image[rows1int, np.full(len(rows1int), np.int(i))]
        ax[0].plot([i, i], [rows1int[0], rows1int[-1]], 'ro-')
        ax[1].plot(z1, label=str(i))
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        TopRowsDF.append(rows1int)
        TopColsDF.append(i)
        TopIntDF.append(z1)
    return [TopRowsDF, TopColsDF, TopIntDF]

ROWS_DF, COLS_DF, INT_DF = Profiles(im_PARTICLES)

ax[1].set_title('Vertical Tansect Profiles')
ax[1].set_ylabel('Intensity')
ax[1].set_xlabel('Pixel Distance Along Image Y Axis')
ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='x-small')
plt.tight_layout()
plt.show()


def riemann(startinterval, stopinterval, y, dx):
    x = startinterval
    n = (stopinterval-startinterval)/dx
    n = int(n)
    s = 0.0
    for i in range(n-1):
        f_i = y[x]
        s += f_i
        x += dx
    return s*dx


def loopintegration( d_cols, d_rows, y):
    # data needed to be a column array
    # data needed to be column array, this needs fixing in labview
    # data needed to be column arrays here as well, this needs fixing in labview
    # create plots
    #acquire the number of columns in the dataframe for use later
    riemannarray = []
    simpsonarray = []
    profilecoordsarray = []
    print(y[0])
    for counter, row in enumerate(y):
        riemannval = riemann(0, len(y[counter]), y[counter], 1)
        simpsonval = simpson(y[counter], np.arange(len(y[counter])))
        riemannarray.append(riemannval)
        simpsonarray.append(simpsonval)
    for element in d_cols:
        profilecoords =  element
        print(profilecoords)
        profilecoordsarray.append(profilecoords)
    return [profilecoordsarray, riemannarray, simpsonarray]

profilescolslist, riemann, simpson = loopintegration(COLS_DF, ROWS_DF, INT_DF)
ProfNum = profilescolslist

f4, ax4 = plt.subplots()
ax4.plot(ProfNum, riemann, 'r-', label='Riemann Sum Top Beam')
ax4.plot(ProfNum, simpson, 'b-', label='Simpson Method Top Beam')
# i had to reverse the array ([::-1]) because the bottom beam is read from right to left, the top beam from left to right
# this is because the bottom beam at the right is close to 90 degrees and the far left is close to 180 degrees
#ax4.plot(BotProfNumberList, riemann_B[::-1], 'b--',label='Riemann Sum Bottom Beam')
#ax4.plot(BotProfNumberList, simpson_B[::-1], 'b*', label='Simpson Method Bottom Beam')
ax4.set_title('Integrated Intensity as a Function of the Number of the Profile Evaluated')
ax4.set_xlabel('Profile Number Along Image X Axis')
ax4.set_ylabel('Integrated Intensity Along Profile')
plt.legend(loc=1)
plt.show()
