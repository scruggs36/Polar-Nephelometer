import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import cv2
from scipy.integrate import simps as simpson
from scipy import interpolate
from scipy.optimize import curve_fit
from skimage import feature
from skimage import exposure


# directory navigation i.e. path to image
PathIndvWind = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/12-06-2017/N2/Image_BKG_SUB_2.BMP'
PathBKGIndvWind = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/data/12-06-2017/N2/Image300s_bkg.BMP'


# scipy.misc package mi.imread reads the BMP as an image format (could use cv2 as well, see version 1)
im = cv2.imread(PathIndvWind, 0)
imBKG = cv2.imread(PathBKGIndvWind, 0)


# Show Images
plt.imshow(imBKG, cmap='gray')
plt.show()
plt.imshow(im, cmap='gray')
plt.show()


# Heatmaps and histograms of both laser and background images
def Image_Heatmaps(bkg, image, bins):
    histBKG, bins_centerBKG = exposure.histogram(bkg, nbins=bins+1)
    histIM, bins_centerIM = exposure.histogram(image, nbins=bins+1)
    f0, ax0 = plt.subplots(2 ,2)
    heatmap_bkg = ax0[0, 0].pcolormesh(np.flip(bkg, 0), cmap=cm.afmhot, vmax=bins)
    heatmap_im = ax0[1, 0].pcolormesh(np.flip(image, 0), cmap=cm.afmhot, vmax=bins)
    # labels for background heatmap
    ax0[0, 0].set_title('Background Heatmap')
    ax0[0, 0].set_ylabel('Y Pixels')
    ax0[0, 0].set_xlabel('X Pixels')
    f0.colorbar(heatmap_bkg, ax=ax0[0, 0])
    # histogram for background heatmap
    ax0[0, 1].bar(bins_centerBKG, histBKG)
    ax0[0, 1].set_title('Histogram: Background Image Pixel Distribution Amongst Intensity Bins')
    ax0[0, 1].set_ylabel('Counts')
    ax0[0, 1].set_xlabel('Bit Number (0-255)')
    ax0[0, 1].set_xlim(0, bins)
    # labels for image heatmap
    ax0[1, 0].set_title('Heatmap')
    ax0[1, 0].set_ylabel('Y Pixels')
    ax0[1, 0].set_xlabel('X Pixels')
    f0.colorbar(heatmap_im, ax=ax0[1, 0])
    # histogram for image heatmap
    ax0[1, 1].bar(bins_centerIM, histIM)
    ax0[1, 1].set_title('Histogram: Image Pixel Distribution Amongst Intensity Bins')
    ax0[1, 1].set_ylabel('Counts')
    ax0[1, 1].set_xlabel('Bit Number (0-255)')
    ax0[1, 1].set_xlim(0, bins)
    plt.tight_layout()
    plt.show()
    return


Image_Heatmaps(imBKG, im, 255)


# This is a function that returns a gaussian based on the input parameters
def gaussian(x, b, c, a, d):
    return d + (abs(a) * np.exp((-(x - b)**2.00)/(2.00 * c**2.00)))


# Draws vertical transects all over the image to find the beam within the specified bounds
def Top_Profiles(Image, numberrows, numbercols, top_bound, bot_bound):
    f1, ax = plt.subplots(1, 2)
    ax[0].imshow(im, cmap='gray')
    ax[0].set_ylabel('Y Pixels')
    ax[0].set_xlabel('X Pixels')
    ax[0].set_title('Polar Nephelometer Image')
    TopRowsDF = []
    TopColsDF = []
    TopIntDF = []
    rows1 = np.linspace(top_bound, bot_bound, numberrows)
    rows1int = rows1.astype(int)
    columns = np.linspace(10, 1000, numbercols)
    columns1int = columns.astype(np.int)
    for i in columns1int:
        z1 = Image[rows1int, np.int(i)]
        ax[0].plot([i, i], [top_bound, bot_bound], 'ro-')
        ax[1].plot(z1, label=str(i))
        ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        TopRowsDF.append(rows1int)
        TopColsDF.append(i)
        TopIntDF.append(z1)
    ax[1].set_title('Top Laser Beam Line Profiles')
    ax[1].set_ylabel('Intensity')
    ax[1].set_xlabel('Pixel Distance on Image Y Axis')
    plt.tight_layout()
    plt.show()
    return [TopRowsDF, TopColsDF, TopIntDF]


# finds nearest pixel
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


# numrows sets the number of datapoints to use in the arbitrary line slicing the laser beam.
# This ultimately sets the number of points to return in the profile as intensities are acquired
# at these points are taken to yield the profile given in the plot.
top = 100
midtop = 550
midbot = 600
bot = 900
numrowstop = midtop - top
numrowsbot = bot - midbot
numcols = 100

# calling function to draw vertical profiles
TOP_ROWS_DF, TOP_COLS_DF, TOP_INT_DF = Top_Profiles(im, numrowstop, numcols, top, midtop)





gausmaxrow1array = []
gausmaxrow2array = []
SIGMA_TOP = []
SIGMA_BOT = []



# Though my image has improved this canny filter shows where the defects and light leaks exactly are!
edges = feature.canny(im, sigma=2, low_threshold=100, high_threshold=125)
hist, bins_center = exposure.histogram(im)
f2, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im, cmap='gray')
ax1[0].set_title('Single Image')
#ax1[1].imshow(edges, cmap='gray')
ax1[1].set_title('Image Histogram')
ax1[1].plot(bins_center, hist, lw=2)
#f2.tight_layout()
plt.show()



def Prof_Curve_Fit(function, rowmat, intmat):
    fitdatay = []
    fitdatax = []
    fittopbound = []
    fitbotbound = []
    fitbeammiddle = []
    sigma = []
    background = []
    fyd1 = []
    for counter, row in enumerate(rowmat):
        print(counter)
        meanx = rowmat[counter][intmat[counter].argmax()]
        popt, pcov = curve_fit(function, rowmat[counter], intmat[counter], p0=[meanx, 50.0, 0.0, 40.0], maxfev=1200)
        topbound = popt[0] - 2 * abs(popt[1])
        botbound = popt[0] + 2 * abs(popt[1])
        fitdatay.append(function(rowmat[counter], *popt))
        fitdatax.append(rowmat[counter])
        fittopbound.append(topbound)
        fitbotbound.append(botbound)
        fitbeammiddle.append(popt[0])
        sigma.append(abs(popt[1]))
        background.append(abs(popt[3]))
    # here I subtracted out the background, which is the area under the flat region of the gaussian
    for counter, array1 in enumerate(fitdatay):
        array1 = [x - background[counter] for x in array1]
        fyd1.append(array1)
    return [fitdatax, fitdatay, fyd1, fittopbound, fitbotbound, fitbeammiddle, sigma, background]

def func(x, a, b, c):
    X = np.array(x)
    return a * np.power(X, 2) + b * X + c


def Coordinate_Fit(coordinates, function):
    popt, pcov = curve_fit(function, coordinates[0], coordinates[1])
    return [popt, pcov]


# Anything fitting to a gaussian isn't going to work for you
# the gaussian fits seem to be significantly broader than the real data
# especially when the signal maximum is closer to the background values....
# thus, a new approach is needed


fitxdata1, fitydata1, fyd1, ft, bt, mt, st, bkgt = Prof_Curve_Fit(gaussian, TOP_ROWS_DF, TOP_INT_DF)
TT_Edge_Coordinates = np.vstack((TOP_COLS_DF, ft))
TB_Edge_Coordinates = np.vstack((TOP_COLS_DF, bt))
#this organizes pixel coordinates for top an bottom edge as an array of [x,y]
#TT_Edge_Coordinates_ndarray = np.column_stack((TOP_COLS_DF, ft))
#TB_Edge_Coordinates_ndarray = np.column_stack((TOP_COLS_DF, bt))
ycoordfitparams_t, ycoordfitcov_t = Coordinate_Fit(TT_Edge_Coordinates, func)
ycoordfitparams_b, ycoordfitcov_b = Coordinate_Fit(TB_Edge_Coordinates, func)
TT_smoothcoords = func(TOP_COLS_DF, *ycoordfitparams_t)
TB_smoothcoords = func(TOP_COLS_DF, *ycoordfitparams_b)

# here we are plotting the fit data
f, ax1 = plt.subplots(1, 2)
ax1[0].imshow(im, cmap='gray')
ax1[0].plot(TOP_COLS_DF, TT_smoothcoords, color='lime', marker='.', ms=0.2)
ax1[0].plot(TOP_COLS_DF, TB_smoothcoords, color='orange', marker='.', ms=0.2)
ax1[0].plot(TOP_COLS_DF, mt, 'rX', ms=0.4)
ax1[1].set_title('Fits to Top Profiles')
ax1[1].set_xlabel('Y Coordinate')
ax1[1].set_ylabel('Intensity')
for counter, row in enumerate(fitxdata1):
    ax1[1].plot(fitxdata1[counter], fyd1[counter])
plt.show()


f, ax1a = plt.subplots()
ax1a.imshow(im, cmap='gray')
ax1a.plot(TOP_COLS_DF, TT_smoothcoords, color='lime', marker='.', ms=0.2)
ax1a.plot(TOP_COLS_DF, TB_smoothcoords, color='orange', marker='.', ms=0.2)
ax1a.plot(TOP_COLS_DF, mt, 'rX', ms=0.4)
plt.show()


# lets see if we can find the tangent at every point along the curves edges
tt_coordsx = TOP_COLS_DF
tt_coordsy = TT_smoothcoords
tb_coordsx = TOP_COLS_DF
tb_coordsy = TB_smoothcoords


# num must be even number
num = 10
def Top_Transect_Profiles(Image, FRONT_TOP, BACK_TOP, TOPCOLSDF):
    TopRowsDF1 = []
    TopColsDF1 = []
    TopIntDF1 = []
    # spline fit top and bot laser boundaries
    tck_tt, u_tt = interpolate.splprep([FRONT_TOP, TOPCOLSDF])
    tck_tb, u_tb = interpolate.splprep([BACK_TOP, TOPCOLSDF])
    dy_tt, dx_tt = interpolate.splev(u_tt, tck_tt, der=0)
    dy_tb, dx_tb = interpolate.splev(u_tb, tck_tb, der=0)
    for counter, i in enumerate(FRONT_TOP):
        # changed vnum_t, put int and round functions in
        vnum_t = int(round(BACK_TOP[counter] - FRONT_TOP[counter]))
        #vnum_t = 100
        rows1 = np.linspace(FRONT_TOP[counter], BACK_TOP[counter], vnum_t)
        rows1int = rows1.astype(np.int)
        cols1start = dx_tt[counter]
        cols1stop = dx_tb[counter]
        # end points of profile are based on spline curve
        cols1 = np.linspace(cols1start, cols1stop, vnum_t)
        cols1int = cols1.astype(np.int)
        #print(cols1int)
        # changed za1 below, used to be za1 = Image[rows1int, cols1int[counter]]
        za1 = Image[rows1int, cols1int]
        TopRowsDF1.append(rows1int)
        TopColsDF1.append(cols1int)
        TopIntDF1.append(za1)

        plt.show()
    return [TopRowsDF1, TopColsDF1, TopIntDF1]



# profiles are now bound by gaussian fits
TRowsDF, TColsDF, TIntDF = Top_Transect_Profiles(im, tt_coordsy, tb_coordsy, tt_coordsx)



# plotting background subtracted data
f, ax2 = plt.subplots(1, 2)
ax2[0].imshow(im, cmap='gray')

for counter, i in enumerate(tt_coordsy):
    ax2[0].plot([TColsDF[counter][0], TColsDF[counter][-1]], [TRowsDF[counter][0], TRowsDF[counter][-1]], color='lime')
    ax2[0].plot(TColsDF[counter][0], TRowsDF[counter][0], 'ro', ms=1)
    ax2[0].plot(TColsDF[counter][-1], TRowsDF[counter][-1], 'ro', ms=1)
    ax2[1].plot(TIntDF[counter], label=str(TColsDF[counter][0]))
ax2[1].set_title('Bounded Top Beam Profiles')
ax2[1].set_ylabel('Intensity')
ax2[1].set_xlabel('Y Coordinate')
ax2[1].legend(loc=1)
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
    for counter, row in enumerate(y):
        riemannval = riemann(0, len(y[counter]), y[counter], 1)
        simpsonval = simpson(y[counter], np.arange(len(y[counter])))
        riemannarray.append(riemannval)
        simpsonarray.append(simpsonval)
    for counter, array in enumerate(d_rows):
        profilecoords = zip(d_rows[counter], d_cols[counter])
        profilecoordsarray.append(profilecoords)
    return [profilecoordsarray, riemannarray, simpsonarray]



profilescolslist_T, riemann_T, simpson_T = loopintegration(TRowsDF, TColsDF, TIntDF)
#profilescolslist_B, riemann_B, simpson_B = loopintegration(PXC_B, PYC_B, ZB)

TopProfNumberList = np.arange(len(riemann_T))
# basically change the first number currently 4 to move the bottom beam data, trying to get it to overlap with top beam
# this is because the geometry is not perfect and there is some overlap in angles between top and bottom beams
BotProfNumberList = [x + (4 - 1) for x in TopProfNumberList]


f4, ax4 = plt.subplots()
ax4.plot(TopProfNumberList, riemann_T, 'r--', label='Riemann Sum Top Beam')
ax4.plot(TopProfNumberList, simpson_T, 'r*', label='Simpson Method Top Beam')
# i had to reverse the array ([::-1]) because the bottom beam is read from right to left, the top beam from left to right
# this is because the bottom beam at the right is close to 90 degrees and the far left is close to 180 degrees
#ax4.plot(BotProfNumberList, riemann_B[::-1], 'b--',label='Riemann Sum Bottom Beam')
#ax4.plot(BotProfNumberList, simpson_B[::-1], 'b*', label='Simpson Method Bottom Beam')
ax4.set_title('Integrated Intensity as a Function of the Number of the Profile Evaluated')
ax4.set_xlabel('Profile Number')
ax4.set_ylabel('Integrated Intensity Along Profile')
plt.legend(loc=1)
plt.show()


#TRowsDF = pd.DataFrame(TRowsDF)
#TColsDF = pd.DataFrame(TColsDF)
#TIntDF = pd.DataFrame(TIntDF)
#BRowsDF = pd.DataFrame(BRowsDF)
#BColsDF = pd.DataFrame(BColsDF)
#BIntDF = pd.DataFrame(BIntDF)

#AllRowsDF = TRowsDF.append(BRowsDF, ignore_index=True)
#AllColsDF = TColsDF.append(BRowsDF, ignore_index=True)
#AllIntDF = TIntDF.append(BRowsDF, ignore_index=True)

#AllRowsDF.to_csv('/home/austen/PycharmProjects/Polar Nephelometer/Data/05-25-2017/Rows.csv', sep=',')
#AllColsDF.to_csv('/home/austen/PycharmProjects/Polar Nephelometer/Data/05-25-2017/Cols.csv', sep=',')
#AllIntDF.to_csv('/home/austen/PycharmProjects/Polar Nephelometer/Data/05-25-2017/Intensities.csv', sep=',')

#Save Phase Function
PF_Path = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Polar-Nephelometer/Data/12-06-2017/N2/PF_N2_SUB_2.txt'
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Top Profile Number'] = TopProfNumberList
PhaseFunctionDF['Top Riemann Sum'] = riemann_T
#PhaseFunctionDF['Bot Profile Number'] = BotProfNumberList
#PhaseFunctionDF['Bot Riemann Sum'] = riemann_B[::-1]
#PhaseFunctionDF.to_csv(PF_Path)






