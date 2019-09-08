'''
Author: Austen K. Scruggs
Date: 03-01-2018
Description: A list of useful functions I wrote to evaluate images or do certain tasks
with regard to the Polar Nephelometer Data
'''

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from scipy.integrate import simps as simpson
from scipy.optimize import curve_fit
from skimage.exposure import histogram
import math

def Ndarray_Average(directory):
    nd_array = []
    file_list = os.listdir(directory)
    for file in file_list:
        print(file)
        nd_arr_element = np.loadtxt(directory + '/' + file, dtype='int', delimiter='\t')
        nd_array.append(nd_arr_element)
    nd_np_array = np.array(nd_array)
    nd_arr_avg = np.transpose(np.squeeze(nd_np_array.mean(axis=0, dtype='int', keepdims=True), axis=0))
    print(nd_arr_avg.shape)
    return nd_arr_avg

def Image_Heatmaps(bkg, image, bins, Path):
    histBKG, bins_centerBKG = histogram(bkg, nbins=bins+1)
    histIM, bins_centerIM = histogram(image, nbins=bins+1)
    f0, ax0 = plt.subplots(2, 2)
    heatmap_bkg = ax0[0, 0].pcolormesh(np.flip(bkg, 0), cmap='gray', vmax=bins)
    heatmap_im = ax0[1, 0].pcolormesh(np.flip(image, 0), cmap='gray', vmax=bins)
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
    plt.savefig(Path)
    plt.close()
    return


# This is a function that returns a gaussian based on the input parameters
def gaussian(x, b, c, a, d):
    return d + (abs(a) * np.exp((-(x - b)**2.00)/(2.00 * c**2.00)))

def two_gaussians(x, b, c, a, d, e, f, g, h):
    return (d + (abs(a) * np.exp((-(x - b)**2.00)/(2.00 * c**2.00)))) + (h + (abs(e) * np.exp((-(x - f)**2.00)/(2.00 * g**2.00))))

def area_under_gaussian(x, b, c, a, d):
    return (1.00/2.00)*((2.00 * d * x) - (math.sqrt(2.00 * math.pi) * a * c * math.erf((b-x)/(math.sqrt(2.00)*c))))


# Draws vertical transects all over the image to find the beam within the specified bounds
def Profiles(Image, start, stop, numberrows, numbercols, top_bound, bot_bound):
    RowsDF = []
    ColsDF = []
    IntDF = []
    rows1 = np.linspace(top_bound, bot_bound, numberrows)
    rows1int = rows1.astype(int)
    columns = np.linspace(start, stop, numbercols)
    columns1int = columns.astype(np.int)
    for i in columns1int:
        z1 = Image[rows1int, np.int(i)]
        RowsDF.append(rows1int)
        ColsDF.append(i)
        IntDF.append(z1)
    return [RowsDF, ColsDF, IntDF]

def Profiles2(Image, start, stop, numberrows, numbercols, top_bound, bot_bound, func, delta):
    RowsDF = []
    TopBound = []
    BotBound = []
    IntDF = []
    max_intensity_col_coords = []
    max_intensity_row_coords = []
    rows1 = np.linspace(top_bound, bot_bound, numberrows)
    rows1int = rows1.astype(int)
    columns = np.linspace(start, stop, numbercols)
    columns1int = columns.astype(np.int)
    for i in columns1int:
        z1 = Image[rows1int, np.int(i)]
        z1_index = np.argmax(z1)
        max_intensity_col_coords.append(i)
        max_intensity_row_coords.append(rows1int[0] + z1_index)
    #print(max_intensity_row_coords)
    popt, pcov = curve_fit(func, max_intensity_col_coords, max_intensity_row_coords)
    mid_fit_row_coords = func(max_intensity_col_coords, *popt)
    MidDF = [int(round(x)) for x in mid_fit_row_coords]
    top_fit_row_coords = np.array(mid_fit_row_coords) - int(round(delta))
    bot_fit_row_coords = np.array(mid_fit_row_coords) + int(round(delta))
    ColsDF = max_intensity_col_coords
    for counter, element in enumerate(max_intensity_col_coords):
        r = np.arange(top_fit_row_coords[counter], bot_fit_row_coords[counter] + 1, 1)
        r = [np.int(x) for x in r]
        z2 = Image[r, np.int(element)]
        IntDF.append(z2)
        RowsDF.append(r)
        TopBound.append(min(r))
        BotBound.append(max(r))
    return [RowsDF, ColsDF, TopBound, BotBound, MidDF, IntDF]


def quadratic(x, a, b, c):
    X = np.array(x)
    return a * np.power(X, 2) + b * X + c

def quartic(x, a, b, c, d, e):
    X = np.array(x)
    return a * np.power(X, 4) + b * np.power(X, 3) + c * np.power(X, 2) + d * X + e

def linear(x, m, b):
    X = np.array(x)
    return m * X + b


def Coordinate_Fit(coordinates, function):
    popt, pcov = curve_fit(function, coordinates[0], coordinates[1])
    return [popt, pcov]


def Curve_Fit_Profiles(function1, function2, rowmat, intmat, CDF):
    fitdatay = []
    fitdatax = []
    fittopbound = []
    fitbotbound = []
    fitbeammiddle = []
    sigma = []
    background = []
    fyd1 = []
    col_array=[]
    area_under_gaussian_array = []
    for counter, row in enumerate(rowmat):
        meanx = rowmat[counter][intmat[counter].argmax()]
        popt, pcov = curve_fit(function1, rowmat[counter], intmat[counter], p0=[meanx, 50.0, 0.0, 40.0], maxfev=1200)
        topbound = popt[0] - abs(3 * popt[1])
        botbound = popt[0] + abs(3 * popt[1])
        # aug is area under the gaussian curve of an individual profile, an array of aug at all profile numbers is the phase function
        # subtracting by popt[3]* botbound from the bottom part and by popt[3]* topbound from the top part ensures
        # that the area underneath the gaussian that is not signal but is background is subtracted out analagous
        # to the work conducted with the PINEPH created by Dolgos and Martins
        aug = (area_under_gaussian(botbound, *popt) - (popt[3] * botbound)) - (area_under_gaussian(topbound, *popt) - (popt[3] * topbound))
        fitdatay.append(function1(rowmat[counter], *popt))
        fitdatax.append(rowmat[counter])
        fittopbound.append(topbound)
        fitbotbound.append(botbound)
        fitbeammiddle.append(popt[0])
        sigma.append(abs(popt[1]))
        background.append(abs(popt[3]))
        area_under_gaussian_array.append(aug)
    # here I subtracted out the background, which is the area under the flat region of the gaussian
    for counter, fitydata in enumerate(fitdatay):
        array1 = [x - background[counter] for x in fitydata]
        fyd1.append(array1)
        #the below is for if you don't want to show fits with background subtracted
        #fyd1.append(fitydata)

    area_under_gaussian_array_norm = [x / np.sum(area_under_gaussian_array) for x in area_under_gaussian_array]
    TT_Edge_Coordinates = np.vstack((CDF, fittopbound))
    TB_Edge_Coordinates = np.vstack((CDF, fitbotbound))
    ycoordfitparams_t, ycoordfitcov_t = Coordinate_Fit(TT_Edge_Coordinates, function2)
    ycoordfitparams_b, ycoordfitcov_b = Coordinate_Fit(TB_Edge_Coordinates, function2)
    TT_smoothcoords = function2(CDF, *ycoordfitparams_t)
    TB_smoothcoords = function2(CDF, *ycoordfitparams_b)
    return [fitdatax, fitdatay, fyd1, fittopbound, fitbotbound, fitbeammiddle, sigma, background, area_under_gaussian_array, area_under_gaussian_array_norm, TT_smoothcoords, TB_smoothcoords, CDF]


def Curve_Fit_Profiles2(function1, rowmat, colmat, tb, bb, md, intmat):
    fitdatay = []
    fitdatax = []
    fittopbound = []
    fitbotbound = []
    fitbeammiddle = []
    sigma = []
    background = []
    fyd1 = []
    col_array = []
    area_under_gaussian_array = []
    for counter, row in enumerate(rowmat):
        meanx = rowmat[counter][intmat[counter].argmax()]
        popt, pcov = curve_fit(function1, rowmat[counter], intmat[counter], p0=[meanx, 50.0, 0.0, 40.0], maxfev=1200)
        topbound = tb[counter]
        botbound = bb[counter]
        midbound = md[counter]
        # aug is area under the gaussian curve of an individual profile, an array of aug at all profile numbers is the phase function
        # subtracting by popt[3]* botbound from the bottom part and by popt[3]* topbound from the top part ensures
        # that the area underneath the gaussian that is not signal but is background is subtracted out analagous
        # to the work conducted with the PINEPH created by Dolgos and Martins
        aug = (area_under_gaussian(botbound, *popt) - (popt[3] * botbound)) - (
                    area_under_gaussian(topbound, *popt) - (popt[3] * topbound))
        fitdatay.append(function1(rowmat[counter], *popt))
        fitdatax.append(rowmat[counter])
        fittopbound.append(topbound)
        fitbotbound.append(botbound)
        fitbeammiddle.append(midbound)
        sigma.append(abs(botbound - topbound))
        background.append(abs(popt[3]))
        area_under_gaussian_array.append(aug)
    # here I subtracted out the background, which is the area under the flat region of the gaussian
    for counter, fitydata in enumerate(fitdatay):
        array1 = [x - background[counter] for x in fitydata]
        fyd1.append(array1)
        # the below is for if you don't want to show fits with background subtracted
        # fyd1.append(fitydata)
    area_under_gaussian_array_norm = [x / np.sum(area_under_gaussian_array) for x in area_under_gaussian_array]
    return [fitdatax, fitdatay, fyd1, fittopbound, fitbotbound, fitbeammiddle, colmat, sigma, background,
            area_under_gaussian_array, area_under_gaussian_array_norm]


def Perp_Profiles(image, ttsmoothcoords, tbsmoothcoords, colsarray, num_cols):
    BES = []
    PES = []
    tt_coords = []
    tb_coords = []
    ts = []
    z = []
    x_pts = []
    y_pts = []
    # find the slopes setting the first to slopes of the first two columns to the same value, append slopes to array
    for counter, element in enumerate(ttsmoothcoords):
        if counter >= 1 and counter < num_cols:
            slope = (ttsmoothcoords[counter] - ttsmoothcoords[counter - 1])/abs(colsarray[counter] - colsarray[counter - 1])
            BES.append(slope)
    # find perpendicular slope to the slopes found and append to array
    for element in BES:
        perp_slope = -1 * (element ** -1)
        #print(perp_slope)
        PES.append(perp_slope)
    # find where perpendicular slope intersects bottom edge of beam
    for counter, element in enumerate(PES):
        if counter >= 1 and counter < num_cols:
            counter = int(counter)
            b = ttsmoothcoords[counter] - (element * colsarray[counter])
            #print(b)
            transect_array = [(element * x) + b for x in colsarray]
            idxt = abs(np.subtract(transect_array, ttsmoothcoords[counter])).argmin()
            tt_rows = transect_array[idxt]
            tt_cols = colsarray[idxt]
            #print(idxt)
            tt_coords.append([tt_rows, tt_cols])
            idxb = abs(np.subtract(transect_array, tbsmoothcoords[counter])).argmin()
            tb_rows = transect_array[idxb]
            tb_cols = colsarray[idxb]
            tb_coords.append([tb_rows, tb_cols])
    for counter, element in enumerate(tt_coords):
        ts_val = np.sqrt((abs(tb_coords[counter][0] - tt_coords[counter][0])) ** 2 + (abs(tt_coords[counter][1] - tb_coords[counter][1])) ** 2)
        ts.append(ts_val)
        y_pts.append([tt_coords[counter][0], tb_coords[counter][0]])
        x_pts.append([tt_coords[counter][1], tb_coords[counter][1]])
    for counter, element in enumerate(tt_coords):
        if counter >= 1 and counter < num_cols:
            counter = int(counter)
            x, y = np.linspace(np.rint(x_pts[counter][0]),np.rint(x_pts[counter][1]), np.rint(ts[counter])), np.linspace(np.rint(y_pts[counter][0]),np.rint(y_pts[counter][1]), np.rint(ts[counter]))
            zi = image[y.astype(np.int),x.astype(np.int)]
            z.append(zi)
    return(z, tt_coords, tb_coords, ts, BES, PES, x_pts, y_pts)



    # find where the line intersects the top bound

    # find where the line intersects the bot bound




def Image_Subtract(Image1, Image2):
    A = Image1.astype(np.int)
    B = Image2.astype(np.int)
    C = np.subtract(A, B)
    # replaces all elements less than zero in an ndarray with zero
    C[C < 0] = 0
    D = C.astype(np.int)
    return(D)


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


def loop_integration(d_cols, y):
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
        profilecoordsarray.append(element)
    return [profilecoordsarray, riemannarray, simpsonarray]


def loop_simps_riemann(y):
    # data needed to be a column array
    # data needed to be column array, this needs fixing in labview
    # data needed to be column arrays here as well, this needs fixing in labview
    # create plots
    #acquire the number of columns in the dataframe for use later
    riemannarray = []
    simpsonarray = []
    for counter, row in enumerate(y):
        riemannval = riemann(0, len(y[counter]), y[counter], 1)
        simpsonval = simpson(y[counter], np.arange(len(y[counter])))
        riemannarray.append(riemannval)
        simpsonarray.append(simpsonval)
    return [simpsonarray, riemannarray]


# points are in (row, column) tuple
def Set_Beam_Edges(Image, top_point1, top_point2, bot_point1, bot_point2, num):
    y_coords_mat = []
    Scattering_Diagram = []
    x_coords_mat = []
    cols = []
    Intensities = []
    f1, ax1 = plt.subplots(ncols=2)
    ax1[0].imshow(Image, cmap='gray')
    top_line_rowvals = np.linspace(top_point1[0], top_point2[0], num).astype(np.int)
    top_line_colvals = np.linspace(top_point1[1], top_point2[1], num).astype(np.int)
    bot_line_rowvals = np.linspace(bot_point1[0], bot_point2[0], num).astype(np.int)
    bot_line_colvals = np.linspace(bot_point1[1], bot_point2[1], num).astype(np.int)
    for counter, i in enumerate(np.arange(num)):
        a = bot_line_rowvals[counter] - top_line_rowvals[counter]
        b = bot_line_colvals[counter] - top_line_colvals[counter]
        magnitude = round(math.sqrt((a) ** 2 + (b) ** 2))
        ycoords = np.linspace(top_line_rowvals[counter], bot_line_rowvals[counter], magnitude).astype(np.int)
        y_coords_mat.append(ycoords)
        xcoords = np.linspace(top_line_colvals[counter], bot_line_colvals[counter], magnitude).astype(np.int)
        x_coords_mat.append(xcoords)
        zi = Image[ycoords, xcoords]
        Intensities.append(zi.astype(np.int))
        ax1[0].plot([xcoords[0].astype(np.int), xcoords[-1].astype(np.int)], [ycoords[0].astype(np.int), ycoords[-1].astype(np.int)], 'r-')
        ax1[1].plot(zi)
    plt.show()
    for element in x_coords_mat:
        cols.append(element[0])
    for counter, element in enumerate(Intensities):
        Column_Integrand = np.sum(element)
        Scattering_Diagram.append(Column_Integrand)
    return [y_coords_mat, x_coords_mat, cols, Intensities, Scattering_Diagram]

# loop average of images (cannot sum due to the fact that everything is 8 bit, averaging mitigates overflow)
# the 0 in imread means its grayscale, a 1 would mean color, and a -1 would mean unchanged
def Loop_Image_Sum(Path_P, *args):
    # list files in directory
    file_list = os.listdir(Path_P)
    # number of files in directory
    num_files = len(file_list)
    if args and len(args) == 1:
        print(args[0])
        for counter, fn in enumerate(file_list):
            if counter == 0 and fn != 'Thumbs.db':
                A = cv2.imread(Path_P+'/'+str(fn), 0)
                #plt.imshow(A, cmap='gray')
                #plt.show()
                A = A.astype('int')
                S = cv2.imread(args[0], 0)
                #plt.imshow(S, cmap='gray')
                #plt.show()
                S = S.astype('int')
                A = A - S
                #plt.imshow(A, cmap='gray')
                #plt.show()
                A[A < 0] = 0
            if counter > 0 and fn != 'Thumbs.db':
                print(fn)
                B = cv2.imread(Path_P + '/' + str(fn), 0)
                B = B.astype('int')
                S = cv2.imread(args[0], 0)
                S = S.astype('int')
                B = B - S
                B[B < 0] = 0
                C = A.astype('int') + B.astype('int')
                A = C
    if args and len(args) == 2:
        for counter, fn in enumerate(file_list):
            if counter == 0 and fn != 'Thumbs.db':
                A = cv2.imread(Path_P + '/' + str(fn), 0)
                A = A.astype('int')
                S = cv2.imread(args[0], 0)
                S = S.astype('int')
                H = cv2.imread(args[1], 0)
                H = H.astype('int')
                A = A - S - H
                A[A < 0] = 0
            if counter > 0 and fn != 'Thumbs.db':
                #print(fn)
                B = cv2.imread(Path_P + '/' + str(fn), 0)
                B = B.astype('int')
                S = cv2.imread(args[0], 0)
                S = S.astype('int')
                H = cv2.imread(args[1], 0)
                H = H.astype('int')
                B = B - S - H
                B[B < 0] = 0
                C = A.astype('int') + B.astype('int')
                A = C
    else:
        for counter, fn in enumerate(file_list):
            if counter == 0 and fn != 'Thumbs.db':
                A = cv2.imread(Path_P + '/' + str(fn), 0)
                A = A.astype('int')
                A[A < 0] = 0
            if counter > 0 and fn != 'Thumbs.db':
                print(fn)
                B = cv2.imread(Path_P + '/' + str(fn), 0)
                B = B.astype('int')
                B[B < 0] = 0
                C = A.astype('int') + B.astype('int')
                A = C
    return(A)


def Loop_Image_Average(Path_P, *args):
    # list files in directory
    file_list = os.listdir(Path_P)
    # number of files in directory
    num_files = len(file_list)
    if args and len(args) == 1:
        print(args[0])
        for counter, fn in enumerate(file_list):
            if counter == 0 and fn != 'Thumbs.db':
                A = cv2.imread(Path_P+'/'+str(fn), -1)
                #plt.imshow(A, cmap='gray')
                #plt.show()
                A = A.astype('int')/16
                S = cv2.imread(args[0], -1)
                #plt.imshow(S, cmap='gray')
                #plt.show()
                S = S.astype('int')/16
                A = A - S
                #plt.imshow(A, cmap='gray')
                #plt.show()
                A[A < 0] = 0
            if counter > 0 and fn != 'Thumbs.db':
                print(fn)
                B = cv2.imread(Path_P + '/' + str(fn), -1)
                B = B.astype('int')/16
                S = cv2.imread(args[0], -1)
                S = S.astype('int')/16
                B = B - S
                B[B < 0] = 0
                C = A.astype('int') + B.astype('int')
                A = C
    if args and len(args) == 2:
        for counter, fn in enumerate(file_list):
            if counter == 0 and fn != 'Thumbs.db':
                A = cv2.imread(Path_P + '/' + str(fn), -1)
                A = A.astype('int')/16
                S = cv2.imread(args[0], -1)
                S = S.astype('int')/16
                H = cv2.imread(args[1], -1)
                H = H.astype('int')/16
                A = A - S - H
                A[A < 0] = 0
            if counter > 0 and fn != 'Thumbs.db':
                #print(fn)
                B = cv2.imread(Path_P + '/' + str(fn), -1)
                B = B.astype('int')/16
                S = cv2.imread(args[0], -1)
                S = S.astype('int')/16
                H = cv2.imread(args[1], -1)
                H = H.astype('int')/16
                B = B - S - H
                B[B < 0] = 0
                C = A.astype('int') + B.astype('int')
                A = C
    else:
        for counter, fn in enumerate(file_list):
            if counter == 0 and fn != 'Thumbs.db':
                A = cv2.imread(Path_P + '/' + str(fn), -1)
                A = A.astype('int')/16
                A[A < 0] = 0
            if counter > 0 and fn != 'Thumbs.db':
                print(fn)
                B = cv2.imread(Path_P + '/' + str(fn), -1)
                B = B.astype('int')/16
                B[B < 0] = 0
                C = A.astype('int') + B.astype('int')
                A = C
    Z = (A/num_files).astype('int')
    return(Z)


def find_nearest(a, a0):
    #Element in nd array `a` closest to the scalar value `a0`
    # the .flat is an iterator over 1d array, it allows me to pick out the proper index even in ndarrays
    idx = np.abs(a - a0).argmin()
    return [idx, a.flat[idx]]

