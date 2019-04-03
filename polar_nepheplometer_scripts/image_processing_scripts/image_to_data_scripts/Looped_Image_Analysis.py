'''
Austen K. Scruggs
10-10-2018
Desctription: This code averages background and sample images, then subtracts the image to get a resultant background
subtracted sample image. The image then gets evaluated in the same way that the labview code evaluates the images!
'''

from Neph_Functions import Loop_Image_Average,  Image_Subtract
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

# directory navigation i.e. path to image '//fcncfs4.franklin.uga.edu/CHEM/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/02-13-2018/N2/im_summed.png'
Path_Samp_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/10-09-2018/'
Path_N2_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/10-09-2018/N2'
Path_He_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/10-09-2018/He'
Path_BKG_Dir = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/10-11-2018/He'
Path_Save = '/home/austen/Documents'

# averaging background images
im_SampAvg = Loop_Image_Average(Path_Samp_Dir)
im_N2Avg = Loop_Image_Average(Path_N2_Dir)
im_HeAvg = Loop_Image_Average(Path_He_Dir)
im_BKGAvg = Loop_Image_Average(Path_BKG_Dir)

# plot of averaged background images
f0, ax0 = plt.subplots()
im_f0 = ax0.imshow(im_BKGAvg, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
f0.colorbar(im_f0, cax=cax)
plt.savefig(Path_Save + '/BKG_Averaged.pdf', format='pdf')
plt.show()


# plot of averaged He images
f1, ax1 = plt.subplots()
im_f1 = ax1.imshow(im_BKGAvg, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
f1.colorbar(im_f1, cax=cax)
plt.savefig(Path_Save + '/Helium_Averaged.pdf', format='pdf')
plt.show()

'''
# I think I can eliminate this
# writes image to new file
cv2.imwrite(Path_Save + '/Background.png', im_BKGAvg)
imBKG = cv2.imread(Path_Save + '/Background.png', 0)
'''

# plot of averaged nitrogen images
f2, ax2 = plt.subplots()
im_f2 = ax2.imshow(im_N2Avg, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
f2.colorbar(im_f2, cax=cax)
plt.savefig(Path_Save + '/Nitrogen_Averaged.pdf', format='pdf')
plt.show()

'''
# I think I can eliminate this
# writes image to new file
cv2.imwrite(Path_Save + '/Sample.png', imAvg)
im = cv2.imread(Path_Save + '/Sample.png', 0)
'''
# Background subtraction, not ready to do this yet this belongs below
imRes = Image_Subtract(im, imBKG)

'''
# I think I can eliminate this
# writes image to new file
cv2.imwrite(Path_Save + '/Sample_Background_Subtracted.png', imRes)
im_Res = cv2.imread(Path_Save + '/Sample_Background_Subtracted.png', 0)
'''

'''
# plot of the N2 corrected (helium and bkg) subtracted data
f3, ax3 = plt.subplots()
im_f3 = ax3.imshow(imRes, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
f3.colorbar(im_f3, cax=cax)
plt.savefig(Path_Save + '/N2_Corrected.pdf', format='pdf')
plt.show()
'''

# Initial boundaries on the image
rows = [526, 671]
cols = [240, 1040]
cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
#ROI = im[rows[0]:rows[1], cols[0]:cols[1]]


row_max_index_array_sub = []
row_max_index_array_nosub = []
for element in cols_array:
    arr = np.arange(rows[0], rows[1], 1).astype(int)
    im_transect_nosub = im_N2Avg[arr, element]
    im_transect_sub = imRes[arr, element]
    index_nosub = np.argmax(im_transect_nosub)
    index_sub = np.argmax(im_transect_sub)
    row_max_index_array_nosub.append(index_nosub + rows[0])
    row_max_index_array_sub.append(index_sub + rows[0])

# polynomial fit to find the middle of the beam, the top bound, and bot bound
polynomial_fit_nosub = np.poly1d(np.polyfit(cols_array, row_max_index_array_nosub, deg=2))
polynomial_fit_sub = np.poly1d(np.polyfit(cols_array, row_max_index_array_sub, deg=2))

sigma_pixels = 40

mid_nosub = polynomial_fit_nosub(cols_array)
top_nosub = polynomial_fit_nosub(cols_array) - sigma_pixels
bot_nosub = polynomial_fit_nosub(cols_array) + sigma_pixels

mid_sub = polynomial_fit_sub(cols_array)
top_sub = polynomial_fit_sub(cols_array) - sigma_pixels
bot_sub = polynomial_fit_sub(cols_array) + sigma_pixels

# loop through transects and acquire profiles and scattering diagram intensities vs profile numbers
SD_nosub = []
arr_ndarray_nosub = []
bound_transect_ndarray_nosub = []
background_nosub = []
SD_nosub_bkg_corrected = []
for counter, element in enumerate(cols_array):
    arr_nosub = np.arange(top_nosub[counter], bot_nosub[counter], 1).astype(int)
    arr_ndarray_nosub.append(arr_nosub)
    bound_transect_nosub = im[arr_nosub, element]
    bkg_pts_nosub = np.average(bound_transect_nosub[:10]) * len(bound_transect_nosub)
    bound_transect_ndarray_nosub.append(bound_transect_nosub)
    transect_summed_nosub = np.sum(bound_transect_nosub)
    SD_nosub.append(transect_summed_nosub)
    background_nosub.append(bkg_pts_nosub)
    SD_nosub_bkg_corrected.append(transect_summed_nosub - bkg_pts_nosub)

SD_sub = []
arr_ndarray_sub = []
bound_transect_ndarray_sub = []
background_sub = []
SD_sub_bkg_corrected = []
for counter, element in enumerate(cols_array):
    arr_sub = np.arange(top_sub[counter], bot_sub[counter], 1).astype(int)
    arr_ndarray_sub.append(arr_sub)
    bound_transect_sub = imRes[arr_sub, element]
    bkg_pts_sub = np.average(bound_transect_sub[:10]) * len(bound_transect_sub)
    bound_transect_ndarray_sub.append(bound_transect_sub)
    transect_summed_sub = np.sum(bound_transect_sub)
    SD_sub.append(transect_summed_sub)
    background_sub.append(bkg_pts_sub)
    SD_sub_bkg_corrected.append(transect_summed_sub - bkg_pts_sub)

# plot of the backgound subtracted data with bounds
f3, ax3 = plt.subplots(2, 2, figsize=(12, 6))
im_f3 = ax3[0, 0].imshow(im, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax3[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f3.colorbar(im_f3, cax=cax)
ax3[0, 1].imshow(im, cmap='gray')
ax3[0, 1].plot(cols_array, mid_nosub, marker='.', ms=0.1, color='red')
ax3[0, 1].plot(cols_array, bot_nosub, marker='.', ms=0.1, color='blue')
ax3[0, 1].plot(cols_array, top_nosub, marker='.', ms=0.1, color='yellow')
ax3[0, 1].set_xlabel('Columns')
ax3[0, 1].set_ylabel('Rows')
ax3[0, 1].set_title('Averaged Nitrogen')
ax3[1, 0].plot(arr_ndarray_nosub, bound_transect_ndarray_nosub, linestyle='-')
ax3[1, 0].set_xlabel('Rows')
ax3[1, 0].set_ylabel('Intensity (DN)')
ax3[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax3[1, 0].grid(True)
ax3[1, 1].plot(cols_array, SD_nosub, linestyle='-', color='red', label='scattering diagram')
ax3[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax3[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax3[1, 1].set_title('Scattering Diagram')
ax3[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Path_Save + '/F3_Rayleigh.pdf', format='pdf')
plt.show()


# plot of the backgound subtracted data with bounds
f4, ax4 = plt.subplots(2, 2, figsize=(12, 6))
im_f4 = ax4[0, 0].imshow(imRes, cmap='gray')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax4[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
f4.colorbar(im_f3, cax=cax)
ax4[0, 1].imshow(imRes, cmap='gray')
ax4[0, 1].plot(cols_array, mid_sub, marker='.', ms=0.1, color='red')
ax4[0, 1].plot(cols_array, bot_sub, marker='.', ms=0.1, color='blue')
ax4[0, 1].plot(cols_array, top_sub, marker='.', ms=0.1, color='yellow')
ax4[0, 1].set_xlabel('Columns')
ax4[0, 1].set_ylabel('Rows')
ax4[0, 1].set_title('Averaged Nitrogen \n Helium Subtracted Image')
ax4[1, 0].plot(arr_ndarray_sub, bound_transect_ndarray_sub, linestyle='-')
ax4[1, 0].set_xlabel('Rows')
ax4[1, 0].set_ylabel('Intensity (DN)')
ax4[1, 0].set_title('Profiles Taken Along Vertical \n Bounded Transects')
ax4[1, 0].grid(True)
ax4[1, 1].plot(cols_array, SD_sub, linestyle='-', color='red', label='scattering diagram')
ax4[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax4[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax4[1, 1].set_title('Scattering Diagram')
ax4[1, 1].grid(True)
plt.tight_layout()
plt.savefig(Path_Save + '/F4_Rayleigh.pdf', format='pdf')
plt.show()

#Save Phase Function
PhaseFunctionDF = pd.DataFrame()
PhaseFunctionDF['Columns'] = cols_array
PhaseFunctionDF['Intensity'] = SD_nosub
PhaseFunctionDF['Background Subtracted Intensity'] = SD_sub
PhaseFunctionDF.to_csv(Path_Save + '/SD.txt')

f5, ax5 = plt.subplots(figsize=(12, 7))
ax5.plot(cols_array, SD_nosub, color='black', linestyle='-', label='N2')
ax5.plot(cols_array, SD_sub, color='blue', linestyle='-', label='N2 - He')
ax5.plot(cols_array, SD_nosub_bkg_corrected, color='red', linestyle='-', label='N2 bkg corrected')
ax5.plot(cols_array, SD_sub_bkg_corrected, color='green', linestyle='-', label='N2 - He bkg corrected')
ax5.set_xlabel('Profile Numbers (column numbers)')
ax5.set_ylabel('Summed Profile Intensities (DN)')
ax5.set_title('Scattering Diagrams Compared')
ax5.grid(True)
ax5.legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F5_Rayleigh.pdf', format='pdf')
plt.show()

f6, ax6 = plt.subplots(2, 2, figsize=(12, 7))
ax6[0, 0].plot(arr_ndarray_nosub[50], bound_transect_ndarray_nosub[50], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_nosub[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_nosub[50][:10])*len(bound_transect_ndarray_nosub[50])))
ax6[0, 0].plot(arr_ndarray_sub[50], bound_transect_ndarray_sub[50], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_sub[50])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_sub[50][:10])*len(bound_transect_ndarray_sub[50])))
ax6[0, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 0].set_title('Profiles Compared')
ax6[0, 0].grid(True)
ax6[0, 0].legend(loc=1)
ax6[0, 1].plot(arr_ndarray_nosub[200], bound_transect_ndarray_nosub[200], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_nosub[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_nosub[200][:10])*len(bound_transect_ndarray_nosub[200])))
ax6[0, 1].plot(arr_ndarray_sub[200], bound_transect_ndarray_sub[200], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_sub[200])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_sub[200][:10])*len(bound_transect_ndarray_sub[200])))
ax6[0, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[0, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[0, 1].set_title('Profiles Compared')
ax6[0, 1].grid(True)
ax6[0, 1].legend(loc=1)
ax6[1, 0].plot(arr_ndarray_nosub[550], bound_transect_ndarray_nosub[550], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_nosub[550])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_nosub[550][:10])*len(bound_transect_ndarray_nosub[550])))
ax6[1, 0].plot(arr_ndarray_sub[550], bound_transect_ndarray_sub[550], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_sub[550])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_sub[550][:10])*len(bound_transect_ndarray_sub[550])))
ax6[1, 0].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 0].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 0].set_title('Profiles Compared')
ax6[1, 0].grid(True)
ax6[1, 0].legend(loc=1)
ax6[1, 1].plot(arr_ndarray_nosub[700], bound_transect_ndarray_nosub[700], 'b-', label='Raw: Int Sum=' + str(np.sum(bound_transect_ndarray_nosub[700])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_nosub[700][:10])*len(bound_transect_ndarray_nosub[700])))
ax6[1, 1].plot(arr_ndarray_sub[700], bound_transect_ndarray_sub[700], 'r-', label='Subtracted: Int Sum=' + str(np.sum(bound_transect_ndarray_sub[700])) + ' Bkg Sum=' + str(np.average(bound_transect_ndarray_sub[700][:10])*len(bound_transect_ndarray_sub[700])))
ax6[1, 1].set_xlabel('Profile Numbers (column numbers)')
ax6[1, 1].set_ylabel('Summed Profile Intensities (DN)')
ax6[1, 1].set_title('Profiles Compared')
ax6[1, 1].grid(True)
ax6[1, 1].legend(loc=1)
plt.tight_layout()
plt.savefig(Path_Save + '/F6_Profiles.pdf', format='pdf')
plt.show()



