'''
Austen K. Scruggs
06-24-2019
Description: This function evaluates images and produces phase functions, it is designed to be used inside the greater
labview code
'''


import numpy as np


def Image_Analysis(im):
    rows = [500, 650]
    cols = [230, 1060]
    cols_array = (np.arange(cols[0], cols[1], 1)).astype(int)
    row_max_index_array = []
    for element in cols_array:
        arr = np.arange(rows[0], rows[1], 1).astype(int)
        im_transect = im[arr, element]
        max_index = np.argmax(im_transect)
        row_max_index_array.append(max_index + rows[0])
        
    tuner = len(cols_array)
    iterator = round(len(cols_array) / tuner)
    mid = []
    top = []
    bot = []
    sigma_pixels = 20
    for counter, element in enumerate(range(iterator)):
        if counter < iterator:
            print(counter)
            x = cols_array[(counter) * tuner: (counter + 1) * tuner]
            y = row_max_index_array[(counter) * tuner: (counter + 1) * tuner]
            print(x)
            # print(y)
            polynomial_fit = np.poly1d(np.polyfit(x, y, deg=6))
            # sigma_pixels = 20
            [mid.append(polynomial_fit(element)) for element in x]
            [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
            [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]
        if counter == iterator:
            print(counter)
            x = cols_array[(counter) * tuner: len(cols_array)]
            y = row_max_index_array[(counter) * tuner: len(row_max_index_array)]
            print(x)
            # print(y)
            polynomial_fit = np.poly1d(np.polyfit(x, y, deg=6))
            # sigma_pixels = 20
            [mid.append(polynomial_fit(element)) for element in x]
            [top.append(polynomial_fit(element) - sigma_pixels) for element in x]
            [bot.append(polynomial_fit(element) + sigma_pixels) for element in x]
