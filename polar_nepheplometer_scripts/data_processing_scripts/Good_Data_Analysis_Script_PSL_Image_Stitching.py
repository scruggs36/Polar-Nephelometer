'''
Austen K. Scruggs
08/06/2020
Description: Phase function stitching using PSL data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import data
file_directory = '/home/austen/Desktop/Recent/Good_Data_Riemann.txt'
df = pd.read_csv(file_directory, sep=',', header=0)
# eliminate extra column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# multiindex dataframe
df.set_index(['Sample', 'Size (nm)', 'Polarization'], inplace=True)
# pandas dataframe.xs returns a cross-section of the data, so basically I am filtering out data that isn't PSL, size 900, and pol = SL
xs_tuple = ('PSL', '900', 'SL')
df1 = df.xs(xs_tuple)
# so it looks like I have the most 900nm size measurements of PSL, I need a lot more at different exposure times and
# laser powers, I have the number of averages the same for all measurements, might wanna spice that up
# hit the 900nm PSL hard!
print(df1.loc[:, 'Exposure Time (s)'], df1.loc[:, 'Laser Power (mW)'], df1.loc[:, 'Number of Averages'])

