'''
Austen K. Scruggs
04-12-2019
Description: Turning smps txt files into viewable data.
'''
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

path = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Thermodenuder/SMPS Data'
save = '/home/austen/Documents'
file_list = os.listdir(path)

# function we wrote
def SMPS_Parser(fn_directory, fn, save_directory, save_fn):
    df = []
    with open(fn_directory + '/' + fn, "rb") as f:
        lines = f.readlines()
        for counter1, element in enumerate(lines):
            if counter1 >= 15:
                line = str(element).split('\\t')
                df.append(line)
        smps_ndarray = np.array(df)
        smps_dataframe = pd.DataFrame(smps_ndarray)
        smps_dataframe.to_csv(save_directory + '/' + save_fn, header=0)
        #print(smps_ndarray.shape)
        #print(smps_ndarray)
    return(smps_ndarray)

# call function to write data into txt
SMPS_Parser(path, file_list[1], save, 'SMPS_DF_20191204.csv')


# working on analysis prior to injecting into parsing function
data_path = '/home/austen/Documents/TD_Data_T0/Data'
f_list = os.listdir(data_path)

# importing txt files and reading
td_data = pd.read_csv(data_path + '/' + f_list[1], sep=',', header=0)
print(td_data.shape)
# creating size bins from headers
size_bins = np.array(td_data.columns.values)[5:112].astype(float)

# datetime conversion
# date time string format
fm = '%m/%d/%y %H:%M:%S'
# format for plots
fm2 = '%H:%M:%S'
# zipping dates and times then converting them the datetime, then date2num
dates_times = [m+' '+n for m, n in zip(td_data['Date'], td_data['Start Time'])]
#print(dates_times)
dt = [datetime.strptime(element, fm) for element in dates_times]
#print(dt)
dtp = mdates.date2num(dt)
#print(dtp)



# create figure
fig0 = plt.figure(figsize=(12, 6))
ax0 = fig0.add_subplot(121)
# pre-allocating array
counts_array = []
# plot all the distributions
for element in range(td_data.shape[0]):
    counts = np.array(td_data)[element, 5:112].astype(float)
    counts_array.append(counts)
    ax0.plot(size_bins, counts)
ax0.set_xlabel('Particle Diameter')
ax0.set_ylabel('Counts')
ax0.set_title('SMPS Distributions')
ax0.grid(True)
# plot geometric mean vs time
ax1 = fig0.add_subplot(122)
myfmt = mdates.DateFormatter(fm2)
ax1.plot_date(dtp, np.array(td_data['Geo. Mean(nm)']), color='green', ls='-')
ax1.xaxis.set_major_formatter(myfmt)
for tick in ax1.get_xticklabels():
        tick.set_rotation(30)
ax1.set_xlabel('Time')
ax1.set_ylabel('Geom. Mean')
ax1.set_title('Geometric Mean as a Function of Time')
ax1.grid(True)
plt.tight_layout()
plt.savefig(save + '/' + 'TD_SizeDist.png', format='png')
plt.show()






