import matplotlib.pyplot as plt
from Neph_Functions import *
import pandas as pd
import os

Path_QW = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/06-26-2018/QW_Analysis/QW_SD'
Path_HW = '/home/austen/media/winshare/Groups/Smith_G/Austen/Projects/Nephelometry/Polar Nephelometer/Data/06-26-2018/HW_Analysis/HW_SD/90_180_Degrees'
Path_Save = '/home/austen/Documents/Compare_Data'


# list files in directory
file_list = os.listdir(Path_QW)
#print(file_list)
# number of files in directory
num_files = len(file_list)
plt.figure(figsize=(10, 6))
for counter, fn in enumerate(file_list):
    data = pd.read_csv(Path_QW + '/' + str(fn), sep=',', engine='python')
    plt.plot(data['Columns'], data['Intensity'], label=str(fn))
plt.title('Quarter Waveplate 663nm Elliptically Polarized Light')
plt.yscale('log')
plt.xlabel('Profile Number')
plt.ylabel('Intensity')
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig(Path_Save + '/QW_SD.pdf', format='pdf')
plt.clf()


# list files in directory
file_list = os.listdir(Path_HW)
#print(file_list)
# number of files in directory
num_files = len(file_list)
plt.figure(figsize=(10, 6))
for counter, fn in enumerate(file_list):
    data = pd.read_csv(Path_HW + '/' + str(fn), sep=',', engine='python')
    plt.plot(data['Columns'], data['Intensity'], label=str(fn))
plt.title('Half Waveplate 663nm Linearly Polarized Light')
plt.yscale('log')
plt.xlabel('Profile Number')
plt.ylabel('Intensity')
plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig(Path_Save + '/HW_SD.pdf', format='pdf')
plt.clf()

'''
# list files in directory
file_list = os.listdir(Path_Perpendicular_600nm)
#print(file_list)
# number of files in directory
num_files = len(file_list)
for counter, fn in enumerate(file_list):
    data = pd.read_csv(Path_Perpendicular_600nm + '/' + str(fn), sep=',')
    plt.plot(data['Columns'], data['Intensity'], label=str(fn))
plt.title('600nm PSL Scattering Diagram of 663nm Light Polarized\n Perpendicular to the Scattering Plane')
plt.yscale('log')
plt.xlabel('Profile Number')
plt.ylabel('Intensity')
plt.legend(loc=1)
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig(Path_Save + '/PSL_600nm_SDPerpendicular.pdf', format='pdf')
plt.clf()


# list files in directory
file_list = os.listdir(Path_Perpendicular_701nm)
#print(file_list)
# number of files in directory
num_files = len(file_list)
for counter, fn in enumerate(file_list):
    data = pd.read_csv(Path_Perpendicular_701nm + '/' + str(fn), sep=',')
    plt.plot(data['Columns'], data['Intensity'], label=str(fn))
plt.title('701nm PSL Scattering Diagram of 663nm Light Polarized\n Perpendicular to the Scattering Plane')
plt.yscale('log')
plt.xlabel('Profile Number')
plt.ylabel('Intensity')
plt.legend(loc=1)
plt.grid()
plt.tight_layout()
#plt.show()
plt.savefig(Path_Save + '/PSL_701nm_SDPerpendicular.pdf', format='pdf')
plt.clf()
'''

