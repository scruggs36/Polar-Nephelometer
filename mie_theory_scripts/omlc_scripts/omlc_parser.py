import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# file path
path = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Mie-Theory/data/450nm_Nigrosin_data.txt'

# open file
file = open(path, 'r')

# setting up empty arrays
array = []
params = []
data = []

# appending each line in the data file to the empty data array
for line in file:
    array.append([line])

# storing removed lines in empty array called params, and returning the remaining lines in 2d list by calling data
# pop stores removed index in data.pop(index) and the 2d list is returned without the removed index by calling data
# here we remove the first line, 36 times, essentially removing the first 36 lines and appending them to separate array, thus separating data and parameters
for element in range(36):
    popped = array.pop(0)
    params.append(popped)

# here you can see params and array if you need to
#print(params)
#print(array)

# del just removes the specified index but doesn't store what was removed
# for example this removes elements in list by their index, works also with a range of indices so no loop required: del data[0:36]
# remove() only removes the specific value, does not work by index: a = [1,2,3] a.remove(1) --> a=[2,3]

# here we remove #, \n, and delimit by \t
for list in array:
    for element in list:
        element = element.strip('#')
        element = element.strip('\n')
        element = element.split('\t')
        data.append(element)

# here you can view the data in array of strings form if you need to
#print(data)

# here we remove the first row of the ndarray and store it for as headers for the dataframe we create
headers = data.pop(0)
headers[4] = 'natural normalized'
headers[5] = 'perpen normalized'
headers[6] = 'parallel normalized'
data = pd.DataFrame(data, columns=headers, dtype=float)

# here you can view the dataframe if you want to
#print(data)

f, ax = plt.subplots(3)
ax[0].plot(data['theta'], data['natural normalized'], 'r-', label='norm. natural PF')
ax[0].set_title('Normalized Natural Phase Function of  Nigrosin Particles of size 650nm at a Wavelength of 663nm')
ax[0].set_xlabel('$\Theta$')
ax[0].set_ylabel('Normalized Intensity')
ax[0].legend(loc=1)
ax[1].plot(data['theta'], data['perpen normalized'], 'b-', label='norm. perpendicular PF')
ax[1].set_title('Normalized Perpendicular Phase Function of  Nigrosin Particles of size 650nm at a Wavelength of 663nm')
ax[1].set_xlabel('$\Theta$')
ax[1].set_ylabel('Normalized Intensity')
ax[1].legend(loc=1)
ax[2].plot(data['theta'], data['parallel normalized'], 'g-', label='norm. parallel PF')
ax[2].set_title('Normalized Parallel Phase Function of  Nigrosin Particles of size 650nm at a Wavelength of 663nm')
ax[2].set_xlabel('$\Theta$')
ax[2].set_ylabel('Normalized Intensity')
ax[2].legend(loc=1)
plt.tight_layout()
plt.show()

# save dataframe
#save_path = 'C:/Users/sm2/Documents/Austen/Github Repository Clone/Mie-Theory/data/PF_Size100nm_Wav663nm'
#data.to_csv(save_path)
print(data)
print(np.sqrt(np.sum(np.square(data['perpen normalized']))))
