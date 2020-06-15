'''
Austen K. Scruggs
06/02/2020
Description: Plot data from datafame from NLLS_tmatrix_in_Mie_space script
'''
import pandas as pd
import matplotlib.pyplot as plt

directory = '/home/austen/Desktop/Recent/'

SLSR_DF = pd.read_csv(directory + 'SLSR_DF_NLLS_Tmat_Mie.txt', sep=',', header=0)



# make figures
f_font =18
t_font = 16
l_font = 14
f0, ax0 = plt.subplots(3, 3, figsize=(24, 18))
ax0[0, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["n"], marker='o', color='black', ls=' ', label='n')
ax0[0, 0].set_title('n vs. axis ratio', fontsize=t_font)
ax0[0, 0].set_xlabel('axis ratio', fontsize=l_font)
ax0[0, 0].set_ylabel('n', fontsize=l_font)
ax0[0, 0].grid(True)
ax0[0, 0].legend(loc=1, fontsize=l_font)
ax0[0, 1].plot(SLSR_DF["axis ratio"], SLSR_DF["k"], marker='o', color='black', ls=' ', label='k')
ax0[0, 1].set_title('k vs. axis ratio', fontsize=t_font)
ax0[0, 1].set_xlabel('axis ratio', fontsize=l_font)
ax0[0, 1].set_ylabel('k', fontsize=l_font)
ax0[0, 1].grid(True)
ax0[0, 1].legend(loc=1, fontsize=l_font)
ax0[0, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["Cext Mie"], marker='o', color='black', ls=' ', label='$C_{ext}$ Mie')
ax0[0, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["Cext Tmat"], marker='^', color='lawngreen', ls=' ', label='$C_{ext}$ Tmat')
ax0[0, 2].set_title('$C_{ext}$ vs. axis ratio', fontsize=t_font)
ax0[0, 2].set_xlabel('axis ratio', fontsize=l_font)
ax0[0, 2].set_ylabel('$C_{ext}$', fontsize=l_font)
ax0[0, 2].grid(True)
ax0[0, 2].legend(loc=1, fontsize=l_font)
ax0[1, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["Csca Mie"], marker='o', color='black', ls=' ', label='$C_{sca}$ Mie')
ax0[1, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["Csca Tmat"], marker='^', color='lawngreen', ls=' ', label='$C_{sca}$ Tmat')
ax0[1, 0].set_title('$C_{sca}$ vs. axis ratio', fontsize=t_font)
ax0[1, 0].set_xlabel('axis ratio', fontsize=l_font)
ax0[1, 0].set_ylabel('$C_{sca}$', fontsize=l_font)
ax0[1, 0].grid(True)
ax0[1, 0].legend(loc=1, fontsize=l_font)
ax0[1, 1].plot(SLSR_DF["axis ratio"], SLSR_DF["Cabs Mie"], marker='o', color='black', ls=' ', label='$C_{abs} Mie$')
ax0[1, 1].set_title('$C_{abs}$ vs. axis ratio', fontsize=t_font)
ax0[1, 1].set_xlabel('axis ratio', fontsize=l_font)
ax0[1, 1].set_ylabel('$C_{abs}$', fontsize=l_font)
ax0[1, 1].grid(True)
ax0[1, 1].legend(loc=1, fontsize=l_font)
ax0[1, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["g Mie"], marker='o', color='black', ls=' ', label='g Mie')
ax0[1, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["SR g Tmat"], marker='^', color='lawngreen', ls=' ', label='g Tmat')

ax0[1, 2].set_title('g vs. axis ratio', fontsize=t_font)
ax0[1, 2].set_xlabel('axis ratio', fontsize=l_font)
ax0[1, 2].set_ylabel('g', fontsize=l_font)
ax0[1, 2].grid(True)
ax0[1, 2].legend(loc=1, fontsize=l_font)
ax0[2, 0].plot(SLSR_DF["axis ratio"], SLSR_DF["GeoMean"], marker='o', color='black', ls=' ', label='Geometric Mean')
ax0[2, 0].set_title('Geometric Mean vs. axis ratio', fontsize=t_font)
ax0[2, 0].set_xlabel('axis ratio', fontsize=l_font)
ax0[2, 0].set_ylabel('Geometric Mean', fontsize=l_font)
ax0[2, 0].grid(True)
ax0[2, 0].legend(loc=1, fontsize=l_font)
ax0[2, 1].plot(SLSR_DF["axis ratio"], SLSR_DF["GeoStdev"], marker='o', color='black', ls=' ', label='Geometric Stdev')
ax0[2, 1].set_title('Geometric Standard Dev. vs. axis ratio', fontsize=t_font)
ax0[2, 1].set_xlabel('axis ratio', fontsize=l_font)
ax0[2, 1].set_ylabel('Geometric Stdev', fontsize=l_font)
ax0[2, 1].grid(True)
ax0[2, 1].legend(loc=1, fontsize=l_font)
ax0[2, 2].plot(SLSR_DF["axis ratio"], SLSR_DF["Cratio Mie"], marker='o', color='black', ls=' ', label='$C_{ratio}$ Mie')
ax0[2, 2].set_title('$C_{ratio}$ vs. axis ratio', fontsize=t_font)
ax0[2, 2].set_xlabel('axis ratio', fontsize=l_font)
ax0[2, 2].set_ylabel('$C_{ratio}$', fontsize=l_font)
ax0[2, 2].grid(True)
ax0[2, 2].legend(loc=1, fontsize=l_font)
f0.suptitle('Nonlinear Least Squares Retrieved vs. Axis Ratio', fontsize=f_font)
#plt.tight_layout()
plt.savefig(directory + 'NLLS_tmat_miespace_retrieval.pdf', format='pdf')
plt.savefig(directory + 'NLLS_tmat_miespace_retrieval.png', format='png')
plt.show()

