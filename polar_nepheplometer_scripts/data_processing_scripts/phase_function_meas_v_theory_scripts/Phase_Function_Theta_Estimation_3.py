import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate
import statsmodels.api as sm
from scipy.optimize import curve_fit
from statsmodels.sandbox.regression.predstd import wls_prediction_std


#'''
#import measured scattering diagram and normalize
SD_Path = '/home/austen/Documents/PSL_600nm_SD_avg_30s.txt'
SD_Data = pd.read_csv(SD_Path, delimiter=',', header=0)
SD = np.asarray(SD_Data['SD Particle'])
SD_PN = np.asarray(SD_Data['Profile Number'])
SD_Normalized = SD / np.linalg.norm(SD, axis=0)

# import calculated scattering diagram and normalize
#SD_Average = np.asarray(SD.mean(axis=0))
SD_STDEV = np.full(len(SD), 5)
#'''




#'''
#compare phase functions
Mie_Path = '/home/austen/Documents/S1_Wavelength663nm_Size600nm.txt'
Theta_Guess = np.linspace(0, 90, 100)
Mie_Data = pd.read_csv(Mie_Path, delimiter=',', header=0)
Mie_Theta = np.asarray(Mie_Data['angles'])
Mie_Int = np.asarray(Mie_Data['S1^2'])
Mie_Int_Normalized = Mie_Int / np.linalg.norm(Mie_Int, axis=0)



f5, ax5 = plt.subplots()
ax5.plot(Mie_Theta[50:125], Mie_Int_Normalized[50:125], 'r--', label='600nm PSL Perp. SD')
ax5.set_title('Scattering Diagram Focused at a Particular Range of Scattering Angles')
ax5.set_xlabel('Scattering Angle (\u00b0)')
ax5.set_ylabel('Normalized Intensity')
ax5.set_yscale('log')
plt.legend(loc=1)
plt.show()
#'''

def linear(x, m, b):
    return m * np.array(x).astype(int) + b

#print(PF_2_Mie['parallel normalized'])
# estimating Theta by finding the minimum in the squared differences array
def Theta_Estimation( Mie_Theta_Array, Mie_Intensity_Array, Experiment_Intensity_Array, Theta_Hi_Res, dTheta_dProfile, Prof_Num, PF_Stdev):
    f_spline = interpolate.interp1d(Mie_Theta_Array, Mie_Intensity_Array, kind='cubic')
    Intensity_spline = f_spline(Theta_Hi_Res)
    f_pchip = interpolate.PchipInterpolator(Mie_Theta_Array, Mie_Intensity_Array, axis=0)
    Intensity_pchip = f_pchip(Theta_Hi_Res)
    Min_Index_Array_Raw = []
    Min_Index_Array_Spline = []
    Min_Index_Array_Pchip = []
    Theta_Estimated_Raw = []
    Theta_Estimated_Spline = []
    Theta_Estimated_Pchip = []
    Residuals_Raw =[]
    Residuals_Spline = []
    Residuals_Pchip = []
    Intensity_Raw = []
    Intensity_Spline = []
    Intensity_Pchip = []
    Prof_Num_Raw = []
    Prof_Num_Spline = []
    Prof_Num_Pchip = []
    PF_Stdev_Raw = []
    PF_Stdev_Spline = []
    PF_Stdev_Pchip = []
    # theta estimation from raw Mie data
    for counter, element in enumerate(Experiment_Intensity_Array):
        Squared_Differences_Raw = [(x - element) ** 2 for x in Mie_Intensity_Array]
        if counter == 0:
            Min_Index = np.argmin(Squared_Differences_Raw)
            Min_Index_Array_Raw.append(Min_Index)
            Residuals_Raw.append(Squared_Differences_Raw[int(Min_Index)])
            Theta_Raw = Mie_Theta_Array[int(Min_Index)]
            Theta_Estimated_Raw.append(Theta_Raw)
            Intensity_Raw.append(element)
            Prof_Num_Raw.append(Prof_Num[counter])
            PF_Stdev_Raw.append(PF_Stdev[counter])
        if counter != 0:
            Min_Index = np.argmin(Squared_Differences_Raw)
            Theta_Raw = Mie_Theta_Array[int(Min_Index)]
            if Theta_Raw > Theta_Estimated_Raw[-1] + dTheta_dProfile:
                Residuals_Raw.append(Squared_Differences_Raw[int(Min_Index)])
                Min_Index_Array_Raw.append(Min_Index)
                Theta_Estimated_Raw.append(Theta_Raw)
                Intensity_Raw.append(element)
                Prof_Num_Raw.append(Prof_Num[counter])
                PF_Stdev_Raw.append(PF_Stdev[counter])
            else:
                continue
    # theta estimation from spline interpolation on Mie data
    for counter, element in enumerate(Experiment_Intensity_Array):
        Squared_Differences_Spline = [(x - element) ** 2 for x in Intensity_spline]
        if counter == 0:
            Min_Index = np.argmin(Squared_Differences_Spline)
            Min_Index_Array_Spline.append(Min_Index)
            Residuals_Spline.append(Squared_Differences_Spline[int(Min_Index)])
            Theta_Spline = Theta_Hi_Res[int(Min_Index)]
            Theta_Estimated_Spline.append(Theta_Spline)
            Intensity_Spline.append(element)
            Prof_Num_Spline.append(Prof_Num[counter])
            PF_Stdev_Spline.append(PF_Stdev[counter])
        if counter != 0:
            Min_Index = np.argmin(Squared_Differences_Spline)
            Theta_Spline = Theta_Hi_Res[int(Min_Index)]
            if Theta_Spline > Theta_Estimated_Spline[-1] + dTheta_dProfile:
                Residuals_Spline.append(Squared_Differences_Spline[int(Min_Index)])
                Min_Index_Array_Spline.append(Min_Index)
                Theta_Estimated_Spline.append(Theta_Spline)
                Intensity_Spline.append(element)
                Prof_Num_Spline.append(Prof_Num[counter])
                PF_Stdev_Spline.append(PF_Stdev[counter])
            else:
                continue
        # theta estimation from pchip interpolation on Mie data
    for counter, element in enumerate(Experiment_Intensity_Array):
        Squared_Differences_Pchip = [(x - element) ** 2 for x in Intensity_pchip]
        if counter == 0:
            Min_Index = np.argmin(Squared_Differences_Pchip)
            Min_Index_Array_Pchip.append(Min_Index)
            Residuals_Pchip.append(Squared_Differences_Pchip[int(Min_Index)])
            Theta_Pchip = Theta_Hi_Res[int(Min_Index)]
            Theta_Estimated_Pchip.append(Theta_Pchip)
            Intensity_Pchip.append(element)
            Prof_Num_Pchip.append(Prof_Num[counter])
            PF_Stdev_Pchip.append(PF_Stdev[counter])
        if counter != 0:
            Min_Index = np.argmin(Squared_Differences_Pchip)
            Theta_Pchip = Theta_Hi_Res[int(Min_Index)]
            if Theta_Pchip > Theta_Estimated_Pchip[-1] + dTheta_dProfile:
                Residuals_Pchip.append(Squared_Differences_Pchip[int(Min_Index)])
                Min_Index_Array_Pchip.append(Min_Index)
                Theta_Estimated_Pchip.append(Theta_Pchip)
                Intensity_Pchip.append(element)
                Prof_Num_Pchip.append(Prof_Num[counter])
                PF_Stdev_Pchip.append(PF_Stdev[counter])
            else:
                continue
    return(Theta_Estimated_Raw, Theta_Estimated_Spline, Theta_Estimated_Pchip, Intensity_Raw, Intensity_Spline, Intensity_Pchip, Residuals_Raw, Residuals_Spline, Residuals_Pchip, Prof_Num_Raw, Prof_Num_Spline, Prof_Num_Pchip, PF_Stdev_Raw, PF_Stdev_Spline, PF_Stdev_Pchip)


# setting up data I want to evaluate
Theta_HiRes = np.arange(0.0, 100.1, 0.1)
#print(Theta_HiRes)

Mie_Intensity_Perpen = np.asarray(Mie_Int_Normalized[89:180])

Mie_Theta_Perpen = Mie_Theta[89:180]
Exp_Intensity = SD_Normalized[12:100]
#PF_stdev = np.asarray(PF_STDEV[12:100])
Profile_Number = SD_PN[12:100]


# calling theta estimation and curve fitting
dtheta_dprof = 0.814
Theta_raw, Theta_spline, Theta_pchip, Int_raw, Int_spline, Int_pchip, residuals_raw, residuals_spline, residuals_pchip, prof_raw, prof_spline, prof_pchip, pf_stdev_raw, pf_stdev_spline, pf_stdev_pchip = Theta_Estimation(Mie_Theta, Mie_Int,  Exp_Intensity, Theta_HiRes, dtheta_dprof, Profile_Number, 5)


#conducting least squares minimizations, curve_fit is a least squares minimization
popt_spline, pcov_spline = curve_fit(linear, prof_spline, Theta_spline)
X_spline = sm.add_constant(prof_spline)
model_spline =sm.OLS(Theta_spline, X_spline)
result_spline = model_spline.fit()
prstd_spline, iv_l_spline, iv_u_spline =  wls_prediction_std(result_spline)
#print(result_spline.summary())


popt_pchip, pcov_pchip = curve_fit(linear, prof_pchip, Theta_pchip)
X_pchip = sm.add_constant(prof_pchip)
model_pchip =sm.OLS(Theta_spline, X_pchip)
result_pchip = model_spline.fit()
prstd_pchip, iv_l_pchip, iv_u_pchip =  wls_prediction_std(result_pchip)
print(result_pchip.summary())


# Make sure your arrays are the same length
#print(len(Theta_spline))
#print(len(Int_spline))
#print(len(pf_stdev_spline))
#print(len(pf_stdev_pchip))



# plotting
f6, ax6 = plt.subplots(2,2)
#ax6.plot(Mie_Theta_Perpen, Mie_Intensity_Perpen, 'r--', label='Nigrosin Mie Perpendicular PF')
#ax6[0, 0].plot(Mie_Theta_Parallel, Mie_Intensity_Parallel, 'b--', label='Nigrosin Mie Parallel PF')
#ax6[0].errorbar(Theta_raw, Exp_Intensity, yerr=PF_stdev, fmt='co', markersize=4, label='Nigrosin PF Theta SD Raw')
ax6[0, 0].errorbar(Theta_spline, Int_spline, yerr=pf_stdev_spline, xerr=np.repeat(1.96 * np.sqrt(np.diag(pcov_spline))[1], len(pf_stdev_spline)), fmt='mo', markersize=4, label='Nigrosin PF Theta SD Spline')
ax6[0, 0].errorbar(Theta_pchip, Int_pchip, yerr=pf_stdev_pchip, xerr=np.repeat(1.96 * np.sqrt(np.diag(pcov_pchip))[1] , len(pf_stdev_pchip)), fmt='ko', markersize=4, label='Nigrosin PF Theta SD Pchip')
ax6[0, 0].set_title('Comparison of Phase Functions')
ax6[0, 0].set_xlabel('$\Theta$')
ax6[0, 0].set_ylabel('Integrated Intensity')
ax6[0, 0].legend(loc=1)
#ax6[1].plot(Theta_raw, residuals_raw, 'co', markersize=4, label='Raw SD Residuals')
ax6[0, 1].plot(Theta_spline, residuals_spline, 'mo', markersize=4, label='Spline SD Residuals')
ax6[0, 1].plot(Theta_pchip, residuals_pchip, 'ko', markersize=4, label='Pchip SD Residuals')
ax6[0, 1].set_title('Residuals as a Function of Theta')
ax6[0, 1].set_xlabel('$\Theta$')
ax6[0, 1].set_ylabel('Integrated Intensity Residual')
ax6[0, 1].legend(loc=1)
ax6[1, 0].plot(prof_spline, Theta_spline, 'ko', markersize=4, label='Profile Number vs Theta from Spline')
ax6[1, 0].plot(prof_spline, linear(prof_spline, *popt_spline), 'r-', markersize=4, label=' Fit: '+'y = '+str(popt_spline[0])+'x + '+str(popt_spline[1]))
ax6[1, 0].plot(prof_spline, iv_u_spline, 'b--', markersize=4, label=' Fit Upper Bound')
ax6[1, 0].plot(prof_spline, iv_l_spline, 'b--', markersize=4, label=' Fit Lower Bound')
ax6[1, 0].set_title('Linear Fit Profiles vs Theta')
ax6[1, 0].set_xlabel('Profile Number')
ax6[1, 0].set_ylabel('$\Theta$')
ax6[1, 0].legend(loc=1)
ax6[1, 1].plot(prof_pchip, Theta_pchip, 'ko', markersize=4, label='Profile Number vs Theta from Pchip')
ax6[1, 1].plot(prof_pchip, linear(prof_pchip, *popt_pchip), 'r-', markersize=4, label=' Fit: ' 'y = '+str(popt_pchip[0])+'x + '+str(popt_pchip[1]))
ax6[1, 1].plot(prof_pchip, iv_u_pchip, 'b--', markersize=4, label=' Fit Upper Bound')
ax6[1, 1].plot(prof_pchip, iv_l_pchip, 'b--', markersize=4, label=' Fit Lower Bound')
ax6[1, 1].set_title('Linear Fit Profiles vs Theta')
ax6[1, 1].set_xlabel('Profile Number')
ax6[1, 1].set_ylabel('$\Theta$')
ax6[1, 1].legend(loc=1)
plt.tight_layout()
plt.show()


print(Theta_pchip)

'''
f7, ax7 = plt.subplots()
ax7.plot(PF_1_Mie['theta'][90:180], PF_1_Mie['perpen normalized'][90:180], 'r--', label='Nigrosin Mie Perpendicular PF')
#ax7.errorbar(Theta_Guess[13:100], PF_Average[13:100], yerr=PF_STDEV[13:100], fmt='go', markersize=4, label='Nigrosin Measured PF')
ax7.errorbar(linear(Profile_Number, *popt_pchip), Exp_Intensity, yerr=PF_stdev, fmt='c*', markersize=4, label='Theta Linear Fit')
ax7.set_title('Comparison of Phase Functions')
ax7.set_xlabel('Profile Number')
ax7.set_ylabel('Integrated Intensity')
plt.legend(loc=1)
plt.show()
'''