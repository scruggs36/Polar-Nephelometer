import numpy as np

FlowRateLPM = 1.50
AvgTimeMins = 60
PNephVol = (np.pi * ((2 * 2.54) ** 2) * (4.0 * 12 * 2.54))/1000
print('Nephelometer chamber volume in liters: ', PNephVol)
CylinderPressurePSI_LQ = 50
CylinderVolumeLiters_LQ = 240
LiquidNitrogenDensity = 0.807 * 1000
NitrogenMolarMass = 28.014
CylinderPressurePSI = 250
CylinderVolumeLiters = 43.2

def Volume_Calculation(CylinderPSI, CylinderVol):
    AtmPSI = 14.7
    V1 = ((CylinderPSI * CylinderVol)/AtmPSI)
    return V1


NitrogenMoles = (CylinderVolumeLiters_LQ * LiquidNitrogenDensity)/(NitrogenMolarMass)
CylinderGasVolume_LQ = ((NitrogenMoles * 0.0821 * 298)/(1)) + Volume_Calculation(CylinderPressurePSI_LQ, CylinderVolumeLiters_LQ)

TotalTimeNeeded = PNephVol/FlowRateLPM + AvgTimeMins
TotalVolumeConsumed = (TotalTimeNeeded * FlowRateLPM)
PercentCylinderVolumeConsumed = (TotalVolumeConsumed/Volume_Calculation(CylinderPressurePSI, CylinderVolumeLiters))*100
PercentCylinderVolumeConsumed_LQ = (TotalVolumeConsumed/CylinderGasVolume_LQ)*100

print('For regular a 43.2 Nitrogen Gas Cylinder: ')
print('The chamber purge time and image averaging together in minutes will be: ', TotalTimeNeeded)
print('The current volume of gas one can extract from the cylinder in liters is: ', Volume_Calculation(CylinderPressurePSI, CylinderVolumeLiters))
print('The total volume you will consume in your experiment in liters is: ', TotalVolumeConsumed)
print('You will be consuming this percentage of the remaining gas volume left in the cylinder: ', PercentCylinderVolumeConsumed)

print('For Liquid Nitrogen Gas Cylinder: ')
print('The chamber purge time and image averaging together in minutes will be: ', TotalTimeNeeded)
print('The current volume of gas one can extract from the cylinder in liters is: ', CylinderGasVolume_LQ)
print('The total volume you will consume in your experiment in liters is: ', TotalVolumeConsumed)
print('You will be consuming this percentage of the remaining gas volume left in the cylinder: ', PercentCylinderVolumeConsumed_LQ)


