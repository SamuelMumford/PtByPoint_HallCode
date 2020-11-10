# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:54:59 2019

@author: KGB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.cm as cm

def fit(x,a,b, c): 
     return a + b*x + c*x*x

def efit(x, a, b):
    return a + b*x*x



Bcurr = np.array([0, 10, -10, -25, -37, -38, -39, -40, -39, 
                  -38, -37, -25, -10, 0, 10, 25, 37, 38, 39,
                  40, 39, 38, 37, 25, 17, -17])

Signal = np.array([3.35, 3.63, -.82, -7.48, -6.74, -6.72, -7.19, -7.59, -7.32, 
                   -6.82, -7.06, -7.18, -1.01, 4.07, 4.2, 12, 10.02, 10.86, 11.81,
                   13.17, 11.67, 11.09, 9.93, 11.32, 8.59, -3.03])

Serr = np.sqrt(2)*np.array([.15, .15, .16, .25, .15, .13, .1, .07, .1, 
              .12, .13, .27, .29, .3, .12, .25, .11, .09, .09,
              .11, .1, .09, .09, .31, .31, .56])


popt,pcov=curve_fit(fit, Bcurr, Signal, p0=(0.0,0.0, 0.0), sigma = Serr)
#popt2,pcov2=curve_fit(fit, FlipBcurr, FlipSignal, p0=(0.0,0.0), sigma = FlipSerr)
print('Original data (black)')
print('Fit Params')
print(popt)
print('Covariance')
print(pcov)
print('Fit Std. Dev.')
print(np.sqrt(np.diag(pcov)))
print('Fit in Std. Dev.')
print(popt/np.sqrt(np.diag(pcov)))
#print('Flip lead data (red)')
#print(popt2)
#print(pcov2)
preds = fit(Bcurr, popt[0], popt[1], popt[2])
#preds2 = fit(FlipBcurr, popt2[0], popt2[1])


diff = Signal - preds
#diff2 = FlipSignal - preds2

pltXs = np.linspace(min(Bcurr), max(Bcurr))
pltYs = fit(pltXs, popt[0], popt[1], popt[2])
#pltYs2 = fit(pltXs, popt2[0], popt2[1])
colors = cm.jet(np.linspace(0, 1, len(Signal)))
for i in range(0, len(Signal)):
    plt.errorbar(Bcurr[i], Signal[i], Serr[i], marker = 'o', mfc=colors[i], ecolor=colors[i], linewidth = 0, elinewidth=1)
plt.plot(pltXs, pltYs, c = 'black')
plt.title('Signal vs Magnet Current')
plt.legend(loc = 2)
plt.ylabel('Signal (mHz)')
plt.xlabel('Magnet Current (Amps)')
plt.show()

for i in range(0, len(Signal)):
    plt.errorbar(Bcurr[i], Signal[i]- preds[i], Serr[i], marker = 'o', mfc=colors[i], ecolor=colors[i], linewidth = 0, elinewidth=1)
plt.axhline(0, c = 'black')
#plt.plot(pltXs, 0*pltYs)
plt.title('Residuals')
plt.ylabel('Residual (mHz)')
plt.xlabel('Magnet Current (Amps)')
plt.show()

BsDict = {}
for i in range(0, len(Bcurr)):
    if(Bcurr[i] in BsDict.keys()):
        t = BsDict[Bcurr[i]]
        t[0] += Signal[i]
        t[1] += 1
    else:
        BsDict[Bcurr[i]] = [Signal[i], 1]
print(BsDict)
for i in range(0, len(Signal)):
    t = BsDict[Bcurr[i]]
    plt.errorbar(Bcurr[i], Signal[i] - t[0]/t[1], Serr[i], marker = 'o', mfc=colors[i], ecolor=colors[i], linewidth = 0, elinewidth=1)
plt.axhline(0, c = 'black')
#plt.plot(pltXs, 0*pltYs)
plt.title('Point Spread')
plt.ylabel('Difference from Mean (mHz)')
plt.xlabel('Magnet Current (Amps)')
plt.show()


order = np.linspace(0, len(preds)-1, len(preds))
for i in range(0, len(Signal)):
    plt.errorbar(order[i], Signal[i]- preds[i], Serr[i], marker = 'o', mfc=colors[i], ecolor=colors[i], linewidth = 0, elinewidth=1)
plt.axhline(0, c = 'black')
plt.title('Residuals')
plt.ylabel('Residual (mHz)')
plt.xlabel('Order of Data')
plt.show()

for i in range(0, len(Signal)):
    plt.errorbar(order[i], (Signal[i]- preds[i])/Serr[i], 1, marker = 'o', mfc=colors[i], ecolor=colors[i], linewidth = 0, elinewidth=1)
plt.axhline(0, c = 'black')
plt.title('Residuals')
plt.ylabel('Residual (std dev)')
plt.xlabel('Order of Data')
plt.show()

evenonly = Signal - (popt[0] + popt[1]*Bcurr)
evenonly2 = popt[2]*pltXs**2
#plt.scatter(Bcurr, evenonly, c = 'blue', s=10, linewidth = 0,  zorder=1)
for i in range(0, len(Signal)):
    plt.errorbar(Bcurr[i], evenonly[i], Serr[i], marker = 'o', mfc=colors[i], ecolor=colors[i], linewidth = 0, elinewidth=1)
plt.plot(pltXs, evenonly2, c = 'red')
plt.title('Hall Signal vs Magnet Current')
plt.ylabel('B^2 Signal (mHz)')
plt.xlabel('Magnet Current (Amps)')
plt.show()

print("Difference from Fit of " + str(np.sqrt(np.mean(((Signal - preds)/Serr)**2))) + " std. dev.")

RunCurrs = Bcurr
RunShifts = Signal
RunSigs = Serr
FieldCounts = np.zeros((0, 3))
ShiftSums = np.zeros((0, 3))
varry = np.zeros((0, 3))
for i in range(len(RunCurrs)):
    value = np.abs(RunCurrs[i])
    sign = np.sign(RunCurrs[i])
    if value not in FieldCounts[:, 0]:
        if(sign > 0):
            FieldCounts = np.vstack((FieldCounts, [value, 1, 0]))
            ShiftSums = np.vstack((ShiftSums, [value, RunShifts[i], 0]))
            varry = np.vstack((varry, [value, RunSigs[i]**2, 0]))
        if(sign < 0):
            FieldCounts = np.vstack((FieldCounts, [value, 0, 1]))
            ShiftSums = np.vstack((ShiftSums, [value, 0, RunShifts[i]]))
            varry = np.vstack((varry, [value, 0, RunSigs[i]**2]))
        if(sign == 0):
            FieldCounts = np.vstack((FieldCounts, [value, .5, .5]))
            ShiftSums = np.vstack((ShiftSums, [value, .5*RunShifts[i], .5*RunShifts[i]]))
            varry = np.vstack((varry, [value, .25*RunSigs[i]**2, .25*RunSigs[i]**2]))
    else:
        location = np.where(FieldCounts[:, 0] == value)
        if(sign > 0):
            FieldCounts[location, 1] += 1
            ShiftSums[location, 1] += RunShifts[i]
            varry[location, 1] += RunSigs[i]**2
        if(sign < 0):
            FieldCounts[location, 2] += 1
            ShiftSums[location, 2] += RunShifts[i]
            varry[location, 2] += RunSigs[i]**2
        if(sign == 0):
            FieldCounts[location, 1] += .5
            ShiftSums[location, 1] += .5*RunShifts[i]
            varry[location, 1] += .25*RunSigs[i]**2
            FieldCounts[location, 2] += .5
            ShiftSums[location, 2] += .5*RunShifts[i]
            varry[location, 2] += .25*RunSigs[i]**2


ShiftAvg = ShiftSums
ShiftAvg[:, 1] = np.divide(ShiftSums[:, 1], FieldCounts[:, 1])
ShiftAvg[:, 2] = np.divide(ShiftSums[:, 2], FieldCounts[:, 2])
print(ShiftAvg)
varry[:, 1] = np.divide(varry[:, 1], FieldCounts[:, 1]**2)
varry[:, 2] = np.divide(varry[:, 2], FieldCounts[:, 2]**2)
#print(ShiftAvg)
ShiftEven = np.zeros(len(ShiftAvg))
SigFin = np.zeros(len(ShiftAvg))
for i in range(len(ShiftAvg)):
    ShiftEven[i] = (ShiftAvg[i, 1] + ShiftAvg[i, 2])/2
    SigFin[i] = np.sqrt(varry[i, 1] + varry[i, 2])/2
print(ShiftEven)
epopt,epcov=curve_fit(efit, .875*ShiftAvg[:, 0]/5, ShiftEven, p0=(0.0,0.0), sigma = SigFin)
print('Original data (black)')
print(epopt)
print(epcov)
print(np.sqrt(np.diag(epcov)))

epreds = efit(.875*ShiftAvg[:, 0]/5, epopt[0], epopt[1])
print("Std. Dev.")
print(np.mean(np.abs(ShiftEven - epreds)/SigFin))
epltXs = np.linspace(min(ShiftAvg[:, 0]), max(ShiftAvg[:, 0]))
epltYs = efit(.875*epltXs/5, 0, epopt[1])
#plt.plot(.875*epltXs/5, epltYs, c = "red")
plt.errorbar(.875*ShiftAvg[:, 0]/5, ShiftEven, SigFin, c = 'red', marker = 'o', linewidth = 0, elinewidth=1, label = "Cooldown 1", capsize = 2.5)

plt.ylabel('Even $\delta f_{0}(\pm V)$ (mHz)')
plt.xlabel('Amplitude of Magnetic Field (T)')
#plt.legend(numpoints=1, loc = 2)
plt.xlim(-.5, 7)
plt.tight_layout()
#plt.savefig("EvenPartField.pdf")
plt.show()

plt.errorbar(.875*ShiftAvg[:, 0]/5, ShiftEven - ShiftEven[0], SigFin, c = 'red', marker = 'o', linewidth = 0, elinewidth=1, label = "Cooldown 1", capsize = 2.5)

plt.ylabel('Even $\delta f_{0}(\pm V)$ (mHz)')
plt.xlabel('Amplitude of Magnetic Field (T)')
#plt.legend(numpoints=1, loc = 2)
plt.xlim(-.5, 7)
plt.tight_layout()
#plt.savefig("EvenPartField.pdf")
plt.show()