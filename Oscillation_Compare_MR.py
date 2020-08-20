#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:14:43 2020
@author: tiffanypaul
"""


import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt

#Define a polynomial fit in B and the run index for the temperature value
#that we read out    
def TFit(data, Bcs, Rcs):
    runs = data[:, 0]
    Bs = data[:, 1]
    
    fit = np.zeros(len(runs))
    i = 2
    #The m's are the fit coefficients, doing only an even fit in B
    for m in zip(Bcs):
        fit += m*np.power(Bs, i)
        i += 2
    #n's are fit coefficients, allow for an offset term in the run index fit
    j = 0
    for n in zip(Rcs):
        fit += n*np.power(runs, j)
    return fit
    
#When you call the curve_fit code, it can ONLY take things of the form 
#function(x, args). We need to make a wrapping function to put TFit in 
#that form.        
def wrapT(x, *args):
    #Note that NTs is a global variable telling which indexes separate
    #the B and run fit coefficients
    polyBs = list(args[:NTs[0]])
    polyRs = list(args[NTs[0]:])
    return TFit(x, polyBs, polyRs)

fname = ['C:/Users/sammy/Downloads/3_15_20_edit.txt', 
         'C:/Users/sammy/Downloads/Edited_cooldown_all_7_25.txt',
         'C:/Users/sammy/Downloads/Edited_cooldown_all_8_16_20.txt']

f = ['C:/Users/sammy/Downloads/3_15_20_editL.txt', 
         'C:/Users/sammy/Downloads/Edited_cooldown_all_7_25L.txt',
         'C:/Users/sammy/Downloads/Edited_cooldown_all_8_16_20L.txt']
startcut = [0, 50, 150]
endcut = [4600, -70, 4500]
cut = [0.85, 0.85, 0.85]
datlen = np.zeros(len(fname))
dataAllLen = np.zeros(len(fname))

Blist_all = np.zeros(0)
OscAmp_all = np.zeros(0)
OscPeriod_all = np.zeros(0)
MeanTemp_all = np.zeros(0)
MT_all = np.zeros(0)

BInd = 0
BFile = fname[BInd]
FdatB = np.loadtxt(BFile, delimiter="\t")
scB = startcut[BInd]
endcutB = endcut[BInd]
cutEndB = True
if(cutEndB):
    #Two endcuts have been used. 5600 gives you more or less the whole file
    #Before the temperature gets out of hand. 4520 is all the data before
    #the .2K rise
    endcutB = 4520
    rb = FdatB[scB:endcutB, 3]
    tb = FdatB[scB:endcutB, 4]
    cb = FdatB[scB:endcutB, 5]
else:
    rb = FdatB[scB:, 3]
    tb = FdatB[scB:, 4]
    cb = FdatB[scB:, 5]
Bsb = cb*.85/5
runsTemp = rb
tempsTemp = tb
BsTemp = Bsb
 #The order of the even in B polynomial fit and run index fits
BOrd = 5
ROrd = 1
#We use NTs in wrapT, so we make it a global variable. Gives the index
#of fit variables of different types
global NTs
NTs = ([BOrd, BOrd + ROrd])
pT = np.zeros(BOrd + ROrd)
#Have a run index after which temperature rises
tendCut = 3570
tstartCut = 170
#Find that index in the data arrays
place = np.where(runsTemp == tendCut)[0]
print(runsTemp)
#Keep only data before the cutoff index
pT[BOrd] = np.mean(tempsTemp[tstartCut:place[0]])
#Stack the runs and B data so that it can be passed as one variable to the fit function
RandB = np.vstack((runsTemp[tstartCut:place[0]], BsTemp[tstartCut:place[0]])).T
#Call the fit function, this syntax is terrible but works. lambda essentially 
#defines a new function that we can pass as our fit function
popt, pcov = curve_fit(lambda RandB, *pT: wrapT(RandB, *pT), RandB, tempsTemp[tstartCut:place[0]], p0=pT)
#if you want to see fit results for temperature, uncomment here
#    print(popt)
#    print(popt/np.sqrt(np.diag(pcov
    
#In order to get the true temperature, we need to subtract off the magnetic 
#field part. So define a parameter array where the run-dependent fit parts are 0
#and the magnetic field dependent parts are the results of the fit
pSub = np.zeros(len(pT))
pSub[:BOrd] = popt[:BOrd]

for h in range(0, len(fname)):

    Fdat1 = np.loadtxt(fname[h], delimiter="\t")
    F1 = np.loadtxt(f[h], delimiter="\t")

    temp1 = Fdat1[startcut[h]:endcut[h], -2]
    runs1 = Fdat1[startcut[h]:endcut[h], 3]
    currs1 = Fdat1[startcut[h]:endcut[h], 5]
    Bs1 = currs1*.85/5 
    TandBFull = np.vstack((runs1, Bs1)).T
    temp1 = temp1 - wrapT(TandBFull, *pSub)
    
    #Define fit function
    def sinfit(subruns1, a, b, c, d):
        return a * np.sin(b*subruns1 + c) + d
    
    #Find list of B fields
    Blist1 = np.zeros(0)

    good = np.zeros(0)
    
    startindex = 0
    sgsize = 1
    OscAmp = np.zeros(0)
    OscFreq = np.zeros(0)
    MeanTemp = np.zeros(0)
    
    for i in range(0, len(Bs1)-1):

        if(Bs1[i] != Bs1[i+1]):
            Blist1 = np.append(Blist1, np.mean(Bs1[startindex:i+1]))
            skip= 20
            startindex = startindex + skip
            subtemp1 = temp1[startindex:(i+1)]
            subruns1 = runs1[startindex:(i+1)]
            
            smooth = signal.savgol_filter(subtemp1, 2*sgsize+1, 1)
            
            av = np.zeros(len(subtemp1))
            av = av + np.mean(subtemp1)
            
            zero_cross = np.where(np.diff(np.sign(subtemp1 - av)))[0]
            
            switchsign = len(zero_cross)
            osc = switchsign/2
            howmanyruns = len(subtemp1)
            freqguess = osc/howmanyruns
            
            p0 = np.zeros(4)
            p0[0] = (max(subtemp1) - min(subtemp1))/2
            p0[1] = freqguess*2*np.pi
            p0[2] = 30
            p0[3] = np.mean(subtemp1)
            
            tfit, tfit_cov = curve_fit(sinfit, subruns1, smooth, p0=p0)
            
            OscAmp = np.append(OscAmp, tfit[0])
            OscFreq = np.append(OscFreq, tfit[1]/(2*np.pi))   
            MeanTemp = np.append(MeanTemp, np.mean(subtemp1))
            
            tfit_plot = (tfit[0] * np.sin(tfit[1]*subruns1 + tfit[2]) + tfit[3])
            
            plt.scatter(subruns1, subtemp1, c = 'blue', s = 2, linewidth = 0)
            plt.plot(subruns1, smooth, c = 'black')
            plt.plot(subruns1, tfit_plot, c = 'red')
            plt.xlabel('Run Index')
            plt.ylabel('Temperature (K)')
            plt.title('Temperature vs Run Index')
            plt.show()
            
            fitdif = np.mean(abs(subtemp1 - np.mean(subtemp1)))/abs(tfit[0])
            good = np.append(good, fitdif)
            print('fitdif')
            print(fitdif)
            startindex = i+1
            
    Blist1 = np.append(Blist1, np.mean(Bs1[startindex:]))   
    
    startindex = startindex+20
    
    subtemp1 = temp1[startindex:i+1]
    subruns1 = runs1[startindex:i+1]
    
    smooth = signal.savgol_filter(subtemp1, 2*sgsize+1, 1)
    
    p0 = np.zeros(4)
    p0[0] = (max(subtemp1) - min(subtemp1))/2
    p0[1] = 0.3
    p0[2] = 30
    p0[3] = np.mean(subtemp1)
    
    tfit, tfit_cov = curve_fit(sinfit, subruns1, smooth, p0=p0)
    
    OscAmp = np.append(OscAmp, tfit[0])
    OscFreq = np.append(OscFreq, tfit[1]/(2*np.pi))
    MeanTemp = np.append(MeanTemp, np.mean(subtemp1))
    
    tfit_plot = (tfit[0] * np.sin(tfit[1]*subruns1 + tfit[2]) + tfit[3])
    
    plt.scatter(subruns1, subtemp1, c = 'blue', s = 2, linewidth = 0)
    plt.plot(subruns1, smooth, c = 'black')
    plt.plot(subruns1, tfit_plot, c = 'red')
    plt.xlabel('Run Index')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature vs Run Index')
    plt.show()    
    
    fitdif = np.mean(abs(subtemp1 - np.mean(subtemp1)))/abs(tfit[0])
    print('fitdif')
    print(fitdif)
    good = np.append(good, fitdif)

    OscPeriod = 1/OscFreq
    print('good')
    print(good)
    
    keep = (good<cut[h])
    print(keep)
    l0 = len(Blist1)
    Blist1 = Blist1[keep]
    MeanTemps = MeanTemp[keep]
    OscAmp = OscAmp[keep]
    OscFreq = OscFreq[keep]

    OscPeriod = OscPeriod[keep]
    
    Blist_all = np.hstack((Blist_all, Blist1))
    OscAmp_all = np.hstack((OscAmp_all, OscAmp))
    OscPeriod_all = np.hstack((OscPeriod_all, OscPeriod))
    MeanTemp_all = np.hstack((MeanTemp_all, MeanTemps))
    MT_all = np.hstack((MT_all, MeanTemp))
    
    datlen[h] = len(Blist1)
    dataAllLen[h] = l0

stop1 = int(datlen[0])
start2 = int(datlen[0])
stop2 = int(datlen[0]+datlen[1])
start3 = int(datlen[0]+datlen[1])

sto1 = int(dataAllLen[0])
sta2 = int(dataAllLen[0])
sto2 = int(dataAllLen[0] + dataAllLen[1])
sta3 = int(dataAllLen[0] + dataAllLen[1])

AllMT1 = MT_all[0:sto1]
AllMT2 = MT_all[sta2:sto2]
AllMT3 = MT_all[sta3:]

MT1 = MeanTemp_all[0:stop1]
MT2 = MeanTemp_all[start2:stop2]
MT3 = MeanTemp_all[start3:]

OscAmp1 = OscAmp_all[0:stop1]
OscAmp2 = OscAmp_all[start2:stop2]
OscAmp3 = OscAmp_all[start3:]

OscPeriod1 = OscPeriod_all[0:stop1]
OscPeriod2 = OscPeriod_all[start2:stop2]
OscPeriod3 = OscPeriod_all[start3:]

plt.scatter(abs(OscAmp1), OscPeriod1, c = 'black', label = '3/15 Cooldown')
plt.scatter(abs(OscAmp2), OscPeriod2, c = 'red', label = '7/25 Cooldown')
plt.scatter(abs(OscAmp3), OscPeriod3, c = 'orange', label = '8/16 Cooldown')
plt.xlabel('Temperature Oscillation Amplitude (K)')
plt.ylabel('Temperature Oscillation Period (Runs)')
plt.title('Temperature Oscillation Period vs Amplitude')
plt.legend()
#plt.savefig('/Users/tiffanypaul/Desktop/Oscillation_compare.pdf', bbox_inches="tight")
plt.show()

plt.scatter(abs(OscAmp1), MT1, c = 'black', label = '3/15 Cooldown')
plt.scatter(abs(OscAmp2), MT2, c = 'red', label = '7/25 Cooldown')
plt.scatter(abs(OscAmp3), MT3, c = 'orange', label = '8/16 Cooldown')
plt.xlabel('Temperature Oscillation Amplitude (K)')
plt.ylabel('Mean Temperature (K)')
plt.title('Temperature Oscillation vs Mean')
plt.legend()
#plt.savefig('/Users/tiffanypaul/Desktop/Oscillation_compare.pdf', bbox_inches="tight")
plt.show()

plt.scatter(np.linspace(0, sto1-1, sto1-1), AllMT1[:-1], c = 'black', label = '3/15 Cooldown')
plt.scatter(np.linspace(sta2, sto2, sto2-sta2), AllMT2, c = 'red', label = '7/25 Cooldown')
plt.scatter(np.linspace(sta3+1, len(MT_all)+1, len(MT_all) - sta3), AllMT3, c = 'orange', label = '8/16 Cooldown')
plt.xlabel('Data Taking Order')
plt.ylabel('Mean Temperature (K)')
plt.title('Mean Temperatures')
plt.legend()
#plt.savefig('/Users/tiffanypaul/Desktop/Oscillation_compare.pdf', bbox_inches="tight")
plt.show()