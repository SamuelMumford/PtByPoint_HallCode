# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:59:04 2021

@author: sammy
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 10:07:12 2021
@author: tiffanypaul
"""



import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

fname = '/home/sam/Documents/Cooldown_all_11_18_20_cool.txt'
f2 = '/home/sam/Documents/Cooldown_all_11_18_20_coolL.txt'
fname12 = '/home/sam/Documents/12_9_20_coolEdit.txt'
f122 = '/home/sam/Documents/12_9_20_coolEditL.txt'
fname2k = '/home/sam/Documents/Cooldown_all_1_14_20_cool.txt'
f22k = '/home/sam/Documents/Cooldown_all_1_14_20_coolL.txt'

def readIn(fname, f2, startcut, endcut):
    Fdat = np.loadtxt(fname, delimiter="\t")
    F2 = np.loadtxt(f2, delimiter="\t")
    #keep = True
    freqs = Fdat[startcut:endcut, 2]
    runs = Fdat[startcut:endcut, 3]
    currs = Fdat[startcut:endcut, 5]
    Q = F2[startcut:endcut, 3]
    Q = np.abs(Q)
    Bs = currs*.875/5
    temps = Fdat[startcut:endcut, 4]
    return freqs, runs, Q, Bs, temps

def evensAndOdds(runs):
    length = int(len(runs))
    eis = np.ones(length, dtype=bool)
    ois = np.ones(length, dtype=bool)
    for i in range(0, len(runs)):
        if(runs[i]%2 == 0):
            ois[i] = 0
        else:
            eis[i] = 0
    return eis, ois

def segmentEO(Bs, freqs, runs):
    start = 0
    startB = Bs[start]
    info = []
    for i in range(0, len(Bs)):
        if(Bs[i] != startB):
            subFs = freqs[start:i]
            subRs = runs[start:i]
            evens, odds = evensAndOdds(subRs)
            efs = subFs[evens]
            ofs = subFs[odds]
            ers = subRs[evens]
            ors = subRs[odds]
            oddPred = np.interp(ors, ers, efs)
            evenPred = np.interp(ers, ors, ofs)
            sigE = np.std(evenPred - efs)/np.sqrt(len(ers))
            sigO = np.std(oddPred - ofs)/np.sqrt(len(ofs))
            sigtot = np.sqrt(sigE**2 + sigO**2)/np.sqrt(2)
            diff = np.mean(ofs -oddPred) - np.mean(efs - evenPred)
            diff = diff/2
            info.append([startB, diff, sigtot])
            
            startB = Bs[i]
            start = i
    subFs = freqs[start:]
    subRs = runs[start:]
    evens, odds = evensAndOdds(subRs)
    efs = subFs[evens]
    ofs = subFs[odds]
    ers = subRs[evens]
    ors = subRs[odds]
    oddPred = np.interp(ors, ers, efs)
    evenPred = np.interp(ers, ors, ofs)
    sigE = np.std(evenPred - efs)/np.sqrt(len(ers))
    sigO = np.std(oddPred - ofs)/np.sqrt(len(ofs))
    sigtot = np.sqrt(sigE**2 + sigO**2)/2
    diff = np.mean(ofs -oddPred) - np.mean(efs - evenPred)
    diff = diff/2
    info.append([startB, diff, sigtot])
    
    info = np.array(info).reshape(-1, 3).T
    return info

def consolidate(info):
    Bs = info[0]
    Shifts = info[1]
    Sigs = info[2]
    Dict = {}
    for i in range(0, len(Bs)):
        if(Bs[i] in Dict.keys()):
            present = Dict[Bs[i]]
            present[1] = (present[1]/(present[2]**2) + Shifts[i]/(Sigs[i]**2))/(1/(present[2]**2) + 1/(Sigs[i]**2))
            present[2] = np.sqrt(1/(1/(present[2]**2) + 1/(Sigs[i]**2)))
        else:
            Dict[Bs[i]] = [Bs[i], Shifts[i], Sigs[i]]
    DictList = []
    for key in Dict:
        DictList.append(Dict[key])
    DictList = np.array(DictList).reshape(-1, 3).T
    return Dict, DictList

def consolidateEven(ShDict):
    Dict = {}
    for key in ShDict:
        if(np.abs(key) not in Dict.keys()):
            if(-key in ShDict.keys()):
                if(key== 0):
                    Dict[key] = ShDict[key]
                else:
                    PB = ShDict[np.abs(key)]
                    MB = ShDict[-np.abs(key)]
                    sh = (PB[1] + MB[1])/2
                    sig = np.sqrt(PB[2]**2 + MB[2]**2)/2
                    Dict[np.abs(key)] = [np.abs(key), sh, sig]
    DictList = []
    for key in Dict:
        DictList.append(Dict[key])
    DictList = np.array(DictList).reshape(-1, 3).T
    return Dict, DictList

def consolidateOdd(ShDict):
    Dict = {}
    for key in ShDict:
        if(np.abs(key) not in Dict.keys()):
            if(-key in ShDict.keys()):
                if(key!= 0):
                    PB = ShDict[np.abs(key)]
                    MB = ShDict[-np.abs(key)]
                    sh = (PB[1] - MB[1])/2
                    sig = np.sqrt(PB[2]**2 + MB[2]**2)/2
                    Dict[np.abs(key)] = [np.abs(key), sh, sig]
    DictList = []
    for key in Dict:
        DictList.append(Dict[key])
    DictList = np.array(DictList).reshape(-1, 3).T
    return Dict, DictList

def fit(x,a,b, c, d, e, f, g, h, i): 
     return a + b*x + c*x**3 + d*x**5 + e*x**7 + f*x**2 + g*x**4 + h*x**6 + i*x**8
 
    
def EOFit(Bs, Es, Os):
    fit = np.zeros(len(Bs))
    i = 0
    #The m's are the fit coefficients, doing only an even fit in B
    for m in zip(Es):
        fit += m*np.power(Bs, i)
        i += 2
    #n's are fit coefficients, allow for an offset term in the run index fit
    j = 1
    for n in zip(Os):
        fit += n*np.power(Bs, j)
        j += 2
    return fit

def wrapEO(x, *args):
    #Note that NTs is a global variable telling which indexes separate
    #the B and run fit coefficients
    polyEs = list(args[:N])
    polyOs = list(args[N:])
    return EOFit(x, polyEs, polyOs)

def makeInfo(Bs, freqs, runs):
    inf = segmentEO(Bs, freqs, runs)
    Bfield = inf[0]
    Signal = inf[1]*1000
    error = inf[2]*1000
    return Bfield, Signal, error

def getColors(Bfield):
    Bcolor = []
    #fields going to higher magnitude fields
    mask_B_high = [True]*len(Bfield)
    #going to lower magnitude fields (towards zero)
    mask_B_low = [True]*len(Bfield)
    for i in range(0,len(Bfield)):
        Bcolor.append('blue')
    #print(Bcolor)
    #Blue for points going up from 0 
    #Red for points going down from high field
        #-4 so we don't change the colors of the last 4 points that were taken at the end 
    for i in range(0, len(Bfield)):
        if np.abs(Bfield[i]) >= np.abs(Bfield[i-1]):
            Bcolor[i] = 'blue' 
            mask_B_low[i] = False
        if np.abs(Bfield[i]) < np.abs(Bfield[i-1]):
            Bcolor[i] = 'red'
            mask_B_high[i] = False
    #Make first 0 point blue
    Bcolor[0] = 'blue'
    return Bcolor, mask_B_high, mask_B_low

def getConsol(Bfield, signal, error, mask_B_high, mask_B_low):
    B_high = Bfield[mask_B_high]
    B_low = Bfield[mask_B_low]
    signal_high = signal[mask_B_high]
    signal_low = signal[mask_B_low]
    error_high = error[mask_B_high]
    error_low = error[mask_B_low]
    _, Info = consolidate([B_high, signal_high, error_high])
    B_high = Info[0]
    signal_high = Info[1]
    error_high = Info[2]
    _, InfoL = consolidate([B_low, signal_low, error_low])
    B_low = InfoL[0]
    signal_low = InfoL[1]
    error_low = InfoL[2]
    B_high, signal_high, error_high = zip(*sorted(zip(B_high, signal_high, error_high)))
    B_low, signal_low, error_low = zip(*sorted(zip(B_low, signal_low, error_low)))
    
    B_high = np.array(B_high)
    signal_high = np.array(signal_high)
    error_high = np.array(error_high)
    
    B_low = np.array(B_low)
    signal_low = np.array(signal_low)
    error_low = np.array(error_low)
    
    return B_high, B_low, signal_high, signal_low, error_high, error_low

def makeInterp(B_high, B_low, signal_high, signal_low, error_high, error_low):
    HighPreds = interp1d(B_high, signal_high, kind='linear')
    LowPreds = interp1d(B_low, signal_low, kind='linear')
    VarHighPreds = interp1d(B_high, (1.0*error_high)**2, kind='linear')
    VarLowPreds = interp1d(B_low, (1.0*error_low)**2, kind='linear')
    
    mask1 = (B_high < max(B_low))
    mask2 = (B_high > min(B_low))
    maskH = mask1*mask2
    B_sub_high = B_high[maskH]
    signal_sub_high = signal_high[maskH]
    error_sub_high = error_high[maskH]
    
    mask1 = (B_low < max(B_high))
    mask2 = (B_low > min(B_high))
    maskL = mask1*mask2
    B_sub_low = B_low[maskL]
    signal_sub_low = signal_low[maskL]
    error_sub_low = error_low[maskL]
    
    HighPredPts = HighPreds(B_sub_low)
    LowPredPts = LowPreds(B_sub_high)
    HighPredVar = VarHighPreds(B_sub_low)
    LowPredVar = VarLowPreds(B_sub_high)
    
    return B_sub_low, B_sub_high, signal_sub_low, signal_sub_high, error_sub_low, error_sub_high, LowPredPts, HighPredPts, LowPredVar, HighPredVar
    

freqs, runs, Q, Bs, temps = readIn(fname, f2, 0, 3450)#3450
freqs12, runs12, Q12, Bs12, temps12 = readIn(fname12, f122, 0, -340)#-340
freqs2k, runs2k, Q2k, Bs2k, temps2k = readIn(fname2k, f22k, 0, -340)#-5

Bfield, signal, error = makeInfo(Bs, freqs, runs)
Bfield12, signal12, error12 = makeInfo(Bs12, freqs12, runs12)
Bfield2k, signal2k, error2k = makeInfo(Bs2k, freqs2k, runs2k)


a = .2212*(5)/.875
base = 5.25
a = a
s0 = 1.*base/3.65
s1 = 1.*base/3.5
s2 = .85*base/1.9
print(Bfield)
print(Bfield12)
print(Bfield2k)
cl = 1.7
cut1 = np.abs(Bfield) < cl
cut2 = np.abs(Bfield12) < cl
cut3 = np.abs(Bfield2k) < cl

B1 = Bfield[cut1]
sig1 = signal[cut1] - a*s0*B1 - 1.3
err1 = error[cut1]

B2 = Bfield12[cut2]
sig2 = signal12[cut2] - a*s1*B2 - 1.6
err2 = error12[cut2]

B3 = Bfield2k[cut3]
sig3 = signal2k[cut3] - a*s2*B3 - 2.0
err3 = error2k[cut3]
#
#plt.errorbar(B1, sig1, yerr = err1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#plt.errorbar(B2, sig2, yerr = err2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#plt.errorbar(B3, sig3, yerr = err3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')
#plt.show()

ch4k = 5.7
ch2k = 5
cut1h = np.abs(Bfield) > ch4k
cut2h = np.abs(Bfield12) > ch4k
cut3h = np.abs(Bfield2k) > ch2k

Bh1 = Bfield[cut1h]
sigh1 = signal[cut1h] - a*s0*Bh1 - 1.3
errh1 = error[cut1h]

Bh2 = Bfield12[cut2h]
sigh2 = signal12[cut2h] - a*s1*Bh2 - 1.6
errh2 = error12[cut2h]

Bh3 = Bfield2k[cut3h]
sigh3 = signal2k[cut3h] - a*s2*Bh3 - 2.0
errh3 = error2k[cut3h]

#plt.errorbar(Bh1, sigh1, yerr = errh1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#plt.errorbar(Bh2, sigh2, yerr = errh2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')
#plt.show()

#plt.errorbar(Bh3, sigh3, yerr = errh3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')


#f, (ax, ax2) = plt.subplots(1, 2, sharey=True)

cut1m = np.abs(Bfield) < ch4k
cut2m = np.abs(Bfield12) < ch4k
cut3m = np.abs(Bfield2k) < ch2k

for i in range(0, len(cut1m)):
    if(cut1m[i] == True):
        if(np.abs(Bfield[i]) < cl):
            cut1m[i] = False
            
for i in range(0, len(cut2m)):
    if(cut2m[i] == True):
        if(np.abs(Bfield12[i]) < cl):
            cut2m[i] = False
            
for i in range(0, len(cut3m)):
    if(cut3m[i] == True):
        if(np.abs(Bfield2k[i]) < cl):
            cut3m[i] = False

Bm1 = Bfield[cut1m]
sigm1 = signal[cut1m] - a*s0*Bm1 - 1.3
errm1 = error[cut1m]

Bm2 = Bfield12[cut2m]
sigm2 = signal12[cut2m] - a*s1*Bm2 - 1.6
errm2 = error12[cut2m]

Bm3 = Bfield2k[cut3m]
sigm3 = signal2k[cut3m] - a*s2*Bm3 - 2.0
errm3 = error2k[cut3m]


# plot the same data on both axes
#ax.errorbar(Bfield, signal- a*s0*Bfield - 1.3, yerr = error, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#ax2.errorbar(Bfield, signal- a*s0*Bfield - 1.3, yerr = error, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#ax.errorbar(Bfield12, signal12- a*s1*Bfield12 - 1.6, yerr = error12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#ax2.errorbar(Bfield12, signal12- a*s1*Bfield12 - 1.6, yerr = error12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#ax.errorbar(Bm3, sigm3, yerr = errm3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#ax2.errorbar(Bm3, sigm3, yerr = errm3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#
## zoom-in / limit the view to different portions of the data
#ax.set_xlim(-ch2k -.1, -cl+ .1)  # outliers only
#ax2.set_xlim(cl-.1, ch2k + .1)  # most of the data
#ax.set_ylim(-2.5, 4)  # outliers only
#
#ax.spines['right'].set_visible(False)
#ax2.spines['left'].set_visible(False)
##ax.xaxis.tick_top()
#ax2.tick_params(labelleft=False)  # don't put tick labels at the top
#ax2.yaxis.tick_right()
#
#d = .03  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs) 
#ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
#ax2.plot((-d, +d), (-d, +d), **kwargs)
#
#ax.set_ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#ax.set_xlabel('Magnetic Field (T)')
#ax2.set_xlabel('Magnetic Field (T)')
#plt.show()


#f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
#
## plot the same data on both axes
#ax.errorbar(Bm1, sigm1, yerr = errm1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#ax2.errorbar(Bm1, sigm1, yerr = errm1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#ax.errorbar(Bm2, sigm2, yerr = errm2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#ax2.errorbar(Bm2, sigm2, yerr = errm2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
##ax.errorbar(Bfield2k, signal2k- a*s2*Bfield2k - 2, yerr = error2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
##ax2.errorbar(Bfield2k, signal2k- a*s2*Bfield2k - 2, yerr = error2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#
## zoom-in / limit the view to different portions of the data
#ax.set_xlim(-ch4k -.1, -cl+ .1)  # outliers only
#ax2.set_xlim(cl-.1, ch4k + .1)  # most of the data
#ax.set_ylim(-2.5, 4)  # outliers only
#
#ax.spines['right'].set_visible(False)
#ax2.spines['left'].set_visible(False)
##ax.xaxis.tick_top()
#ax2.tick_params(labelleft=False)  # don't put tick labels at the top
#ax2.yaxis.tick_right()
#
#d = .03  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs) 
#ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
#ax2.plot((-d, +d), (-d, +d), **kwargs)
#
#ax.set_ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#ax.set_xlabel('Magnetic Field (T)')
#ax2.set_xlabel('Magnetic Field (T)')
#plt.show()




#f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
#
## plot the same data on both axes
#ax.errorbar(Bh3, sigh3, yerr = errh3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#ax2.errorbar(Bh3, sigh3, yerr = errh3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#
## zoom-in / limit the view to different portions of the data
#ax.set_xlim(np.amin(Bh3)-.1, -ch2k+.1)  # outliers only
#ax2.set_xlim(ch2k-.1, np.amax(Bh3) + .1)  # most of the data
#
#ax.spines['right'].set_visible(False)
#ax2.spines['left'].set_visible(False)
##ax.xaxis.tick_top()
#ax2.tick_params(labelleft=False)  # don't put tick labels at the top
#ax2.yaxis.tick_right()
#
#d = .03  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs) 
#ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
#ax2.plot((-d, +d), (-d, +d), **kwargs)
#
#ax.set_ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#ax.set_xlabel('Magnetic Field (T)')
#ax2.set_xlabel('Magnetic Field (T)')
#plt.show()
#
#
#f, (ax, ax2) = plt.subplots(1, 2, sharey=True)
#
## plot the same data on both axes
#ax.errorbar(Bh1, sigh1, yerr = errh1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#ax2.errorbar(Bh1, sigh1, yerr = errh1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#ax.errorbar(Bh2, sigh2, yerr = errh2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#ax2.errorbar(Bh2, sigh2, yerr = errh2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#
## zoom-in / limit the view to different portions of the data
#ax.set_xlim(np.amin(Bh3)-.1, -ch4k+.1)  # outliers only
#ax2.set_xlim(ch4k-.1, np.amax(Bh3) + .1)  # most of the data
#
#ax.spines['right'].set_visible(False)
#ax2.spines['left'].set_visible(False)
##ax.xaxis.tick_top()
#ax2.tick_params(labelleft=False)  # don't put tick labels at the top
#ax2.yaxis.tick_right()
#
#d = .03  # how big to make the diagonal lines in axes coordinates
## arguments to pass to plot, just so we don't keep repeating them
#kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
#ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs) 
#ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
#
#kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
#ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
#ax2.plot((-d, +d), (-d, +d), **kwargs)
#
#ax.set_ylabel('$\delta f_{0}(\pm V)$ (mHz)')
#ax.set_xlabel('Magnetic Field (T)')
#ax2.set_xlabel('Magnetic Field (T)')
#plt.show()

Bcolor, mask_B_high, mask_B_low = getColors(Bfield)
Bcolor12, mask_B_high12, mask_B_low12 = getColors(Bfield12)
Bcolor2k, mask_B_high2k, mask_B_low2k = getColors(Bfield2k)

#plt.errorbar(Bfield12[mask_B_high12], signal12[mask_B_high12], yerr = error12[mask_B_high12], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")
#plt.errorbar(Bfield12[mask_B_low12], signal12[mask_B_low12], yerr = error12[mask_B_low12], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = "ramp down")
#plt.show()

B_high, B_low, signal_high, signal_low, error_high, error_low = getConsol(Bfield, signal, error, mask_B_high, mask_B_low)
B_high12, B_low12, signal_high12, signal_low12, error_high12, error_low12 = getConsol(Bfield12, signal12, error12, mask_B_high12, mask_B_low12)
B_high2k, B_low2k, signal_high2k, signal_low2k, error_high2k, error_low2k = getConsol(Bfield2k, signal2k, error2k, mask_B_high2k, mask_B_low2k)

B_sub_low, B_sub_high, signal_sub_low, signal_sub_high, error_sub_low, error_sub_high, LowPredPts, HighPredPts, LowPredVar, HighPredVar = makeInterp(B_high, B_low, signal_high, signal_low, error_high, error_low)
B_sub_low12, B_sub_high12, signal_sub_low12, signal_sub_high12, error_sub_low12, error_sub_high12, LowPredPts12, HighPredPts12, LowPredVar12, HighPredVar12 = makeInterp(B_high12, B_low12, signal_high12, signal_low12, error_high12, error_low12)
B_sub_low2k, B_sub_high2k, signal_sub_low2k, signal_sub_high2k, error_sub_low2k, error_sub_high2k, LowPredPts2k, HighPredPts2k, LowPredVar2k, HighPredVar2k = makeInterp(B_high2k, B_low2k, signal_high2k, signal_low2k, error_high2k, error_low2k)

##############

#plt.plot(B_sub_low, HighPredPts)
#plt.plot(B_sub_high, LowPredPts)

plt.errorbar(B_low, signal_low - a*s0*B_low - 1.3, yerr = error_low, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = "ramp down")
plt.errorbar(B_high, signal_high- a*s0*B_high - 1.3, yerr = error_high, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")

plt.errorbar(B_low12, signal_low12 - a*s1*B_low12 -1.6, yerr = error_low12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = "ramp down")
plt.errorbar(B_high12, signal_high12 - a*s1*B_high12 - 1.6, yerr = error_high12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")

plt.axvline(x=-ch4k)
plt.axvline(x=ch4k)
plt.axvline(x=-cl, color = 'green')
plt.axvline(x=cl, color = 'green')

plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc = 'best')
plt.show()

plt.errorbar(B_low2k, signal_low2k - a*s2*B_low2k - 2, yerr = error_low2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = "ramp down")
plt.errorbar(B_high2k, signal_high2k - a*s2*B_high2k -2, yerr = error_high2k, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = "ramp up")
plt.axvline(x=-ch2k)
plt.axvline(x=ch2k)
plt.axvline(x=-cl, color = 'green')
plt.axvline(x=cl, color = 'green')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc = 'best')
plt.show()

def getHist(B_sub_low, B_sub_high, signal_sub_low, signal_sub_high, error_sub_low, error_sub_high, LowPredPts, HighPredPts, LowPredVar, HighPredVar):
    Sig1 = (HighPredPts - signal_sub_low)
    Sig2 = (LowPredPts - signal_sub_high)
    Err1 = np.sqrt(HighPredVar + error_sub_low**2)
    Err2 = np.sqrt(LowPredVar + error_sub_high**2)
    
    AllBs = np.hstack((B_sub_low, B_sub_high))
    AllSs = np.hstack((Sig1, -Sig2))
    AllEs = np.hstack((Err1, Err2))
    
    return AllBs, AllSs, AllEs

AllBs, AllSs, AllEs = getHist(B_sub_low, B_sub_high, signal_sub_low, signal_sub_high, error_sub_low, error_sub_high, LowPredPts, HighPredPts, LowPredVar, HighPredVar)
AllBs12, AllSs12, AllEs12 = getHist(B_sub_low12, B_sub_high12, signal_sub_low12, signal_sub_high12, error_sub_low12, error_sub_high12, LowPredPts12, HighPredPts12, LowPredVar12, HighPredVar12)
AllBs2k, AllSs2k, AllEs2k = getHist(B_sub_low2k, B_sub_high2k, signal_sub_low2k, signal_sub_high2k, error_sub_low2k, error_sub_high2k, LowPredPts2k, HighPredPts2k, LowPredVar2k, HighPredVar2k)

#plt.errorbar(AllBs, AllSs, yerr = AllEs, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 1')
#plt.errorbar(AllBs12, AllSs12, yerr = AllEs12, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label  = '4kOhm, Run 2')
##plt.errorbar(AllBs2k, AllSs2k, yerr = AllEs2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '2kOhm')
#
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Hysteretic $\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')
#plt.show()
#
#plt.errorbar(AllBs2k, AllSs2k, yerr = AllEs2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '2kOhm')
#
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Hysteretic $\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')
#plt.show()

Ab, As, Ae = np.copy(AllBs), np.copy(AllSs), np.copy(AllEs)
Ab12, As12, Ae12 = np.copy(AllBs12), np.copy(AllSs12), np.copy(AllEs12)
Ab2k, As2k, Ae2k = np.copy(AllBs2k), np.copy(AllSs2k), np.copy(AllEs2k)

def getNonHist(B_sub_low, B_sub_high, signal_sub_low, signal_sub_high, error_sub_low, error_sub_high, LowPredPts, HighPredPts, LowPredVar, HighPredVar):
    AllBs = np.hstack((B_sub_low, B_sub_high))
    LandPLs = np.hstack((signal_sub_low, LowPredPts))
    LVandPLVs = np.sqrt(np.hstack((error_sub_low**2, LowPredVar)))
    HandPHs = np.hstack((HighPredPts, signal_sub_high))
    HVandPHVs = np.sqrt(np.hstack((HighPredVar, error_sub_high**2)))
    
    SigBase = (LandPLs + HandPHs)/2
    ErrBase = np.sqrt((LVandPLVs**2 + HVandPHVs**2))/2
    
    return AllBs, SigBase, ErrBase

AllBs0, SigBase0, ErrBase0 = getNonHist(B_sub_low, B_sub_high, signal_sub_low, signal_sub_high, error_sub_low, error_sub_high, LowPredPts, HighPredPts, LowPredVar, HighPredVar)
AllBs120, SigBase120, ErrBase120 = getNonHist(B_sub_low12, B_sub_high12, signal_sub_low12, signal_sub_high12, error_sub_low12, error_sub_high12, LowPredPts12, HighPredPts12, LowPredVar12, HighPredVar12)
AllBs2k0, SigBase2k0, ErrBase2k0 = getNonHist(B_sub_low2k, B_sub_high2k, signal_sub_low2k, signal_sub_high2k, error_sub_low2k, error_sub_high2k, LowPredPts2k, HighPredPts2k, LowPredVar2k, HighPredVar2k)

def correctNonHist(AllBs, SigBase, ErrBase, Bfield, signal, error):
    l = len(Bfield)
    for i in range(0, l):
        listy=np.where(AllBs == Bfield[i])
        b = listy[0]
        if (len(b) == 0):
            AllBs = np.hstack((AllBs, [Bfield[i]]))
            SigBase = np.hstack((SigBase, [signal[i]]))
            ErrBase = np.hstack((ErrBase, [error[i]]))
    return AllBs, SigBase, ErrBase
        
AllBs, SigBase, ErrBase = correctNonHist(AllBs0, SigBase0, ErrBase0, Bfield, signal, error)
AllBs12, SigBase12, ErrBase12 = correctNonHist(AllBs120, SigBase120, ErrBase120, Bfield12, signal12, error12)
AllBs2k, SigBase2k, ErrBase2k = correctNonHist(AllBs2k0, SigBase2k0, ErrBase2k0, Bfield2k, signal2k, error2k)



def EO(AllBs, SigBase, ErrBase):   
    DicBase, _ = consolidate([AllBs, SigBase, ErrBase])
    _, EInfoB = consolidateEven(DicBase)
    _, OInfoB = consolidateOdd(DicBase)
    
    EBsB = EInfoB[0]
    EHSigB = EInfoB[1]
    EHUncB = EInfoB[2]
    OBsB = OInfoB[0]
    OHSigB = OInfoB[1]
    OHUncB = OInfoB[2]
    return EBsB, OBsB, EHSigB, OHSigB, EHUncB, OHUncB

EBsB, OBsB, EHSigB, OHSigB, EHUncB, OHUncB = EO(AllBs, SigBase, ErrBase)
EBsB12, OBsB12, EHSigB12, OHSigB12, EHUncB12, OHUncB12 = EO(AllBs12, SigBase12, ErrBase12)
EBsB2k, OBsB2k, EHSigB2k, OHSigB2k, EHUncB2k, OHUncB2k = EO(AllBs2k, SigBase2k, ErrBase2k)

def FF(x, a, b):
    return a*x**2 + b

R1Es = EHSigB -1.3
R2Es = EHSigB12 - 1.6
R3Es = EHSigB2k - 2.0

Bs2kf = np.hstack((EBsB, EBsB12))
Ss2k = np.hstack((R2Es, R2Es))
Es2k = np.hstack((EHUncB, EHUncB12))

pof, pcf = curve_fit(FF, Bs2kf, Ss2k, sigma = Es2k, p0=(0, 0))

print(pof)
print(np.sqrt(np.diag(pcf)))

pof4k, pcf4k = curve_fit(FF, EBsB2k, R3Es, sigma = EHUncB2k, p0=(0, 0))

print(pof4k)
print(np.sqrt(np.diag(pcf4k)))


#plt.errorbar(EBsB, R2Es, yerr = EHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
#plt.errorbar(EBsB12, R2Es, yerr = EHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 2')
#plt.errorbar(EBsB2k, R3Es, yerr = EHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '2kOhm')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Even Base $\delta f_{0}(\pm V)$ (mHz)')
##plt.legend(loc= "best")
#plt.show()
#
#plt.errorbar(OBsB, OHSigB, yerr = OHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
#plt.errorbar(OBsB12, OHSigB12, yerr = OHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
#plt.errorbar(OBsB2k, OHSigB2k, yerr = OHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Odd Base $\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')
#plt.show()
#
#plt.errorbar(OBsB, OHSigB/s0, yerr = OHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
#plt.errorbar(OBsB12, OHSigB12/s1, yerr = OHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
#plt.errorbar(OBsB2k, OHSigB2k/s2, yerr = OHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('R-Scaled Odd Base $\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc = 'best')
#plt.show()
#
##plt.errorbar(OBsB, OHSigB - a*s0*OBsB, yerr = OHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
#plt.errorbar(OBsB12, OHSigB12 - a*s1*OBsB12, yerr = OHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2, odd')
##plt.errorbar(OBsB2k, OHSigB2k - a*s2*OBsB2k, yerr = OHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.errorbar(EBsB12, EHSigB12 - 1.6, yerr = EHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '4kOhm, Run 2, even')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Nonlinear Odd Component (mHz)')
#plt.legend(loc = 'best')
#plt.show()
#
#plt.errorbar(OBsB, OHSigB - a*s0*OBsB, yerr = OHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1, odd')
##plt.errorbar(OBsB12, OHSigB12 - a*s1*OBsB12, yerr = OHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2, odd')
##plt.errorbar(OBsB2k, OHSigB2k - a*s2*OBsB2k, yerr = OHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.errorbar(EBsB, EHSigB - 1.3, yerr = EHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '4kOhm, Run 1, even')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Nonlinear Odd Component (mHz)')
#plt.legend(loc = 'best')
#plt.show()
#
#
##plt.errorbar(OBsB, OHSigB - a*s0*OBsB, yerr = OHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
##plt.errorbar(OBsB12, OHSigB12 - a*s1*OBsB12, yerr = OHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2, odd')
#plt.errorbar(OBsB2k, OHSigB2k - a*s2*OBsB2k, yerr = OHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm, odd')
#plt.errorbar(EBsB2k, EHSigB2k - 2, yerr = EHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2kOhm, even')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Nonlinear Odd Component (mHz)')
#plt.legend(loc = 'best')
#plt.show()


def segment(Bs, temp, outliers):
    start = 0
    startB = Bs[start]
    info = []
    index = -1
    count = 0
    if(len(outliers) != 0):
        index = outliers[0]
    j = 0
    for i in range(0, len(Bs)):
        if(Bs[i] != startB):
            sub = temp[start:i]
            mean = np.mean(sub)
            dev = np.std(sub)/3
            if(j != index):
                info.append([startB, mean, dev])
                j += 1
            else:
                j += 1
                count += 1
                if(count < len(outliers)):
                    index = outliers[count]
            startB = Bs[i]
            start = i
    sub = temp[start:]
    mean = np.mean(sub)
    dev = np.std(sub)/3
    info.append([startB, mean, dev])
    
    info = np.array(info).reshape(-1, 3).T
    return info

def TFit(data, off, poly):
    part1 = data[:, 0]
    part2 = data[:, 1]
    part3 = data[:, 2]
    Bs = data[:, 3]
    
    fit = np.zeros(len(part1))
    fit += off[0]*part1 + off[1]*part2 + off[2]*part3
    fit += poly[0]*(np.abs(Bs) > .1)
    j = 2
    for i in range(1, len(poly)):
        fit += poly[i]*np.power(Bs, j)
        j += 2
    return fit
    
    #When you call the curve_fit code, it can ONLY take things of the form 
    #function(x, args). We need to make a wrapping function to put TFit in 
    #that form.        
def wrapT(x, *args):
    #Note that NTs is a global variable telling which indexes separate
    #the B and run fit coefficients
    off = list(args[:3])
    poly = list(args[3:])
    return TFit(x, off, poly)

def F0Fit(dataf, off, polyB, polyT, polyQ):
    part1 = dataf[:, 0]
    part2 = dataf[:, 1]
    part3 = dataf[:, 2]
    Bs = dataf[:, 3]
    Ts = dataf[:, 4]
    Qs = dataf[:, 5]
    Qs = (Qs - np.mean(Qs))/np.std(Qs)
    
    fit = np.zeros(len(part1))
    fit += off[0]*part1 + off[1]*part2 + off[2]*part3
    fit += polyB[0]*np.tanh(Bs*polyB[1])**2
    j = 2
    for i in range(2, len(polyB)):
        fit += polyB[i]*np.power(Bs, j)
        j += 2
    j = 2
    for i in range(0, len(polyT)):
        fit += polyT[i]*np.power(Ts, j)
        j += 2
    j = 1
    for i in range(0, len(polyQ)):
        fit += polyQ[i]*np.power(Qs, j)
        j += 1
    return fit
    
    #When you call the curve_fit code, it can ONLY take things of the form 
    #function(x, args). We need to make a wrapping function to put TFit in 
    #that form.        
def wrapF(x, *args):
    #Note that NTs is a global variable telling which indexes separate
    #the B and run fit coefficients
    off = list(args[:3])
    polyB = list(args[3:3+N])
    polyT = list(args[3+N:3+N+NT])
    polyQ = list(args[3+N+NT:])
    return F0Fit(x, off, polyB, polyT, polyQ)


def GetTs(infoT1, infoT2, infoT3):
    l1 = len(infoT1[0])
    l2 = len(infoT2[0])
    l3 = len(infoT3[0])
    
    Cat1 = np.hstack((np.ones(l1), np.zeros(l2), np.zeros(l3)))
    Cat2 = np.hstack((np.zeros(l1), np.ones(l2), np.zeros(l3)))
    Cat3 = np.hstack((np.zeros(l1), np.zeros(l2), np.ones(l3)))
    
    AllBs = np.hstack((infoT1[0], infoT2[0], infoT3[0]))
    AllTs = np.hstack((infoT1[1], infoT2[1], infoT3[1]))
    AllSigs = np.hstack((infoT1[2], infoT2[2], infoT3[2]))
    
    data = np.vstack((Cat1, Cat2, Cat3, AllBs)).T
    NTot = 7
    p0 = np.zeros(NTot)
    p0[0] = 5.75
    p0[1] = 5.75
    p0[2] = 5.75
    p0[3] = -.35
    p0[4] = .1
    p0[5] = -.012
    popt, pcov = curve_fit(lambda data, *p0: wrapT(data, *p0), data, AllTs, sigma = AllSigs, p0=p0)
    fitty = wrapT(data, *popt)
    
    pCopy = np.copy(popt)
    pCopy[0] = 0
    pCopy[1] = 0
    pCopy[2] = 0
    fittySub = wrapT(data, *pCopy)
    TrueTs = AllTs - fittySub
    
    return TrueTs

def GetFs(infoF1, infoF2, infoF3, infoQ1, infoQ2, infoQ3, TrueTs):
    l1 = len(infoF1[0])
    l2 = len(infoF2[0])
    l3 = len(infoF3[0])
    
    AllBs = np.hstack((infoF1[0], infoF2[0], infoF3[0]))
    Cat1 = np.hstack((np.ones(l1), np.zeros(l2), np.zeros(l3)))
    Cat2 = np.hstack((np.zeros(l1), np.ones(l2), np.zeros(l3)))
    Cat3 = np.hstack((np.zeros(l1), np.zeros(l2), np.ones(l3)))
    
    AllQs = np.hstack((infoQ1[1], infoQ2[1], infoQ3[1]))
    
    dataF0 = np.vstack((Cat1, Cat2, Cat3, AllBs, TrueTs, AllQs)).T
    
    NTot = N + 3 + NT + NQ
    pf = np.zeros(NTot)
    pf[0] = np.mean(infoF1[1])
    pf[1] = np.mean(infoF2[1])
    pf[2] = np.mean(infoF3[1])
    pf[3] = .3
    pf[4] = 1
    
    AllFs = np.hstack((infoF1[1], infoF2[1], infoF3[1]))
    AllFSigs = np.hstack((infoF1[2], infoF2[2], infoF3[2]))
    
    pof, pcf = curve_fit(lambda dataF0, *pf: wrapF(dataF0, *pf), dataF0, AllFs, p0=pf)
    #print(pof)
    #print((pof/np.sqrt(np.diag(pcf)))[3:])
    EFs = wrapF(dataF0, *pof)
    pc = np.copy(pof)
    for i in range(3, len(pc)):
        pc[i] = 0
    offy = wrapF(dataF0, *pc)
    #plt.errorbar(np.abs(AllBs), AllFs - offy, yerr = AllFSigs, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
    #plt.plot(np.abs(AllBs), EFs - offy, '.')
    #plt.ylabel('$<f_{0}>$ (Hz)')
    #plt.xlabel('B (T)')
    #plt.show()
    
    #plt.errorbar(np.abs(AllBs), AllFs - EFs, yerr = AllFSigs, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
    #plt.show()
    
    Rs1 = (AllFs - EFs)[Cat1 > 0]
    Rs2 = (AllFs - EFs)[Cat2 > 0]
    Rs3 = (AllFs - EFs)[Cat3 > 0]
    Fs1 = AllFSigs[Cat1 > 0]
    Fs2 = AllFSigs[Cat2 > 0]
    Fs3 = AllFSigs[Cat3 > 0]
    pBs1 = infoF1[0]
    pBs2 = infoF2[0]
    pBs3 = infoF3[0]
    return Rs1, Rs2, Rs3, Fs1, Fs2, Fs3, pBs1, pBs2, pBs3, pof

global N
global NT
global NQ
N = 7
NT = 2
NQ = 2

outliers = [0]
outliers12 = [0, 1]
outliers2k = [0]

infoT = segment(Bs, temps, outliers)
infoT12 = segment(Bs12, temps12, outliers12)
infoT2k = segment(Bs2k, temps2k, outliers2k)

TrueTs = GetTs(infoT, infoT12, infoT2k)

infoF = segment(Bs, freqs, outliers)
infoF12 = segment(Bs12, freqs12, outliers12)
infoF2k = segment(Bs2k, freqs2k, outliers2k)

infoQ = segment(Bs, Q, outliers)
infoQ12 = segment(Bs12, Q12, outliers12)
infoQ2k = segment(Bs2k, Q2k, outliers2k)

Rs, Rs12, Rs2k, Fs, Fs12, Fs2k, pBs, pBs12, pBs2k, pof = GetFs(infoF, infoF12, infoF2k, infoQ, infoQ12, infoQ2k, TrueTs)

plt.errorbar(infoF[0], infoF[1]-pof[0], yerr = infoF[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
plt.errorbar(infoF12[0], infoF12[1]-pof[1], yerr = infoF12[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
plt.errorbar(infoF2k[0], infoF2k[1]-pof[2], yerr = infoF2k[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
plt.show()
#
#plt.errorbar(pBs, Rs, yerr = Fs, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
#plt.errorbar(pBs12, Rs12, yerr = Fs12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
#plt.errorbar(pBs2k, Rs2k, yerr = Fs2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.show()

co2k, tkUp, tkDown = getColors(pBs2k)
co0, tkUp0, tkDown0 = getColors(pBs)
co12, tkUp12, tkDown12 = getColors(pBs12)

def getHist(pBs2k, Rs2k, Fs2k, tkUp, tkDown):
    Up2kB = pBs2k[tkUp]
    Up2kR = Rs2k[tkUp]
    Up2kS = Fs2k[tkUp]
    Dw2kB = pBs2k[tkDown]
    Dw2kR = Rs2k[tkDown]
    Dw2kS = Fs2k[tkDown]
    
    UpPreds = interp1d(Up2kB, Up2kR, kind='linear')
    DwPreds = interp1d(Dw2kB, Dw2kR, kind='linear')
    
    mask1 = (Up2kB < max(Dw2kB))
    mask2 = (Up2kB > min(Dw2kB))
    maskH = mask1*mask2
    Up2kBint = Up2kB[maskH]
    Up2kRint = Up2kR[maskH]
    Up2kSint = Up2kS[maskH]
        
    mask1 = (Dw2kB < max(Up2kB))
    mask2 = (Dw2kB > min(Up2kB))
    maskL = mask1*mask2
    Dw2kBint = Dw2kB[maskL]
    Dw2kRint = Dw2kR[maskL]
    Dw2kSint = Dw2kS[maskL]
    
    UpPredPts = UpPreds(Dw2kBint)
    DwPredPts = DwPreds(Up2kBint)
    
    comboB = np.hstack([Up2kBint, Dw2kBint])
    comboH = np.hstack([(Up2kRint - DwPredPts), -(Dw2kRint - UpPredPts)])
    comboS = np.hstack([Up2kSint, Dw2kSint])
    return comboB, comboH, comboS, Up2kB, Up2kR, Up2kS, Dw2kB, Dw2kR, Dw2kS

comboB, comboH, comboS, Up2kB, Up2kR, Up2kS, Dw2kB, Dw2kR, Dw2kS = getHist(pBs2k, Rs2k, Fs2k, tkUp, tkDown)
comboB0, comboH0, comboS0, Up2kB0, Up2kR0, Up2kS0, Dw2kB0, Dw2kR0, Dw2kS0 = getHist(pBs, Rs, Fs, tkUp0, tkDown0)
comboB12, comboH12, comboS12, Up2kB12, Up2kR12, Up2kS12, Dw2kB12, Dw2kR12, Dw2kS12 = getHist(pBs12, Rs12, Fs12, tkUp12, tkDown12)

#plt.errorbar(Up2kB0, Up2kR0, yerr = Up2kS0, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, run 1')
#plt.errorbar(Dw2kB0, Dw2kR0, yerr = Dw2kS0, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, run 1')
#plt.ylabel('magnetization f0')
#plt.show()
#
#plt.errorbar(Up2kB12, Up2kR12, yerr = Up2kS12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, run 2')
#plt.errorbar(Dw2kB12, Dw2kR12, yerr = Dw2kS12, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, run 2')
#plt.ylabel('magnetization f0')
#plt.show()
#
#plt.errorbar(Up2kB, Up2kR, yerr = Up2kS, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.errorbar(Dw2kB, Dw2kR, yerr = Dw2kS, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.ylabel('magnetization f0')
#plt.show()

#plt.errorbar(comboB0, comboH0, yerr = comboS0, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, run 1')
#plt.errorbar(comboB12, comboH12, yerr = comboS12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, run 2')
#plt.title('hystersis in f0')
#plt.show()
#
#plt.errorbar(Ab, As, yerr=Ae, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, run 1')
#plt.errorbar(Ab12, As12, yerr=Ae12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, run 2')
#plt.title('hysteretic tranport data')
#plt.show()
#
#plt.errorbar(comboB, comboH, yerr = comboS, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.title('hysteresis in f0')
#plt.show()
#
#plt.errorbar(Ab2k, As2k, yerr=Ae2k,  marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
#plt.title('hysteretic transport data')
#plt.show()

def split(comboB, comboH, B_low, B_high):
    Hlow = np.zeros(len(B_low))
    Hhigh = np.zeros(len(B_high))
    for i in range(0, len(B_low)):
        a = np.where(comboB == B_low[i])[0]
        if(len(a) == 0):
            Hlow[i] = 0
        else:
            Hlow[i] = comboH[a]
    for i in range(0, len(B_high)):
        a = np.where(comboB == B_high[i])[0]
        if(len(a) == 0):
            Hhigh[i] = 0
        else:
            Hhigh[i] = comboH[a]
    return Hlow, Hhigh

def SigFit(data, poly):
    Bs = data
    
    fit = np.zeros(len(Bs))
    for i in range(0, len(poly)):
        fit += poly[i]*np.power(Bs, i)
    return fit
    
    #When you call the curve_fit code, it can ONLY take things of the form 
    #function(x, args). We need to make a wrapping function to put TFit in 
    #that form.        
def wrapS(x, *args):
    #Note that NTs is a global variable telling which indexes separate
    #the B and run fit coefficients
    poly = list(args)
    return SigFit(x, poly)

Subsignal2k = signal2k - a*s2*Bfield2k - 2

mask2k = [True]*len(Bfield2k)
mask2kf = [False]*len(Bfield2k)
for i in range(1, len(Bfield2k)):
    if(np.abs(Bfield2k[i]) < np.abs(Bfield2k[i-1])):
        mask2k[i] = False
        mask2kf[i] = True
Bfield2k, Subsignal2k, error2k, mask2k, mask2kf = zip(*sorted(zip(Bfield2k, Subsignal2k, error2k, mask2k, mask2kf)))
mask2k = np.array(mask2k)
mask2kf = np.array(mask2kf)
Bfield2k = np.array(Bfield2k)
Subsignal2k = np.array(Subsignal2k)
error2k = np.array(error2k)
print(Bfield2k)
sigSm = savgol_filter(Subsignal2k, 3, 1)
#pp = np.zeros(7)
#pp[0] = 0
#pp[1] = 0
#pp[2] = .03
#pp[3] = .05
#pp[4] = 0
#pp[5] = -.0015
#pp[6] = 0
#ppf, ppcf = curve_fit(lambda Bfield2k, *pf: wrapS(Bfield2k, *pp), Bfield2k, Subsignal2k, p0=pp)
#print(ppf)
#test = wrapS(Bfield2k, *pp)

#Hlow, Hhigh = split(comboB, comboH, B_low2k, B_high2k)
#plt.errorbar(B_low2k, 10*Hlow*B_low2k**2, yerr = error_low2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp down")
#plt.errorbar(B_high2k, -10*Hhigh*B_high2k**2, yerr = error_high2k, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")
#plt.plot(Bfield2k, sigSm)
#plt.errorbar(Bfield2k, Subsignal2k, yerr= error2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#plt.errorbar(B_low2k, signal_low2k - a*s2*B_low2k - 2, yerr = error_low2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = "ramp down")
#plt.errorbar(B_high2k, signal_high2k - a*s2*B_high2k -2, yerr = error_high2k, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = "ramp up")
#plt.axvline(x=-ch2k)
#plt.axvline(x=ch2k)
#plt.axvline(x=-cl, color = 'green')
#plt.axvline(x=cl, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
##plt.legend(loc = 'best')
#plt.show()

plt.errorbar(Bfield2k[mask2k], Subsignal2k[mask2k] - sigSm[mask2k], yerr= error2k[mask2k], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2k$\Omega$, up')
plt.errorbar(Bfield2k[mask2kf], -(Subsignal2k[mask2kf] - sigSm[mask2kf]), yerr= error2k[mask2kf], marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2k$\Omega$, down')

#plt.errorbar(B_low2k, 10*Hlow*B_low2k, yerr = error_low2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp down")
#plt.errorbar(B_high2k, -10*Hhigh*B_high2k, yerr = error_high2k, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")
plt.axvline(x=-ch2k)
plt.axvline(x=ch2k)
plt.axvline(x=-cl, color = 'green')
plt.axvline(x=cl, color = 'green')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc = 'best', prop={'size': 12})
plt.savefig('H2k.pdf', bbox_inches="tight")
plt.show()

plt.errorbar(Bfield2k, np.abs(Subsignal2k - sigSm)/error2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2k$\Omega$')
#plt.errorbar(B_low2k, 10*Hlow*B_low2k, yerr = error_low2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp down")
#plt.errorbar(B_high2k, -10*Hhigh*B_high2k, yerr = error_high2k, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")
plt.axvline(x=-ch2k)
plt.axvline(x=ch2k)
plt.axvline(x=-cl, color = 'green')
plt.axvline(x=cl, color = 'green')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Hysteretic Signal (std. dev)')
plt.legend(loc = 'best')
plt.savefig('Hdev2k.pdf', bbox_inches="tight")
plt.show()

#Bfield, signal, error = makeInfo(Bs, freqs, runs)
#Bfield12, signal12, error12 = makeInfo(Bs12, freqs12, runs12)
subSignal = signal - a*s0*Bfield - 1.3
subSignal12 = signal12 - a*s1*Bfield12 - 1.6

mask = [True]*len(Bfield)
maskf = [False]*len(Bfield)
for i in range(1, len(Bfield)):
    if(np.abs(Bfield[i]) < np.abs(Bfield[i-1])):
        mask[i] = False
        maskf[i] = True
mask12 = [True]*len(Bfield12)
mask12f = [False]*len(Bfield12)
for i in range(1, len(Bfield12)):
    if(np.abs(Bfield12[i]) < np.abs(Bfield12[i-1])):
        mask12[i] = False
        mask12f[i] = True
Bfield, subSignal, error, mask, maskf = zip(*sorted(zip(Bfield, subSignal, error, mask, maskf)))
mask = np.array(mask)
maskf = np.array(maskf)
Bfield = np.array(Bfield)
subSignal = np.array(subSignal)
error = np.array(error)
Bfield12, subSignal12, error12, mask12, mask12f = zip(*sorted(zip(Bfield12, subSignal12, error12, mask12, mask12f)))
mask12 = np.array(mask12)
mask12f = np.array(mask12f)
Bfield12 = np.array(Bfield12)
subSignal12 = np.array(subSignal12)
error12 = np.array(error12)
sigSm0 = savgol_filter(subSignal, 3, 1)
sigSm12 = savgol_filter(subSignal12, 3, 1)

plt.errorbar(Bfield[mask], (subSignal[mask] - sigSm0[mask]), yerr= error[mask], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4k$\Omega$, run 1')
plt.errorbar(Bfield[maskf], -(subSignal[maskf] - sigSm0[maskf]), yerr= error[maskf], marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'red')

plt.errorbar(Bfield12[mask12], subSignal12[mask12] - sigSm12[mask12], yerr= error12[mask12], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '4k$\Omega$, run 2')
plt.errorbar(Bfield12[mask12f], -(subSignal12[mask12f] - sigSm12[mask12f]), yerr= error12[mask12f], marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')

plt.axvline(x=-ch4k)
plt.axvline(x=ch4k)
plt.axvline(x=-cl, color = 'green')
plt.axvline(x=cl, color = 'green')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
plt.savefig('H4k.pdf', bbox_inches="tight")
plt.legend(loc = 'best', prop={'size': 12})
plt.show()

plt.errorbar(Bfield, np.abs(subSignal - sigSm0)/error, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4k$\Omega$, run 1')
plt.errorbar(Bfield12, np.abs(subSignal12 - sigSm12)/error12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '4k$\Omega$, run 2')
plt.axvline(x=-ch4k)
plt.axvline(x=ch4k)
plt.axvline(x=-cl, color = 'green')
plt.axvline(x=cl, color = 'green')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Hysteretic Signal (std. dev)')
plt.savefig('Hdev4k.pdf', bbox_inches="tight")
plt.legend(loc = 'best')
plt.show()

#plt.errorbar(Bfield2k, sigSm, yerr = error2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#plt.axvline(x=-ch2k)
#plt.axvline(x=ch2k)
#plt.axvline(x=-cl, color = 'green')
#plt.axvline(x=cl, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Hysteretic Signal (std. dev)')
#plt.show()
#
#plt.errorbar(Bfield, sigSm0, yerr = error, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#plt.errorbar(Bfield12, sigSm12, yerr = error12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#plt.axvline(x=-ch4k)
#plt.axvline(x=ch4k)
#plt.axvline(x=-cl, color = 'green')
#plt.axvline(x=cl, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Hysteretic Signal (std. dev)')
#plt.show()

EBsB2k, OBsB2k, EHSigB2k, OHSigB2k, EHUncB2k, OHUncB2k = EO(Bfield2k, sigSm, error2k)
EBsB12, OBsB12, EHSigB12, OHSigB12, EHUncB12, OHUncB12 = EO(Bfield12, sigSm12, error12)
EBsB, OBsB, EHSigB, OHSigB, EHUncB, OHUncB = EO(Bfield, sigSm0, error)

#plt.errorbar(EBsB2k, EHSigB2k, yerr = EHUncB2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green')
#plt.axvline(x=ch2k)
#plt.axvline(x=cl, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Even Signal (mHz)')
#plt.show()

def OFit(x, a, b, c):
    return a*x**2 + b*x**4 + c*x**6

OS2k = (a*s2*OBsB2k)/(OHSigB2k + a*s2*OBsB2k) - 1
OHUncB2k = (a*s2*OBsB2k)*OHUncB2k/((OHSigB2k + a*s2*OBsB2k)**2)

#ftry = OFit(EBsB, a*s2, 0, 0)
oFit2k, oCov2k = curve_fit(OFit, OBsB2k, OS2k, sigma = OHUncB2k, p0=[0, 0, 0])
print(oFit2k)
print(oFit2k/np.sqrt(np.diag(oCov2k)))
#ftry = OFit(OBsB2k, oFit2k[0], oFit2k[1], oFit2k[2])
BPlot = np.linspace(0, max(OBsB2k))
FPlot = OFit(BPlot, oFit2k[0], oFit2k[1], oFit2k[2])

plt.plot(BPlot, FPlot)
plt.errorbar(OBsB2k, OS2k, yerr = OHUncB2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2k$\Omega$')
plt.axvline(x=ch2k)
plt.axvline(x=cl, color = 'green')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta$R/$R_{0}$')
plt.legend(loc='best')
plt.savefig('MR2k.pdf', bbox_inches="tight")
plt.show()

#plt.errorbar(EBsB, EHSigB, yerr = EHUncB, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
#plt.errorbar(EBsB12, EHSigB12, yerr = EHUncB12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
#plt.axvline(x=ch4k)
#plt.axvline(x=cl, color = 'green')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Even Signal (mHz)')
#plt.show()

OS = (a*s0*OBsB)/(OHSigB + a*s0*OBsB) - 1
OHUncB = (a*s0*OBsB)*OHUncB/((OHSigB + a*s0*OBsB)**2)
OS12 = (a*s1*OBsB12)/(OHSigB12 + a*s1*OBsB12) - 1
OHUncB12 = (a*s1*OBsB12)*OHUncB12/((OHSigB12 + a*s1*OBsB12)**2)

os = np.hstack((OS, OS12))
obs = np.hstack((OBsB, OBsB12))
ounc = np.hstack((OHUncB, OHUncB12))

oFit, oCov = curve_fit(OFit, obs, os, sigma = ounc, p0=[0, 0, 0])
print(oFit)
print(oFit/np.sqrt(np.diag(oCov)))
FPlot4 = OFit(BPlot, oFit[0], oFit[1], oFit[2])

plt.plot(BPlot, FPlot4)
plt.errorbar(OBsB, OS, yerr = OHUncB, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4k$\Omega$, run 1')
plt.errorbar(OBsB12, OS12, yerr = OHUncB12, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '4k$\Omega$, run 2')
plt.axvline(x=ch4k)
plt.axvline(x=cl, color = 'green')
plt.legend(loc='best')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta$R/$R_{0}$')
plt.savefig('MR4k.pdf', bbox_inches="tight")
plt.show()

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def MagHallFit(x, a, b, c, d, e):
    return a + b*x**2 + c*x*sigmoid((x - e)/d)

def MagFit(x, a, b, c, d, e, f, g, h):
    M = sigmoid((x - e)/d) - sigmoid(-(x + e)/d)
    return a + b*x**2 + c*x*M + f*x + g*x**3 + h*x**2*M


ftry = MagHallFit(EBsB, .5, .15, -1, 1, 3)
AllEBs, AllESig, AllEUnc = np.hstack((EBsB, EBsB12)), np.hstack((EHSigB, EHSigB12)), np.hstack((EHUncB, EHUncB12))
eFit, eCov = curve_fit(MagHallFit, AllEBs, AllESig, sigma = AllEUnc, p0=[.5, .15, -1, 1, 3])
ftry = MagHallFit(EBsB, eFit[0], eFit[1], eFit[2], eFit[3], eFit[4])
BPlot = np.linspace(0, max(AllEBs))
FPlot = MagHallFit(BPlot, eFit[0], eFit[1], eFit[2], eFit[3], eFit[4])
print(eFit)
print(np.sqrt(np.diag(eCov)))
#plt.plot(BPlot, FPlot)
mask = EBsB > .5
mask2 = EBsB12 > .5
plt.errorbar(EBsB[mask], ((EHSigB-eFit[0])/EBsB)[mask], yerr = (EHUncB/EBsB)[mask], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4k$\Omega$, run 1')
plt.errorbar(EBsB12[mask2], ((EHSigB12-eFit[0])/EBsB12)[mask2], yerr = (EHUncB12/EBsB12)[mask2], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '4k$\Omega$, run 2')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Hall Conductivity (AU)')
plt.legend(loc='best')
plt.savefig('HC4k.pdf', bbox_inches="tight")
plt.show()


ftry2k = MagHallFit(EBsB2k, 0, -.15, 1, 1, 3)
eFit2k, eCov2k = curve_fit(MagHallFit, EBsB2k, EHSigB2k, sigma = EHUncB2k, p0=[.5, .15, -1, 1, 3])
ftry2k = MagHallFit(EBsB, eFit2k[0], eFit2k[1], eFit2k[2], eFit2k[3], eFit2k[4])
BPlot2k = np.linspace(0, max(EBsB2k))
FPlot2k = MagHallFit(BPlot2k, eFit2k[0], eFit2k[1], eFit2k[2], eFit2k[3], eFit2k[4])
print(eFit2k)
print(np.sqrt(np.diag(eCov2k)))

#plt.plot(BPlot2k, FPlot2k)
mask = EBsB2k > .5
plt.errorbar(EBsB2k[mask], ((EHSigB2k- eFit2k[0])/EBsB2k)[mask], yerr = (EHUncB2k/EBsB2k)[mask], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2k$\Omega$')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Hall Conductivity (AU)')
plt.legend(loc='best')
plt.savefig('HC2k.pdf', bbox_inches="tight")
plt.show()


#plt.errorbar(Bfield, sigSm0, yerr = error, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '4kOhm, Run 1')
#plt.errorbar(Bfield12, sigSm12, yerr = error12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Non-Hist $\delta f_{0}(\pm V)$ (mHz)')
#plt.legend(loc= "best")
#plt.show()

#tb = np.linspace(min(Bfield2k), max(Bfield2k))
#p0 = [eFit2k[0], eFit2k[1], eFit2k[2], eFit2k[3], eFit2k[4], 0, 0, 0]
#MFit = p0
#MFit, MCov = curve_fit(MagFit, Bfield2k, sigSm, sigma = error2k, p0=p0, maxfev=5000)
##print(MFit)
#tt = MagFit(tb, MFit[0], MFit[1], MFit[2], MFit[3], MFit[4], MFit[5], MFit[6], MFit[7])
#print(MFit)
#print(np.sqrt(np.diag(MCov)))
#print(MFit/np.sqrt(np.diag(MCov)))
#plt.plot(tb, tt)
#plt.errorbar(Bfield2k, sigSm, yerr = error2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = '2kOhm')
#plt.xlabel('Magnetic Field (T)')
#plt.ylabel('Non-Hist $\delta f_{0}(\pm V)$ (mHz)')
#plt.show()