# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:27:54 2020

@author: sammy
"""


import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fname = 'C:/Users/sammy/Downloads/12_9_20_coolEdit.txt'
f2 = 'C:/Users/sammy/Downloads/12_9_20_coolEditL.txt'
#fname = 'C:/Users/sammy/Downloads/11_5_20_cool1.txt'
#f2 = 'C:/Users/sammy/Downloads/11_5_20_cool1L.txt'

Fdat = np.loadtxt(fname, delimiter="\t")
F2 = np.loadtxt(f2, delimiter="\t")
startcut = 0
endcut = -5#4481#-40
keep = False
freqs = Fdat[startcut:endcut, 2]
runs = Fdat[startcut:endcut, 3]
currs = Fdat[startcut:endcut, 5]
Q = F2[startcut:endcut, 3]
Q = np.abs(Q)
Bs = currs*.875/5

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
            sigtot = np.sqrt(sigE**2 + sigO**2)/2
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

def fit(x,a,b, c, d, e, f, g, h): 
     return a + b*x + c*x**3 + d*x**5 + e*x**7 + f*x**2 + g*x**4 + h*x**6

inf = segmentEO(Bs, freqs, runs)
print(inf)
mask = [True]*len(inf[0])
if(not keep):
    mask[18] = False
a = inf[0][mask]
b = inf[1][mask]
c = inf[2][mask]
print(a)
info = [a, b, c]
print(info)

popt,pcov=curve_fit(fit, info[0], info[1], p0=(1,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), sigma = info[2])
print(popt)
pltX = np.linspace(np.min(info[0]), np.max(info[0]), 100)
pltY = fit(pltX, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
plt.plot(pltX, 1000*pltY)
colors = cm.jet(np.linspace(0, 1, len(info[0])))
for i in range(0, len(info[0])):
    plt.errorbar(info[0][i], 1000*info[1][i], yerr = 1000*info[2][i], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = colors[i])
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
plt.savefig("1120.pdf")
plt.show()

plt.plot(pltX, np.zeros(len(pltX)))
for i in range(0, len(info[0])):
    plt.errorbar(info[0][i], 1000*(info[1][i] - fit(info[0][i], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])), yerr = 1000*info[2][i], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = colors[i])
plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ Residual (mHz)')
plt.savefig("1120O.pdf")
plt.show()

ShDict, ShList = consolidate(info)
plt.errorbar(ShList[0], ShList[1], yerr = ShList[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2)
plt.show()

def fitE(x, a, b, c, d, e):
    return a + b*np.abs(x) + c*x**2 + d*x**4 + e*x**6

EvDict, EvList = consolidateEven(ShDict)
eBs = EvList[0]
ps = (3.46, -2.3, .8, -.02, 0)
ep,epc=curve_fit(fitE, eBs, EvList[1], p0=ps, sigma = EvList[2])
xs = np.linspace(0, np.max(eBs))
q = ep


plt.errorbar(EvList[0], 1000*EvList[1], 1000*EvList[2], c = 'red', marker = 'o', linewidth = 0, elinewidth=1, label = "Cooldown 1", capsize = 2.5)
plt.plot(xs, 1000*fitE(xs, q[0], q[1], q[2], q[3], q[4]))
plt.ylabel('Even $\delta f_{0}(\pm V)$ (mHz)')
plt.xlabel('Amplitude of Magnetic Field (T)')
#plt.legend(numpoints=1, loc = 2)
plt.xlim(-.5, 7.5)
plt.tight_layout()
#plt.savefig("EvenPartField.pdf")
plt.show()

devl = np.abs((EvList[1] - fitE(EvList[0], q[0], q[1], q[2], q[3], q[4]))/EvList[2])
print(devl)
print(np.mean(devl))



plt.errorbar(EvList[0], 1000*(EvList[1]-q[0]), 1000*EvList[2], c = 'red', marker = 'o', linewidth = 0, elinewidth=1, label = "Cooldown 1", capsize = 2.5)
plt.plot(xs, 1000*fitE(xs, 0, q[1], q[2], q[3], q[4]))
plt.ylabel('Even $\delta f_{0}(\pm V)$ (mHz)')
plt.xlabel('Amplitude of Magnetic Field (T)')
#plt.legend(numpoints=1, loc = 2)
plt.xlim(-.5, 7.5)
plt.tight_layout()
#plt.savefig("EvenPartField.pdf")
plt.show()

OdDict, OdList = consolidateOdd(ShDict)
def fitO(x, a):
    return a*x
oBs = OdList[0]
ps = (0)
ep,epc=curve_fit(fitO, oBs, OdList[1], p0=ps, sigma = OdList[2])
xs = np.linspace(0, np.max(oBs))
q = ep

plt.errorbar(OdList[0], 1000*OdList[1], 1000*OdList[2], c = 'red', marker = 'o', linewidth = 0, elinewidth=1, label = "Cooldown 1", capsize = 2.5)
#plt.plot(xs, 1000*fitO(xs, q[0]))
plt.ylabel('Odd $\delta f_{0}(\pm V)$ (mHz)')
plt.xlabel('Amplitude of Magnetic Field (T)')
#plt.legend(numpoints=1, loc = 2)
plt.xlim(-.5, 7.5)
plt.tight_layout()
#plt.savefig("EvenPartField.pdf")
plt.show()

plt.errorbar(OdList[0], 1000*(OdList[1]-q[0]*OdList[0]), 1000*OdList[2], c = 'red', marker = 'o', linewidth = 0, elinewidth=1, label = "Cooldown 1", capsize = 2.5)
plt.plot(xs, 1000*fitO(xs, 0))
plt.ylabel('Odd $\delta f_{0}(\pm V)$ (mHz)')
plt.xlabel('Amplitude of Magnetic Field (T)')
#plt.legend(numpoints=1, loc = 2)
plt.xlim(-.5, 7.5)
plt.tight_layout()
#plt.savefig("EvenPartField.pdf")
plt.show()