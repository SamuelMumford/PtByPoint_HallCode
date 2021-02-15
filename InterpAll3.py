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

fname = 'C:/Users/sammy/Downloads/Cooldown_all_11_18_20_cool.txt'
f2 = 'C:/Users/sammy/Downloads/Cooldown_all_11_18_20_coolL.txt'#f2 = 'C:/Users/sammy/Downloads/11_5_20_cool1L.txt'
fname12 = 'C:/Users/sammy/Downloads/12_9_20_coolEdit.txt'
f122 = 'C:/Users/sammy/Downloads/12_9_20_coolEditL.txt' #f2 = 'C:/Users/sammy/Downloads/11_5_20_cool1L.txt'
fname2k = 'C:/Users/sammy/Downloads/Cooldown_all_1_14_20_cool.txt'
f22k = 'C:/Users/sammy/Downloads/Cooldown_all_1_14_20_coolL.txt'#f2 = 'C:/Users/sammy/Downloads/11_5_20_cool1L.txt'



Fdat = np.loadtxt(fname, delimiter="\t")
F2 = np.loadtxt(f2, delimiter="\t")
startcut = 0
endcut = 3450#-450
keep = True
#keep = True
freqs = Fdat[startcut:endcut, 2]
runs = Fdat[startcut:endcut, 3]
currs = Fdat[startcut:endcut, 5]
Q = F2[startcut:endcut, 3]
Q = np.abs(Q)
Bs = currs*.875/5

Fdat12 = np.loadtxt(fname12, delimiter="\t")
F122 = np.loadtxt(f122, delimiter="\t")
startcut = 0
endcut = -380
keep = True
#keep = True
freqs12 = Fdat12[startcut:endcut, 2]
runs12 = Fdat12[startcut:endcut, 3]
currs12 = Fdat12[startcut:endcut, 5]
Q12 = F122[startcut:endcut, 3]
Q12 = np.abs(Q12)
Bs12 = currs12*.875/5

Fdat2k = np.loadtxt(fname2k, delimiter="\t")
F22k = np.loadtxt(f22k, delimiter="\t")
startcut = 0
endcut = -5#4481#-40
keep = True
#keep = True
freqs2k = Fdat2k[startcut:endcut, 2]
runs2k = Fdat2k[startcut:endcut, 3]
currs2k = Fdat2k[startcut:endcut, 5]
Q2k = F22k[startcut:endcut, 3]
Q2k = np.abs(Q2k)
Bs2k = currs2k*.875/5


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

inf = segmentEO(Bs, freqs, runs)
global N
N = 3
Ntot = 11

mask = [True]*len(inf[0])

if(not keep):
    mask[18] = False
a = inf[0][mask]
b = inf[1][mask]
c = inf[2][mask]

info = [a, b, c]


#Magnetic field (T)
Bfield = info[0]
#Signal in mHz
signal = 1000*info[1]
#Error in mHz
error = 1000*info[2]

inf12 = segmentEO(Bs12, freqs12, runs12)

mask12 = [True]*len(inf12[0])

if(not keep):
    mask12[18] = False
a12 = inf12[0][mask12]
b12 = inf12[1][mask12]
c12 = inf12[2][mask12]

info12 = [a12, b12, c12]


#Magnetic field (T)
Bfield = info[0]
#Signal in mHz
signal = 1000*info[1]
#Error in mHz
error = 1000*info[2]

#Magnetic field (T)
Bfield12 = info12[0]
#Signal in mHz
signal12 = 1000*info12[1]
#Error in mHz
error12 = 1000*info12[2]


inf2k = segmentEO(Bs2k, freqs2k, runs2k)
mask2k = [True]*len(inf2k[0])

if(not keep):
    mask2k[18] = False
a2k = inf2k[0][mask2k]
b2k = inf2k[1][mask2k]
c2k = inf2k[2][mask2k]

info2k = [a2k, b2k, c2k]


#Magnetic field (T)
Bfield2k = info2k[0]
#Signal in mHz
signal2k = 1000*info2k[1]
#Error in mHz
error2k = 1000*info2k[2]

#popt,pcov=curve_fit(fit, info[0], info[1], p0=(1,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), sigma = info[2]

print(Bfield)
#print(len(Bfield))
Bcolor = []
for i in range(0,len(Bfield)):
    Bcolor.append('blue')
#print(Bcolor)
#Blue for points going up from 0 
#Red for points going down from high field
    #-4 so we don't change the colors of the last 4 points that were taken at the end 
for i in range(0, len(Bfield)):
    if np.abs(Bfield[i]) >= np.abs(Bfield[i-1]):
        Bcolor[i] = 'blue' 
    if np.abs(Bfield[i]) < np.abs(Bfield[i-1]):
        Bcolor[i] = 'red'
#Make first 0 point blue
Bcolor[0] = 'blue'


Bcolor12 = []
for i in range(0,len(Bfield12)):
    Bcolor12.append('blue')
#print(Bcolor)
#Blue for points going up from 0 
#Red for points going down from high field
    #-4 so we don't change the colors of the last 4 points that were taken at the end 
for i in range(0, len(Bfield12)):
    if np.abs(Bfield12[i]) >= np.abs(Bfield12[i-1]):
        Bcolor12[i] = 'blue' 
    if np.abs(Bfield12[i]) < np.abs(Bfield12[i-1]):
        Bcolor12[i] = 'red'
#Make first 0 point blue
Bcolor12[0] = 'blue'

Bcolor2k = []
for i in range(0,len(Bfield2k)):
    Bcolor2k.append('blue')
#print(Bcolor)
#Blue for points going up from 0 
#Red for points going down from high field
    #-4 so we don't change the colors of the last 4 points that were taken at the end 
for i in range(0, len(Bfield2k)-4):
    if np.abs(Bfield2k[i]) >= np.abs(Bfield2k[i-1]):
        Bcolor2k[i] = 'blue' 
    if np.abs(Bfield2k[i]) < np.abs(Bfield2k[i-1]):
        Bcolor2k[i] = 'red'
#Make first 0 point blue
Bcolor2k[0] = 'blue'

#fields going to higher magnitude fields
mask_B_high = [True]*len(Bfield)
#going to lower magnitude fields (towards zero)
mask_B_low = [True]*len(Bfield)

for i in range(0, len(Bcolor)):
    if Bcolor[i] != 'blue':
        mask_B_high[i] = False
    if Bcolor[i] != 'red':
        mask_B_low[i] = False
        
#fields going to higher magnitude fields
mask_B_high12 = [True]*len(Bfield12)
#going to lower magnitude fields (towards zero)
mask_B_low12 = [True]*len(Bfield12)

for i in range(0, len(Bcolor12)):
    if Bcolor12[i] != 'blue':
        mask_B_high12[i] = False
    if Bcolor12[i] != 'red':
        mask_B_low12[i] = False
        
#fields going to higher magnitude fields
mask_B_high2k = [True]*len(Bfield2k)
#going to lower magnitude fields (towards zero)
mask_B_low2k = [True]*len(Bfield2k)

for i in range(0, len(Bcolor2k)):
    if Bcolor2k[i] != 'blue':
        mask_B_high2k[i] = False
    if Bcolor2k[i] != 'red':
        mask_B_low2k[i] = False


B_high = Bfield[mask_B_high]
B_low = Bfield[mask_B_low]
signal_high = signal[mask_B_high]
signal_low = signal[mask_B_low]
error_high = error[mask_B_high]
error_low = error[mask_B_low]

B_high12 = Bfield12[mask_B_high]
B_low12 = Bfield12[mask_B_low]
signal_high12 = signal12[mask_B_high]
signal_low12 = signal12[mask_B_low]
error_high12 = error12[mask_B_high]
error_low12 = error12[mask_B_low]

B_high2k = Bfield2k[mask_B_high2k]
B_low2k = Bfield2k[mask_B_low2k]
signal_high2k = signal2k[mask_B_high2k]
signal_low2k = signal2k[mask_B_low2k]
error_high2k = error2k[mask_B_high2k]
error_low2k = error2k[mask_B_low2k]

_, Info = consolidate([B_high, signal_high, error_high])
B_high = Info[0]
signal_high = Info[1]
error_high = Info[2]

_, Info12 = consolidate([B_high12, signal_high12, error_high12])
B_high12 = Info12[0]
signal_high12 = Info12[1]
error_high12 = Info12[2]

_, Info2k = consolidate([B_high2k, signal_high2k, error_high2k])
B_high2k = Info2k[0]
signal_high2k = Info2k[1]
error_high2k = Info2k[2]

B_high, signal_high, error_high = zip(*sorted(zip(B_high, signal_high, error_high)))
B_low, signal_low, error_low = zip(*sorted(zip(B_low, signal_low, error_low)))

B_high = np.array(B_high)
signal_high = np.array(signal_high)
error_high = np.array(error_high)

B_low = np.array(B_low)
signal_low = np.array(signal_low)
error_low = np.array(error_low)

B_high12, signal_high12, error_high12 = zip(*sorted(zip(B_high12, signal_high12, error_high12)))
B_low12, signal_low12, error_low12 = zip(*sorted(zip(B_low12, signal_low12, error_low12)))

B_high12 = np.array(B_high12)
signal_high12 = np.array(signal_high12)
error_high12 = np.array(error_high12)

B_low12 = np.array(B_low12)
signal_low12 = np.array(signal_low12)
error_low12 = np.array(error_low12)

B_high2k, signal_high2k, error_high2k = zip(*sorted(zip(B_high2k, signal_high2k, error_high2k)))
B_low2k, signal_low2k, error_low2k = zip(*sorted(zip(B_low2k, signal_low2k, error_low2k)))

B_high2k = np.array(B_high2k)
signal_high2k = np.array(signal_high2k)
error_high2k = np.array(error_high2k)

B_low2k = np.array(B_low2k)
signal_low2k = np.array(signal_low2k)
error_low2k = np.array(error_low2k)


##############

HighPreds = interp1d(B_high, signal_high, kind='cubic')
LowPreds = interp1d(B_low, signal_low, kind='cubic')
VarHighPreds = interp1d(B_high, (1.0*error_high)**2, kind='cubic')
VarLowPreds = interp1d(B_low, (1.0*error_low)**2, kind='cubic')

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

HighPreds12 = interp1d(B_high12, signal_high12, kind='cubic')
LowPreds12 = interp1d(B_low12, signal_low12, kind='cubic')
VarHighPreds12 = interp1d(B_high12, (1.0*error_high12)**2, kind='cubic')
VarLowPreds12 = interp1d(B_low12, (1.0*error_low12)**2, kind='cubic')

mask112 = (B_high12 < max(B_low12))
mask212 = (B_high12 > min(B_low12))
maskH12 = mask112*mask212
B_sub_high12 = B_high12[maskH12]
signal_sub_high12 = signal_high12[maskH12]
error_sub_high12 = error_high12[maskH12]
mask112 = (B_low12 < max(B_high12))
mask212 = (B_low12 > min(B_high12))
maskL12 = mask112*mask212
B_sub_low12 = B_low12[maskL12]
signal_sub_low12 = signal_low12[maskL12]
error_sub_low12 = error_low12[maskL12]

HighPredPts12 = HighPreds12(B_sub_low12)
LowPredPts12 = LowPreds12(B_sub_high12)
HighPredVar12 = VarHighPreds12(B_sub_low12)
LowPredVar12 = VarLowPreds12(B_sub_high12)


HighPreds2k = interp1d(B_high2k, signal_high2k, kind='cubic')
LowPreds2k = interp1d(B_low2k, signal_low2k, kind='cubic')
VarHighPreds2k = interp1d(B_high2k, (1.0*error_high2k)**2, kind='cubic')
VarLowPreds2k = interp1d(B_low2k, (1.0*error_low2k)**2, kind='cubic')

mask12k = (B_high2k < max(B_low2k))
mask22k = (B_high2k > min(B_low2k))
maskH2k = mask12k*mask22k
B_sub_high2k = B_high2k[maskH2k]
signal_sub_high2k = signal_high2k[maskH2k]
error_sub_high2k = error_high2k[maskH2k]
mask12k = (B_low2k < max(B_high2k))
mask22k = (B_low2k > min(B_high2k))
maskL2k = mask12k*mask22k
B_sub_low2k = B_low2k[maskL2k]
signal_sub_low2k = signal_low2k[maskL2k]
error_sub_low2k = error_low2k[maskL2k]

HighPredPts2k = HighPreds2k(B_sub_low2k)
LowPredPts2k = LowPreds2k(B_sub_high2k)
HighPredVar2k = VarHighPreds2k(B_sub_low2k)
LowPredVar2k = VarLowPreds2k(B_sub_high2k)

plt.plot(B_sub_low, HighPredPts)
plt.plot(B_sub_high, LowPredPts)
plt.errorbar(B_low, signal_low, yerr = error_low, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = "ramp down")
plt.errorbar(B_high, signal_high, yerr = error_high, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")

plt.plot(B_sub_low12, HighPredPts12)
plt.plot(B_sub_high12, LowPredPts12)
plt.errorbar(B_low12, signal_low12, yerr = error_low12, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = "ramp down")
plt.errorbar(B_high12, signal_high12, yerr = error_high12, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")

plt.plot(B_sub_low2k, HighPredPts2k)
plt.plot(B_sub_high2k, LowPredPts2k)
plt.errorbar(B_low2k, signal_low2k, yerr = error_low2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = "ramp down")
plt.errorbar(B_high2k, signal_high2k, yerr = error_high2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = "ramp up")


plt.xlabel('Magnetic Field (T)')
plt.ylabel('$\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc = 'best')
plt.show()

Sig1 = (HighPredPts - signal_sub_low)
Sig2 = (LowPredPts - signal_sub_high)
Err1 = np.sqrt(HighPredVar + error_sub_low**2)
Err2 = np.sqrt(LowPredVar + error_sub_high**2)

Sig112 = (HighPredPts12 - signal_sub_low12)
Sig212 = (LowPredPts12 - signal_sub_high12)
Err112 = np.sqrt(HighPredVar12 + error_sub_low12**2)
Err212 = np.sqrt(LowPredVar12 + error_sub_high12**2)

Sig12k = (HighPredPts2k - signal_sub_low2k)
Sig22k = (LowPredPts2k - signal_sub_high2k)
Err12k = np.sqrt(HighPredVar2k + error_sub_low2k**2)
Err22k = np.sqrt(LowPredVar2k + error_sub_high2k**2)

AllBs = np.hstack((B_sub_low, B_sub_high))
AllSs = np.hstack((Sig1, -Sig2))
AllEs = np.hstack((Err1, Err2))

AllBs12 = np.hstack((B_sub_low12, B_sub_high12))
AllSs12 = np.hstack((Sig112, -Sig212))
AllEs12 = np.hstack((Err112, Err212))

AllBs2k = np.hstack((B_sub_low2k, B_sub_high2k))
AllSs2k = np.hstack((Sig12k, -Sig22k))
AllEs2k = np.hstack((Err12k, Err22k))

plt.errorbar(AllBs, AllSs, yerr = AllEs, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
plt.errorbar(AllBs12, AllSs12, yerr = AllEs12, marker = 'x', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
plt.errorbar(AllBs2k, AllSs2k, yerr = AllEs2k, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, color = 'black')

plt.xlabel('Magnetic Field (T)')
plt.ylabel('Hysteretic $\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc = 'best')
plt.show()

Dic, _ = consolidate([AllBs, AllSs, AllEs])
_, EInfo = consolidateEven(Dic)
_, OInfo = consolidateOdd(Dic)
EBs = EInfo[0]
EHSig = EInfo[1]
EHUnc = EInfo[2]
OBs = OInfo[0]
OHSig = OInfo[1]
OHUnc = OInfo[2]

plt.errorbar(EBs, EHSig, yerr = EHUnc, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Even Hysteretic $\delta f_{0}(\pm V)$ (mHz)')
plt.show()

plt.errorbar(OBs, OHSig, yerr = OHUnc, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Odd Hysteretic $\delta f_{0}(\pm V)$ (mHz)')
plt.show()

AllBs = np.hstack((B_sub_low, B_sub_high))
LandPLs = np.hstack((signal_sub_low, LowPredPts))
LVandPLVs = np.sqrt(np.hstack((error_sub_low**2, LowPredVar)))
HandPHs = np.hstack((HighPredPts, signal_sub_high))
HVandPHVs = np.sqrt(np.hstack((HighPredVar, error_sub_high**2)))

SigBase = (LandPLs + HandPHs)/2
ErrBase = np.sqrt((LVandPLVs**2 + HVandPHVs**2))/2


AllBs12 = np.hstack((B_sub_low12, B_sub_high12))
LandPLs12 = np.hstack((signal_sub_low12, LowPredPts12))
LVandPLVs12 = np.sqrt(np.hstack((error_sub_low12**2, LowPredVar12)))
HandPHs12 = np.hstack((HighPredPts12, signal_sub_high12))
HVandPHVs12 = np.sqrt(np.hstack((HighPredVar12, error_sub_high12**2)))

SigBase12 = (LandPLs12 + HandPHs12)/2
ErrBase12 = np.sqrt((LVandPLVs12**2 + HVandPHVs12**2))/2

AllBs2k = np.hstack((B_sub_low2k, B_sub_high2k))
LandPLs2k = np.hstack((signal_sub_low2k, LowPredPts2k))
LVandPLVs2k = np.sqrt(np.hstack((error_sub_low2k**2, LowPredVar2k)))
HandPHs2k = np.hstack((HighPredPts2k, signal_sub_high2k))
HVandPHVs2k = np.sqrt(np.hstack((HighPredVar2k, error_sub_high2k**2)))

SigBase2k = (LandPLs2k + HandPHs2k)/2
ErrBase2k = np.sqrt((LVandPLVs2k**2 + HVandPHVs2k**2))/2

# plt.errorbar(AllBs, LandPLs, yerr = LVandPLVs, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red')
# plt.errorbar(AllBs, HandPHs, yerr = HVandPHVs, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue')
# plt.errorbar(AllBs, SigBase, yerr = ErrBase, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black')
# plt.xlabel('Magnetic Field (T)')
# plt.ylabel('Shift for Up and Down (mHz)')
# plt.show()
AllBs0 = np.copy(AllBs)
l = len(Bfield)
for i in range(0, l):
    listy=np.where(AllBs == Bfield[i])
    b = listy[0]
    if (len(b) == 0):
        print(Bfield[i])
        AllBs = np.hstack((AllBs, [Bfield[i]]))
        SigBase = np.hstack((SigBase, [signal[i]]))
        ErrBase = np.hstack((ErrBase, [error[i]]))
        
AllBs012 = np.copy(AllBs12)
l12 = len(Bfield12)
for i in range(0, l12):
    listy12=np.where(AllBs12 == Bfield12[i])
    b = listy12[0]
    if (len(b) == 0):
        print(Bfield12[i])
        AllBs12 = np.hstack((AllBs12, [Bfield12[i]]))
        SigBase12 = np.hstack((SigBase12, [signal12[i]]))
        ErrBase12 = np.hstack((ErrBase12, [error12[i]]))

AllBs02k = np.copy(AllBs2k)
l = len(Bfield2k)
for i in range(0, l):
    listy2k=np.where(AllBs2k == Bfield2k[i])
    b = listy2k[0]
    if (len(b) == 0):
        print(Bfield2k[i])
        AllBs2k = np.hstack((AllBs2k, [Bfield2k[i]]))
        SigBase2k = np.hstack((SigBase2k, [signal2k[i]]))
        ErrBase2k = np.hstack((ErrBase2k, [error2k[i]]))
        
DicBase, _ = consolidate([AllBs, SigBase, ErrBase])
_, EInfoB = consolidateEven(DicBase)
_, OInfoB = consolidateOdd(DicBase)

EBsB = EInfoB[0]
EHSigB = EInfoB[1]
EHUncB = EInfoB[2]
OBsB = OInfoB[0]
OHSigB = OInfoB[1]
OHUncB = OInfoB[2]


DicBase12, _ = consolidate([AllBs12, SigBase12, ErrBase12])
_, EInfoB12 = consolidateEven(DicBase12)
_, OInfoB12 = consolidateOdd(DicBase12)

EBsB12 = EInfoB12[0]
EHSigB12 = EInfoB12[1]
EHUncB12 = EInfoB12[2]
OBsB12 = OInfoB12[0]
OHSigB12 = OInfoB12[1]
OHUncB12 = OInfoB12[2]

DicBase2k, _ = consolidate([AllBs2k, SigBase2k, ErrBase2k])
_, EInfoB2k = consolidateEven(DicBase2k)
_, OInfoB2k = consolidateOdd(DicBase2k)

EBsB2k = EInfoB2k[0]
EHSigB2k = EInfoB2k[1]
EHUncB2k = EInfoB2k[2]
OBsB2k = OInfoB2k[0]
OHSigB2k = OInfoB2k[1]
OHUncB2k = OInfoB2k[2]

plt.errorbar(EBsB, EHSigB, yerr = EHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
plt.errorbar(EBsB12, EHSigB12, yerr = EHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
plt.errorbar(EBsB2k, EHSigB2k, yerr = EHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Even Base $\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc= "best")
plt.show()

plt.errorbar(EBsB*(3.5/3.65), EHSigB, yerr = EHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
plt.errorbar(EBsB12, EHSigB12, yerr = EHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
plt.errorbar(EBsB2k*(1.9/3.65), EHSigB2k-.8, yerr = EHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm, scaled B')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Even Base $\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc= "best")
plt.show()

delB = .001

plt.errorbar(1/(EBsB + delB), EHSigB, yerr = EHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
plt.errorbar(1/(EBsB12 + delB), EHSigB12, yerr = EHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
plt.errorbar(1/(EBsB2k*(1.9/3.65) + delB), EHSigB2k-.8, yerr = EHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm, scaled B')
plt.xlabel('1/Magnetic Field (T)')
plt.xlim(0, 2)
plt.ylabel('Even Base $\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc= "best")
plt.show()

plt.errorbar(OBsB, OHSigB, yerr = OHUncB, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'black', label = '4kOhm, Run 1')
plt.errorbar(OBsB12, OHSigB12, yerr = OHUncB12, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = '4kOhm, Run 2')
plt.errorbar(OBsB2k, OHSigB2k, yerr = OHUncB2k, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = '2kOhm')
plt.xlabel('Magnetic Field (T)')
plt.ylabel('Odd Base $\delta f_{0}(\pm V)$ (mHz)')
plt.legend(loc = 'best')
plt.show()