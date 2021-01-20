# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:05:52 2021

@author: sammy
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt


f129 = 'C:/Users/sammy/Downloads/12_9_20_coolEdit.txt'
f129L = 'C:/Users/sammy/Downloads/12_9_20_coolEditL.txt'
f1118 = 'C:/Users/sammy/Downloads/Cooldown_all_11_18_20_cool.txt'
f1118L = 'C:/Users/sammy/Downloads/Cooldown_all_11_18_20_coolL.txt'
f114 = 'C:/Users/sammy/Downloads/Cooldown_all_1_14_20_cool.txt'
f114L = 'C:/Users/sammy/Downloads/Cooldown_all_1_14_20_coolL.txt'

name1 = '4kOhm, 12/9'#'12/9/20'
name2 = '4kOhm, 11/18'#'11/18/20'
name3 = '2kOhm, 1/14'#'1/14/21'

Fdat = np.loadtxt(f129, delimiter="\t")
F2 = np.loadtxt(f129L, delimiter="\t")
startcut1 = 0
endcut1 = -340#-340
freqs1 = Fdat[startcut1:endcut1, 2]
runs1 = Fdat[startcut1:endcut1, 3]
temps1 = Fdat[startcut1:endcut1, 4]
currs1 = Fdat[startcut1:endcut1, 5]
Q1 = F2[startcut1:endcut1, 3]
dc1 = Fdat[startcut1:endcut1, 0]
Q1 = np.abs(Q1)
Bs1 = currs1*.875/5
outliers1 = []

Fdat = np.loadtxt(f1118, delimiter="\t")
F2 = np.loadtxt(f1118L, delimiter="\t")
startcut1 = 0
endcut1 = -470
freqs2 = Fdat[startcut1:endcut1, 2]
runs2 = Fdat[startcut1:endcut1, 3]
temps2 = Fdat[startcut1:endcut1, 4]
currs2 = Fdat[startcut1:endcut1, 5]
Q2 = F2[startcut1:endcut1, 3]
dc2 = Fdat[startcut1:endcut1, 0]
Q2 = np.abs(Q2)
Bs2 = currs2*.875/5
outliers2 = []

Fdat = np.loadtxt(f114, delimiter="\t")
F2 = np.loadtxt(f114L, delimiter="\t")
startcut1 = 0
endcut1 = -340
freqs3 = Fdat[startcut1:endcut1, 2]
runs3 = Fdat[startcut1:endcut1, 3]
temps3 = Fdat[startcut1:endcut1, 4]
currs3 = Fdat[startcut1:endcut1, 5]
Q3 = F2[startcut1:endcut1, 3]
dc3 = Fdat[startcut1:endcut1, 0]
Q3 = np.abs(Q3)
Bs3 = currs3*.875/5
outliers3 = []

def segment(Bs, temp, outliers):
    start = 0
    startB = Bs[start]
    info = []
    index = -1
    count = 0
    if(len(outliers) != 0):
        index = outliers[0]
    for i in range(0, len(Bs)):
        if(Bs[i] != startB):
            sub = temp[start:i]
            mean = np.mean(sub)
            dev = np.std(sub)/3
            if(i != index):
                info.append([startB, mean, dev])
            else:
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

infoT1 = segment(Bs1, temps1, outliers1)
infoT2 = segment(Bs2, temps2, outliers2)
infoT3 = segment(Bs3, temps3, outliers3)

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

print(popt)
print(np.mean(np.abs(AllTs - fitty)/AllSigs))

pCopy = np.copy(popt)
pCopy[0] = 0
pCopy[1] = 0
pCopy[2] = 0
fittySub = wrapT(data, *pCopy)
TrueTs = AllTs - fittySub

Ts1 = TrueTs[Cat1 > 0]
Ts2 = TrueTs[Cat2 > 0]
Ts3 = TrueTs[Cat3 > 0]
sTs1 = AllSigs[Cat1 > 0]
sTs2 = AllSigs[Cat2 > 0]
sTs3 = AllSigs[Cat3 > 0]
pBs1 = infoT1[0]
pBs2 = infoT2[0]
pBs3 = infoT3[0]

plt.errorbar(np.abs(pBs1), Ts1, yerr = sTs1, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
plt.errorbar(np.abs(pBs2), Ts2, yerr = sTs2, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
plt.errorbar(np.abs(pBs3), Ts3, yerr = sTs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
plt.show()

infoF1 = segment(Bs1, freqs1, outliers1)
infoF2 = segment(Bs2, freqs2, outliers2)
infoF3 = segment(Bs3, freqs3, outliers3)

AllFs = np.hstack((infoF1[1], infoF2[1], infoF3[1]))
AllFSigs = np.hstack((infoF1[2], infoF2[2], infoF3[2]))

infoQ1 = segment(Bs1, Q1, outliers1)
infoQ2 = segment(Bs2, Q2, outliers2)
infoQ3 = segment(Bs3, Q3, outliers3)
AllQs = np.hstack((infoQ1[1], infoQ2[1], infoQ3[1]))

dataF0 = np.vstack((Cat1, Cat2, Cat3, AllBs, TrueTs, AllQs)).T
global N
global NT
global NQ
N = 5
NT = 1
NQ = 1
NTot = N + 3 + NT + NQ

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

pf = np.zeros(NTot)
pf[0] = np.mean(infoF1[1])
pf[1] = np.mean(infoF2[1])
pf[2] = np.mean(infoF3[1])
pf[3] = .3
pf[4] = 1

pof, pcf = curve_fit(lambda dataF0, *pf: wrapF(dataF0, *pf), dataF0, AllFs, p0=pf)
print(pof)
print((pof/np.sqrt(np.diag(pcf)))[3:])
EFs = wrapF(dataF0, *pof)
pc = np.copy(pof)
for i in range(3, len(pc)):
    pc[i] = 0
offy = wrapF(dataF0, *pc)
plt.errorbar(np.abs(AllBs), AllFs - offy, yerr = AllFSigs, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
plt.plot(np.abs(AllBs), EFs - offy, '.')
plt.ylabel('$<f_{0}>$ (Hz)')
plt.xlabel('B (T)')
plt.show()

plt.errorbar(np.abs(AllBs), AllFs - EFs, yerr = AllFSigs, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
plt.show()

Rs1 = (AllFs - EFs)[Cat1 > 0]
Rs2 = (AllFs - EFs)[Cat2 > 0]
Rs3 = (AllFs - EFs)[Cat3 > 0]
Fs1 = AllFSigs[Cat1 > 0]
Fs2 = AllFSigs[Cat2 > 0]
Fs3 = AllFSigs[Cat3 > 0]
pBs1 = infoT1[0]
pBs2 = infoT2[0]
pBs3 = infoT3[0]

plt.errorbar(np.abs(pBs1), Rs1, yerr = Fs1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name1)
plt.errorbar(np.abs(pBs2), Rs2, yerr = Fs2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name2)
plt.errorbar(np.abs(pBs3), Rs3, yerr = Fs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3)
plt.ylabel('$<f_{0}>$ (Hz)')
plt.xlabel('B (T)')
plt.legend(loc = 'best')
plt.show()

def makeZeroOut(BList, RemList):
    Dict = {}
    for i in range(0, len(BList)):
        key = np.abs(BList[i])
        if(key not in Dict.keys()):
            Dict[key] = [RemList[i], 1, RemList[i]]
        else:
            temp = Dict[key]
            temp[0] += RemList[i]
            temp[1] += 1
            temp[2] = temp[0]/temp[1]
            Dict[key] = temp
    zList = np.zeros(len(BList))
    for i in range(0, len(BList)):
        zList[i] = RemList[i] - Dict[np.abs(BList[i])][2]
    return zList

z1 = makeZeroOut(pBs1, Rs1)
z2 = makeZeroOut(pBs2, Rs2)
z3 = makeZeroOut(pBs3, Rs3)
plt.errorbar(np.abs(pBs1), z1, yerr = Fs1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name1)
plt.errorbar(np.abs(pBs2), z2, yerr = Fs2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name2)
#plt.errorbar(np.abs(pBs3), z3, yerr = Fs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3)
plt.show()

plt.errorbar(np.abs(pBs3), z3, yerr = Fs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3)
plt.show()

RampUp1 = [False]*len(pBs1)
for i in range(1, len(pBs1)):
    if(np.abs(pBs1[i]) > np.abs(pBs1[i-1])):
        RampUp1[i] = True
RampDown1 = [not elem for elem in RampUp1]
for i in range(0, len(pBs1)):
    if(np.abs(pBs1[i]) < .5):
        RampUp1[i] = False
        RampDown1[i] = False
        
RampUp2 = [False]*len(pBs2)
for i in range(1, len(pBs2)):
    if(np.abs(pBs2[i]) > np.abs(pBs2[i-1])):
        RampUp2[i] = True
RampDown2 = [not elem for elem in RampUp2]
for i in range(0, len(pBs2)):
    if(np.abs(pBs2[i]) < .5):
        RampUp2[i] = False
        RampDown2[i] = False

RampUp3 = [False]*len(pBs3)
for i in range(1, len(pBs3)):
    if(np.abs(pBs3[i]) >= np.abs(pBs3[i-1])):
        RampUp3[i] = True
RampDown3 = [not elem for elem in RampUp3]
for i in range(0, len(pBs3)):
    if(np.abs(pBs3[i]) < .5):
        RampUp3[i] = False
        RampDown3[i] = False

def PMFMFit(x, a, b):
    q = ((b - x) + np.abs(b - x))/2
    return a*q**2

cutoff = 1.5
maskFM3 = (np.abs(pBs3[RampUp3]) > cutoff)
pltXs = np.linspace(cutoff, np.max(np.abs(pBs3)), 100)
pFM, cFM = curve_fit(PMFMFit, np.abs(pBs3[RampUp3])[maskFM3], (z3[RampUp3])[maskFM3], p0=[.001, 3.75])
pltYs3 = PMFMFit(pltXs, pFM[0], pFM[1])

BsOld = np.hstack((np.abs(pBs1)[RampUp1], np.abs(pBs2)[RampUp2]))
SignalOld = np.hstack((z1[RampUp1], z2[RampUp2]))
SigmOld = np.hstack((Fs1[RampUp1], Fs2[RampUp2]))

maskOld = (BsOld > cutoff)

maskFM1 = (np.abs(pBs1[RampUp1]) > cutoff)
maskFM2 = (np.abs(pBs2[RampUp2]) > cutoff)

pFMold, cFMold = curve_fit(PMFMFit, BsOld[maskOld], SignalOld[maskOld], p0=[.001, 3.75])
pltYsold = PMFMFit(pltXs, pFMold[0], pFMold[1])

print(pFM)
print(np.sqrt(np.diag(cFM)))
print(pFMold)
print(np.sqrt(np.diag(cFMold)))

plt.errorbar(np.abs(pBs1)[RampUp1], z1[RampUp1], yerr = Fs1[RampUp1], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1 + ' Ramp Up', color = "red")
plt.errorbar(np.abs(pBs1)[RampDown1], z1[RampDown1], yerr = Fs1[RampDown1], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1 + ' Ramp Down', color = "blue")
plt.errorbar(np.abs(pBs2)[RampUp2], z2[RampUp2], yerr = Fs2[RampUp2], marker = 'x', linewidth = 0, elinewidth=1, capsize=2, label = name2 + ' Ramp Up', color = "red")
plt.errorbar(np.abs(pBs2)[RampDown2], z2[RampDown2], yerr = Fs2[RampDown2], marker = 'x', linewidth = 0, elinewidth=1, capsize=2, label = name2 + ' Ramp Down', color = "blue")
plt.plot(pltXs, -pltYsold)
plt.plot(pltXs, pltYsold)
plt.ylabel('$<f_{0}>$ (Hz)')
plt.xlabel('B (T)')
plt.legend(loc = 'best')
plt.show()

plt.errorbar(np.abs(pBs3)[RampUp3], z3[RampUp3], yerr = Fs3[RampUp3], marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3 + ' Ramp Up', color = "red")
plt.errorbar(np.abs(pBs3)[RampDown3], z3[RampDown3], yerr = Fs3[RampDown3], marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3 + ' Ramp Down', color = "blue")
plt.plot(pltXs, -pltYs3)
plt.plot(pltXs, pltYs3)
plt.ylabel('$<f_{0}>$ (Hz)')
plt.xlabel('B (T)')
plt.legend(loc = 'best')
plt.show()

plt.errorbar(np.abs(pBs1[RampUp1])[maskFM1], (z1[RampUp1])[maskFM1], yerr = (Fs1[RampUp1])[maskFM1], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1)
plt.errorbar(np.abs(pBs2[RampUp2])[maskFM2], (z2[RampUp2])[maskFM2], yerr = (Fs2[RampUp2])[maskFM2], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name2)
plt.plot(pltXs, pltYsold)
plt.ylabel('$<f_{0}>$ (Hz)')
plt.xlabel('B (T)')
plt.legend(loc = 'best')
plt.show()

plt.errorbar(np.abs(pBs3[RampUp3])[maskFM3], (z3[RampUp3])[maskFM3], yerr = (Fs3[RampUp3])[maskFM3], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name3)
plt.plot(pltXs, pltYs3)
plt.ylabel('$<f_{0}>$ (Hz)')
plt.xlabel('B (T)')
plt.legend(loc = 'best')
plt.show()