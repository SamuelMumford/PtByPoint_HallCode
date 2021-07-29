# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:05:52 2021
@author: sammy
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt

sq = '\u25A1'
f129 = '/home/sam/Documents/12_9_20_coolEdit.txt'
f129L = '/home/sam/Documents/12_9_20_coolEditL.txt'
f1118 = '/home/sam/Documents/Cooldown_all_11_18_20_cool.txt'
f1118L = '/home/sam/Documents/Cooldown_all_11_18_20_coolL.txt'
f114, f114L, name3, endcut3 = '/home/sam/Documents/Cooldown_all_1_14_20_cool.txt', '/home/sam/Documents/Cooldown_all_1_14_20_coolL.txt', '17 k$\Omega$/' + sq, -340
f114b, f114bL, name3b, endcut3b = '/home/sam/Documents/2_10_21cool2.txt', '/home/sam/Documents/2_10_21cool2L.txt', '8.5 k$\Omega$/' + sq, -2660
nameb3 = name3b
name1 = '32 k$\Omega$/' + sq + 'Run 2'#'12/9/20'
name2 = '32 k$\Omega$/' + sq + 'Run 1'#'11/18/20'

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
outliers1 = [0]

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
outliers2 = [0]

Fdat = np.loadtxt(f114, delimiter="\t")
F2 = np.loadtxt(f114L, delimiter="\t")
startcut1 = 0
freqs3 = Fdat[startcut1:endcut3, 2]
runs3 = Fdat[startcut1:endcut3, 3]
temps3 = Fdat[startcut1:endcut3, 4]
currs3 = Fdat[startcut1:endcut3, 5]
Q3 = F2[startcut1:endcut3, 3]
dc3 = Fdat[startcut1:endcut3, 0]
Q3 = np.abs(Q3)
Bs3 = currs3*.875/5
outliers3 = [0]

Fdat = np.loadtxt(f114b, delimiter="\t")
F2 = np.loadtxt(f114bL, delimiter="\t")
startcut1 = 0
freqs4 = Fdat[startcut1:endcut3b, 2]
runs4 = Fdat[startcut1:endcut3b, 3]
temps4 = Fdat[startcut1:endcut3b, 4]
currs4 = Fdat[startcut1:endcut3b, 5]
Q4 = F2[startcut1:endcut3b, 3]
dc4 = Fdat[startcut1:endcut3b, 0]
Q4 = np.abs(Q4)
Bs4 = currs4*.875/5
outliers4 = [0]

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
            dev = np.std(sub)/4
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

infoT1 = segment(Bs1, temps1, outliers1)
infoT2 = segment(Bs2, temps2, outliers2)
infoT3 = segment(Bs3, temps3, outliers3)
infoT4 = segment(Bs3, temps3, outliers3)

l1 = len(infoT1[0])
l2 = len(infoT2[0])
l3 = len(infoT3[0])
l4 = len(infoT4[0])

Cat1 = np.hstack((np.ones(l1), np.zeros(l2), np.zeros(l3), np.zeros(l4)))
Cat2 = np.hstack((np.zeros(l1), np.ones(l2), np.zeros(l3), np.zeros(l4)))
Cat3 = np.hstack((np.zeros(l1), np.zeros(l2), np.ones(l3), np.zeros(l4)))
Cat4 = np.hstack((np.zeros(l1), np.zeros(l2), np.zeros(l3), np.ones(l4)))
AllBs = np.hstack((infoT1[0], infoT2[0], infoT3[0], infoT4[0]))
AllTs = np.hstack((infoT1[1], infoT2[1], infoT3[1], infoT4[1]))
AllSigs = np.hstack((infoT1[2], infoT2[2], infoT3[2], infoT4[2]))

print(len(AllBs))
print(len(AllTs))
print(len(AllSigs))
print(len(Cat4))
print(len(Cat3))
print(len(Cat2))
print(len(Cat1))
data = np.vstack((Cat1, Cat2, Cat3, Cat4, AllBs)).T

def TFit(data, off, poly):
    part1 = data[:, 0]
    part2 = data[:, 1]
    part3 = data[:, 2]
    part4 = data[:, 3]
    Bs = data[:, 4]
    
    fit = np.zeros(len(part1))
    fit += off[0]*part1 + off[1]*part2 + off[2]*part3 + off[3]*part4
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
    off = list(args[:4])
    poly = list(args[4:])
    return TFit(x, off, poly)

NTot = 8
p0 = np.zeros(NTot)
p0[0] = 5.75
p0[1] = 5.75
p0[2] = 5.75
p0[3] = 5.75
p0[4] = -.35
p0[5] = .1
p0[6] = -.012
popt, pcov = curve_fit(lambda data, *p0: wrapT(data, *p0), data, AllTs, sigma = AllSigs, p0=p0)
fitty = wrapT(data, *popt)

#print(popt)
#print(np.mean(np.abs(AllTs - fitty)/AllSigs))

pCopy = np.copy(popt)
pCopy[0] = 0
pCopy[1] = 0
pCopy[2] = 0
pCopy[3] = 0
fittySub = wrapT(data, *pCopy)
TrueTs = AllTs - fittySub

Ts1 = TrueTs[Cat1 > 0]
Ts2 = TrueTs[Cat2 > 0]
Ts3 = TrueTs[Cat3 > 0]
Ts4 = TrueTs[Cat4 > 0]
sTs1 = AllSigs[Cat1 > 0]
sTs2 = AllSigs[Cat2 > 0]
sTs3 = AllSigs[Cat3 > 0]
sTs4 = AllSigs[Cat4 > 0]
pBs1 = infoT1[0]
pBs2 = infoT2[0]
pBs3 = infoT3[0]
pBs4 = infoT4[0]

#plt.errorbar(np.abs(pBs1), Ts1, yerr = sTs1, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
#plt.errorbar(np.abs(pBs2), Ts2, yerr = sTs2, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
#plt.errorbar(np.abs(pBs3), Ts3, yerr = sTs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2)
#plt.show()

infoF1 = segment(Bs1, freqs1, outliers1)
infoF2 = segment(Bs2, freqs2, outliers2)
infoF3 = segment(Bs3, freqs3, outliers3)
infoF4 = segment(Bs4, freqs4, outliers4)

AllFs = np.hstack((infoF1[1], infoF2[1], infoF3[1], infoF4[1]))
AllFSigs = np.hstack((infoF1[2], infoF2[2], infoF3[2], infoF3[2]))

infoQ1 = segment(Bs1, Q1, outliers1)
infoQ2 = segment(Bs2, Q2, outliers2)
infoQ3 = segment(Bs3, Q3, outliers3)
infoQ4 = segment(Bs4, Q4, outliers4)
AllQs = np.hstack((infoQ1[1], infoQ2[1], infoQ3[1], infoQ4[1]))

dataF0 = np.vstack((Cat1, Cat2, Cat3, Cat4, AllBs, TrueTs, AllQs)).T
global N
global NT
global NQ
N = 6
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
    off = list(args[:4])
    polyB = list(args[4:4+N])
    polyT = list(args[4+N:4+N+NT])
    polyQ = list(args[4+N+NT:])
    return F0Fit(x, off, polyB, polyT, polyQ)

pf = np.zeros(NTot)
pf[0] = np.mean(infoF1[1])
pf[1] = np.mean(infoF2[1])
pf[2] = np.mean(infoF3[1])
pf[3] = np.mean(infoF4[1])
print(pf[0])
pf[4] = .3
pf[5] = 1

pof, pcf = curve_fit(lambda dataF0, *pf: wrapF(dataF0, *pf), dataF0, AllFs, p0=pf)
#print(pof)
#print((pof/np.sqrt(np.diag(pcf)))[3:])
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

plt.errorbar(infoF1[0], infoF1[1] - pof[0]+.1, yerr = infoF1[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'red', label = name1)
plt.errorbar(infoF2[0], infoF2[1] - pof[1]+.1, yerr = infoF2[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = name2)
plt.errorbar(infoF3[0], infoF3[1] - pof[2]+.1, yerr = infoF3[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'green', label = name3)
plt.errorbar(infoF4[0], infoF4[1] - pof[3]+.1, yerr = infoF4[2], marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'purple', label = nameb3)
plt.ylabel('$\Delta f_{0} (Hz)$')
plt.xlabel('B (T)')
plt.axvline(x=0, alpha = .5, color = 'k')
plt.axvspan(-1.3, 1.3, alpha=0.25, color='yellow')
plt.legend(loc = 'best', prop={'size':10})
plt.savefig('basef0.pdf', bbox_inches="tight")
plt.show()

Rs1 = (AllFs - EFs)[Cat1 > 0]
Rs2 = (AllFs - EFs)[Cat2 > 0]
Rs3 = (AllFs - EFs)[Cat3 > 0]
Rs4 = (AllFs - EFs)[Cat4 > 0]
Fs1 = AllFSigs[Cat1 > 0]
Fs2 = AllFSigs[Cat2 > 0]
Fs3 = AllFSigs[Cat3 > 0]
Fs4 = AllFSigs[Cat4 > 0]
pBs1 = infoT1[0]
pBs2 = infoT2[0]
pBs3 = infoT3[0]
pBs4 = infoT4[0]

#plt.errorbar(np.abs(pBs1), Rs1, yerr = Fs1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name1)
#plt.errorbar(np.abs(pBs2), Rs2, yerr = Fs2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name2)
#plt.errorbar(np.abs(pBs3), Rs3, yerr = Fs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3)
#plt.ylabel('$<f_{0}>$ (Hz)')
#plt.xlabel('B (T)')
#plt.legend(loc = 'best')
#plt.show()

RampUp1 = [False]*len(pBs1)
RampUp1[0] = True
for i in range(1, len(pBs1)):
    if(np.abs(pBs1[i]) > np.abs(pBs1[i-1])):
        RampUp1[i] = True
RampDown1 = [not elem for elem in RampUp1]
for i in range(0, len(pBs1)):
    if(np.abs(pBs1[i]) < .5):
        RampUp1[i] = False
        RampDown1[i] = False
        
RampUp2 = [False]*len(pBs2)
RampUp2[0] = True
for i in range(1, len(pBs2)):
    if(np.abs(pBs2[i]) > np.abs(pBs2[i-1])):
        RampUp2[i] = True
RampDown2 = [not elem for elem in RampUp2]
for i in range(0, len(pBs2)):
    if(np.abs(pBs2[i]) < .5):
        RampUp2[i] = False
        RampDown2[i] = False

RampUp3 = [False]*len(pBs3)
RampUp3[0] = True
for i in range(1, len(pBs3)):
    if(np.abs(pBs3[i]) > np.abs(pBs3[i-1])):
        RampUp3[i] = True
RampDown3 = [not elem for elem in RampUp3]
for i in range(0, len(pBs3)):
    if(np.abs(pBs3[i]) < .5):
        RampUp3[i] = False
        RampDown3[i] = False
        
RampUp4 = [False]*len(pBs4)
RampUp4[0] = True
for i in range(1, len(pBs4)):
    if(np.abs(pBs4[i]) > np.abs(pBs4[i-1])):
        RampUp4[i] = True
RampDown4 = [not elem for elem in RampUp4]
for i in range(0, len(pBs4)):
    if(np.abs(pBs4[i]) < .5):
        RampUp4[i] = False
        RampDown4[i] = False

def makeZeroOut(BList, RemList, sigma, cat, catn):
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
    zList = np.zeros(0)
    bl = np.zeros(0)
    sl = np.zeros(0)
    cl = []
    cnl = []
    for i in range(0, len(BList)):
        if(Dict[np.abs(BList[i])][1] > .1):
            zList = np.hstack((zList, RemList[i] - Dict[np.abs(BList[i])][2]))
            bl = np.hstack((bl, np.abs(BList[i])))
            sl = np.hstack((sl, sigma[i]))
            cl.append(cat[i])
            cnl.append(catn[i])
    return zList, bl, sl, cl, cnl

z1, b1, e1, c1, cn1 = makeZeroOut(pBs1, Rs1, Fs1, RampUp1, RampDown1)
z2, b2, e2, c2, cn2 = makeZeroOut(pBs2, Rs2, Fs2, RampUp2, RampDown2)
z3, b3, e3, c3, cn3 = makeZeroOut(pBs3, Rs3, Fs3, RampUp3, RampDown3)
z4, b4, e4, c4, cn4 = makeZeroOut(pBs4, Rs4, Fs4, RampUp4, RampDown4)

#plt.errorbar(b1, z1, yerr = e1, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name1)
#plt.errorbar(b2, z2, yerr = e2, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name2)
##plt.errorbar(np.abs(pBs3), z3, yerr = Fs3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3)
#plt.show()
#
#plt.errorbar(b3, z3, yerr = e3, marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3)
#plt.show()

def PMFMFit(x, a, b):
    q = ((b - x) + np.abs(b - x))/2
    return a*q*x

cutoff = 1.5
maskFM3 = (b3[c3] > cutoff)
pltXs = np.linspace(cutoff, np.max(np.abs(b3)), 100)
pFM, cFM = curve_fit(PMFMFit, (b3[c3])[maskFM3], (z3[c3])[maskFM3], p0=[.001, 6])
pltYs3 = PMFMFit(pltXs, pFM[0], pFM[1])

maskFM4 = (b4[c4] > cutoff)
pltXs4 = np.linspace(cutoff, np.max(np.abs(b4)), 100)
pFM4, cFM4 = curve_fit(PMFMFit, (b4[c4])[maskFM4], (z4[c4])[maskFM4], p0=[.001, 6])
pltYs4 = PMFMFit(pltXs, pFM4[0], pFM4[1])

BsOld = np.hstack((b1[c1], b2[c2]))
SignalOld = np.hstack((z1[c1], z2[c2]))
SigmOld = np.hstack((e1[c1], e2[c2]))

maskOld = (BsOld > cutoff)

maskFM1 = (b1[c1] > cutoff)
maskFM2 = (b2[c2] > cutoff)
maskFMn1 = (b1[cn1] > cutoff)
maskFMn2 = (b2[cn2] > cutoff)

pFMold, cFMold = curve_fit(PMFMFit, BsOld[maskOld], SignalOld[maskOld], p0=[.001, 6])
pltYsold = PMFMFit(pltXs, pFMold[0], pFMold[1])

print('Dataset 1')
print(pFM)
print(np.sqrt(np.diag(cFM)))
print('Dataset 2')
print(pFMold)
print(np.sqrt(np.diag(cFMold)))

xZ = np.linspace(0, np.max(b1))
tol = .2

plt.errorbar(b2[c2][maskFM2], 2*z2[c2][maskFM2], yerr = 2*e2[c2][maskFM2], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name2, color = "red")
#plt.errorbar(b2[cn2][maskFMn2], -z2[cn2][maskFMn2], yerr = e2[cn2][maskFMn2], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name2 + ' Down', color = "red")
plt.errorbar(b1[c1][maskFM1], 2*z1[c1][maskFM1], yerr = 2*e1[c1][maskFM1], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1, color = "blue")
#plt.errorbar(b1[cn1][maskFMn1], -z1[cn1][maskFMn1], yerr = e1[cn1][maskFMn1], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1 + ' Down', color = "blue")
#plt.plot(pltXs, -pltYsold)
plt.plot(pltXs, 2*pltYsold)
plt.plot(pltXs, 0*pltXs, color = 'black')
plt.ylabel(r'Hysteretic $\Delta f_{0}$')
plt.xlabel('B (T)')
plt.legend(loc = 'best', prop={'size': 12})
plt.savefig('Mag4k.pdf', bbox_inches="tight")
plt.show()

plt.errorbar(b3[c3][maskFM3], 2*z3[c3][maskFM3], yerr = 2*e3[c3][maskFM3], marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3, color = "green")
#plt.errorbar(b3[cn3], z3[cn3], yerr = e3[cn3], marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3 + ' Down', color = "blue")
#plt.plot(pltXs, -pltYs3)
plt.plot(pltXs, 2*pltYs3)
plt.plot(pltXs, 0*pltXs, color = 'black')
plt.ylabel(r'Hysteretic $\Delta f_{0}$')
plt.xlabel('B (T)')
plt.legend(loc = 'best', prop={'size': 12})
plt.savefig('Mag2k.pdf', bbox_inches="tight")
plt.show()

plt.errorbar(b4[c4][maskFM4], 2*z4[c4][maskFM4], yerr = 2*e4[c4][maskFM4], marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3b, color = "purple")
#plt.errorbar(b3[cn3], z3[cn3], yerr = e3[cn3], marker = '.', linewidth = 0, elinewidth=1, capsize=2, label = name3 + ' Down', color = "blue")
#plt.plot(pltXs, -pltYs3)
plt.plot(pltXs4, 2*pltYs4)
plt.plot(pltXs4, 0*pltXs4, color = 'black')
plt.ylabel(r'Hysteretic $\Delta f_{0}$')
plt.xlabel('B (T)')
plt.legend(loc = 'best', prop={'size': 12})
plt.savefig('Mag800.pdf', bbox_inches="tight")
plt.show()

cf = .13*1000

cutty = 1.5
mask = b1[c1] > cutty
maski = b2[c2]> cutty
plt.errorbar(b1[c1][mask], cf*(z1[c1]/b1[c1])[mask], yerr = cf*(e1[c1]/b1[c1])[mask], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1+ ' Ramp Up', color = "red")
plt.errorbar(b1[cn1], cf*(z1[cn1]/b1[cn1]), yerr = cf*(e1[cn1]/b1[cn1]), marker = 'x', linewidth = 0, elinewidth=1, capsize=2, label = name1+ ' Ramp Down', color = "red")
plt.errorbar(b2[c2][maski], cf*(z2[c2]/b2[c2])[maski], yerr = cf*(e2[c2]/b2[c2])[maski], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name2+ ' Ramp Up', color = "blue")
plt.errorbar(b2[cn2], cf*(z2[cn2]/b2[cn2]), yerr = cf*(e2[cn2]/b2[cn2]), marker = 'x', linewidth = 0, elinewidth=1, capsize=2, label = name2+ ' Ramp Down', color = "blue")
plt.plot(pltXs, cf*pltYsold/pltXs, color = 'black')
plt.plot(pltXs, -cf*pltYsold/pltXs, color = 'black')
plt.plot(xZ, 0*xZ, color = 'black')
plt.xlim((cutty, np.amax(b1)+tol))
plt.ylim((-1.2, 1))
plt.ylabel(r'$\Delta m$ ($\mu_{B}$/1000 F.U.)')
plt.xlabel('|B| (T)')
plt.legend(loc = 'best', prop={'size':14})
plt.savefig('Mag32k.pdf', bbox_inches="tight")
plt.show()

masky = b3[c3] > cutty
plt.errorbar(b3[c3][masky], cf*(z3[c3]/b3[c3])[masky], yerr = cf*(e3[c3]/b3[c3])[masky], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name3 + ' Ramp Up', color = "green")
plt.errorbar(b3[cn3], cf*(z3[cn3]/b3[cn3]), yerr = cf*(e3[cn3]/b3[cn3]), marker = 'x', linewidth = 0, elinewidth=1, capsize=2, label = name3 + ' Ramp Down', color = "green")
plt.plot(pltXs, cf*pltYs3/pltXs, color = 'black')
plt.plot(pltXs, -cf*pltYs3/pltXs, color = 'black')
plt.plot(xZ, 0*xZ, color = 'black')
plt.ylabel(r'$\Delta m$ ($\mu_{B}$/1000 F.U.)')
plt.xlim((cutty, np.amax(b3)+tol))
plt.xlabel('|B| (T)')
plt.legend(loc = 'best')
plt.savefig('Mag17k.pdf', bbox_inches="tight")
plt.show()

maskyy = b4[c4] > cutty
plt.errorbar(b4[c4][maskyy], cf*(z4[c4]/b4[c4])[maskyy], yerr = cf*(e4[c4]/b4[c4])[maskyy], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name3b + ' Ramp Up', color = "purple")
plt.errorbar(b4[cn4], cf*(z4[cn4]/b4[cn4]), yerr = cf*(e4[cn4]/b4[cn4]), marker = 'x', linewidth = 0, elinewidth=1, capsize=2, label = name3b + ' Ramp Down', color = "purple")
plt.plot(pltXs4, cf*pltYs4/pltXs4, color = 'black')
plt.plot(pltXs4,- cf*pltYs4/pltXs4, color = 'black')
plt.plot(xZ, 0*xZ, color = 'black')
plt.ylabel(r'$\Delta m$ ($\mu_{B}$/1000 F.U.)')
plt.xlim((cutty, np.amax(b4)+tol))
plt.xlabel('|B| (T)')
plt.legend(loc = 'best')
plt.savefig('Mag800.pdf', bbox_inches="tight")
plt.show()


plt.errorbar(b1[c1][mask], cf*(z1[c1]/b1[c1])[mask], yerr = cf*(e1[c1]/b1[c1])[mask], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name1, color = "red")
plt.errorbar(b2[c2][maski], cf*(z2[c2]/b2[c2])[maski], yerr = cf*(e2[c2]/b2[c2])[maski], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name2, color = "blue")
plt.errorbar(b3[c3][masky], cf*(z3[c3]/b3[c3])[masky], yerr = cf*(e3[c3]/b3[c3])[masky], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name3, color = "green")
plt.errorbar(b4[c4][maskyy], cf*(z4[c4]/b4[c4])[maskyy], yerr = cf*(e4[c4]/b4[c4])[maskyy], marker = 'o', linewidth = 0, elinewidth=1, capsize=2, label = name3b, color = "orange")
plt.plot(pltXs, cf*pltYsold/pltXs, color = 'purple')
plt.plot(pltXs, cf*pltYs3/pltXs, color = 'green')
plt.plot(pltXs4, cf*pltYs4/pltXs4, color = 'orange')
plt.plot(xZ, 0*xZ, color = 'black')
plt.xlim((cutty, np.amax(b1)+tol))
plt.ylabel(r'$\Delta m$ ($\mu_{B}$/1000 F.U.)')
plt.xlabel('B (T)')
plt.legend(loc = 'best')
plt.show()