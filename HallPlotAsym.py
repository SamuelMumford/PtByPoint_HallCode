
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:09:11 2018

@author: KGB
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def secSplit(Bs, amps, ampsH):
    aParts = []
    hParts = []
    asParts = []
    hsParts = []
    BsL = []
    start = 0
    val = Bs[0]
    for i in range(1, len(Bs)):
        if(Bs[i] != val):
            aParts = aParts + [np.mean(amps[start:i])]
            asParts = asParts + [np.std(amps[start:i])/3]
            hParts = hParts + [np.mean(ampsH[start:i])]
            hsParts = hsParts + [np.std(ampsH[start:i])/3]
            BsL = BsL + [val]
            start = i
            val = Bs[i]
    return aParts, hParts, asParts, hsParts, BsL

def consolidate(aParts, hParts, asParts, hsParts, BsL):
    a = []
    h = []
    av = []
    hv = []
    b = []
    c = []
    for i in range(len(BsL)):
        if(BsL[i] in b):
            j = b.index(BsL[i])
            c[j] += 1
            hv[j] += hsParts[i]**2
            av[j] += asParts[i]**2
            a[j] += aParts[i]
            h[j] += hParts[i]
        else:
            b = b + [BsL[i]]
            c = c + [1]
            hv = hv + [hsParts[i]**2]
            av = av + [asParts[i]**2]
            a = a + [aParts[i]]
            h = h + [hParts[i]]
    a = [v/o for v, o in zip(a, c)]
    h = [v/o for v, o in zip(h, c)]
    asi = [np.sqrt(v/o) for v, o in zip(av, c)]
    hsi = [np.sqrt(v/o) for v, o in zip(hv, c)]
    return a, h, asi, hsi, b

def Opa(a, h, asi, hsi, b):
    oh = []
    ohv = []
    ob = []
    for i in range(len(b)):
        if(-b[i] in b):
            if(np.abs(b[i]) in ob):
                j = ob.index(np.abs(b[i]))
                oh[j] += h[i]*np.sign(b[i])/2
                ohv[j] += hsi[i]**2/2
            else:
                ob = ob + [np.abs(b[i])]
                oh = oh + [h[i]*np.sign(b[i])/2]
                ohv = ohv + [hsi[i]**2/2]
    ohsi = [np.sqrt(l) for l in ohv]
    return ob, oh, ohsi
        
            
            

#fname = 'D:/Dropbox (KGB Group)/Gravity/LabView/3_17_21_t1.txt'
#fname = 'D:/Dropbox (KGB Group)/Gravity/LabView/3_17_21_t3.txt'
fname = 'D:/Dropbox (KGB Group)/Gravity/LabView/3_18_21_pump1.txt'
startcut = 100#20

Fdat = np.loadtxt(fname, delimiter="\t")
startcut *= 10
endcut =-1
Bs = Fdat[startcut:endcut, 0]*.875/(5)
amps = Fdat[startcut:endcut, 1]
Phs = Fdat[startcut:endcut, 2]*np.pi/180
amps = amps/np.cos(Phs)
ampsH = Fdat[startcut:endcut, 3]
PhsH = Fdat[startcut:endcut, 4]*np.pi/180
#ampsH = Fdat[startcut:endcut, 3]/np.cos(PhsH)
ts = Fdat[startcut:endcut, -1]

colors = cm.jet(np.linspace(0, 1, len(Bs)))
    
aParts, hParts, asParts, hsParts, BsL = secSplit(Bs, amps, ampsH)
a, h, asi, hsi, b = consolidate(aParts, hParts, asParts, hsParts, BsL)
ob, oh, ohsi = Opa(a, h, asi, hsi, b)
plt.errorbar(ob, oh, ohsi, marker = '.', linewidth = 0, elinewidth=1)
plt.show()
def FF(x, a):
    y = a*x
    return y

#print(BsL)
#FT = FF(BsL, 0)
#print(FT)
#
#plt.errorbar(BsL, hParts-FT, hsParts, marker = '.', linewidth = 0, elinewidth=1)
#plt.xlabel('B Field (T)')
#plt.ylabel('Amp (V)')
#plt.ylim(np.amin(hParts-FT), np.amax(hParts-FT))
#plt.show()