
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
from scipy.interpolate import interp1d

def FF(x, a, b, c):
    Vs = x[:, 0]
    Bs = x[:, 1]
    y = a*Vs + b*Bs + c
    return y

def Fl(x, a):
    return x*a

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
                oh[j] += h[i]*np.sign(b[i])/4
                ohv[j] += hsi[i]**2/2
            else:
                ob = ob + [np.abs(b[i])]
                oh = oh + [h[i]*np.sign(b[i])/2]
                ohv = ohv + [hsi[i]**2/4]
    ohsi = [np.sqrt(l) for l in ohv]
    return ob, oh, ohsi            

def Hall(fname, startcut):
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
    
    data = np.vstack((amps, Bs)).T
    p0 = (.022, 0, 0)
    popt,pcov=curve_fit(FF, data, ampsH, p0=p0)
    test = FF(data, popt[0], popt[1], popt[2])
    
    
    pollute = FF(data, popt[0], 0, popt[2])
    
    aParts, hParts, asParts, hsParts, BsL = secSplit(Bs, amps, ampsH)
    dataBin = np.vstack((aParts, BsL)).T
    polluteBin = FF(dataBin, popt[0], 0, popt[2])
    FT = FF(dataBin, popt[0], popt[1], popt[2])
    BD = np.linspace(min(BsL), max(BsL))
    PD = np.zeros(len(BD))
    DD = np.vstack((PD, BD)).T
    Sign = FF(DD, 0, popt[1], 0)
    
    colors = cm.jet(np.linspace(0, 1, len(BsL)))
    #plt.errorbar(BsL, hParts-polluteBin, hsParts, marker = '.', linewidth = 0, elinewidth=1)
    
    a, h, asi, hsi, b = consolidate(aParts, hParts-polluteBin, asParts, hsParts, BsL)
    ob, oh, ohsi = Opa(a, h, asi, hsi, b)
    pl,plc=curve_fit(Fl, ob, oh, p0=(0), sigma = ohsi)
    blin = np.linspace(0, np.amax(ob))
    Hfit = pl[0]*blin
    
    return ob, oh, ohsi, BsL, (hParts - polluteBin), pl[0], colors

fname, startcut = '/home/sam/Documents/3_18_cooldown/3_17_21_t1.txt', 65
#fname = '/home/sam/Documents/3_18_cooldown/3_17_21_t3.txt'
fname4, startcut4 = '/home/sam/Documents/3_18_cooldown/3_18_21_pump1.txt', 100

ob, oh, ohsi, BsL, hrem, fit, colors = Hall(fname, startcut)
ob4, oh4, ohsi4, BsL4, hrem4, fit4, colors4 = Hall(fname4, startcut4)

blin - np.linspace(min(ob), max(ob))
Hfit = fit*blin
Hfit4 = fit4*blin

plt.errorbar(ob, oh, ohsi, marker = '.', linewidth = 0, elinewidth=1)
plt.plot(blin, Hfit)
plt.errorbar(ob4, oh4, ohsi4, marker = '.', linewidth = 0, elinewidth=1)
plt.plot(blin, Hfit4)
plt.xlabel('B Field (T)')
plt.ylabel('V_{xy}')
plt.show()
    
plt.errorbar(ob, oh-np.array(ob)*fit, ohsi, marker = '.', linewidth = 0, elinewidth=1)
plt.errorbar(ob4, oh4-np.array(ob4)*fit4, ohsi4, marker = '.', linewidth = 0, elinewidth=1)
plt.xlabel('B Field (T)')
plt.ylabel('Resid Amp (V)')
plt.show()

rat = 2.517E20/1.8641E-6
print(fit*rat)
print(fit4*rat)