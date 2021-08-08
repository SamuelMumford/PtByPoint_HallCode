
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
                oh[j] += h[i]*np.sign(b[i])/2
                ohv[j] += hsi[i]**2/2
            else:
                ob = ob + [np.abs(b[i])]
                oh = oh + [h[i]*np.sign(b[i])/2]
                ohv = ohv + [hsi[i]**2/2]
    ohsi = [np.sqrt(l) for l in ohv]
    return np.array(ob), np.array(oh), np.array(ohsi)      

def postFilter(ob, oh, ohsi, a, b):
    tol = 2
    filt = (ohsi < tol)
    return ob[filt], oh[filt], ohsi[filt], a[filt], b[filt]

fname = '/home/sam/Documents/HallCool/3_18_21_pump1.txt'
startcut = 100
fnameW = '/home/sam/Documents/HallCool/3_17_21_t1.txt'
startcutW = 20

fnameA, startcutA, label6 = '/home/sam/Documents/4_10_cooldown/4_10_21_t1.txt', 40, 'Anneal 2, 6.5 K'
fnameA4, startcutA4, label4 = '/home/sam/Documents/4_10_cooldown/4_11_21_pumpB.txt', 200, 'Anneal 2, 3.5 K'

def makeHallData(fname, startcut):
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
    
    a, h, asi, hsi, b = consolidate(aParts, hParts-polluteBin, asParts, hsParts, BsL)
    ob, oh, ohsi = Opa(a, h, asi, hsi, b)
    pl,plc=curve_fit(Fl, ob, oh, p0=(0), sigma = ohsi)
    blin = ob#np.linspace(0, np.amax(ob))
    Hfit = pl[0]*blin
    cf = (5E6)
    oh *= cf
    ohsi *= cf
    Hfit *= cf
    cd = 1/(pl[0]*5E-8*cf*1.6E-19)
    uncCd = np.sqrt(np.diag(plc)[0])/(pl[0]**2*(5E-8*cf*1.6E-19))
    print(np.sqrt(np.diag(plc)[0])/pl[0])
    return ob, oh, ohsi, blin, Hfit, cd*1E-6, uncCd*1E-6

obC1, ohC1, ohsiC1, blinC1, HfitC1, cdC1, uncCdC1 = makeHallData(fname, startcut)
obW1, ohW1, ohsiW1, blinW1, HfitW1, cdW1, uncCdW1 = makeHallData(fnameW, startcutW)

obC2, ohC2, ohsiC2, blinC2, HfitC2, cdC2, uncCdC2 = makeHallData(fnameA4, startcutA4)
obW2, ohW2, ohsiW2, blinW2, HfitW2, cdW2, uncCdW2 = makeHallData(fnameA, startcutA)
obC2, ohC2, ohsiC2, blinC2, HfitC2 = postFilter(obC2, ohC2, ohsiC2, blinC2, HfitC2)

print(str(cdC1) + ' +- ' + str(uncCdC1))
print(str(cdW1) + ' +- ' + str(uncCdW1))
print(str(cdC2) + ' +- ' + str(uncCdC2))
print(str(cdW2) + ' +- ' + str(uncCdW2))

plt.errorbar(obW1, ohW1, ohsiW1, fmt='o', color='r', markersize=5, label = 'Anneal 1, 6.5 K')
plt.plot(blinW1, HfitW1, color = 'k')
plt.errorbar(obC1, ohC1, ohsiC1, fmt='x', color='r', markersize=5, label = 'Anneal 1, 3.5 K')
plt.plot(blinC1, HfitC1, color = 'k')

plt.errorbar(obW2, ohW2, ohsiW2, fmt='o', color='b', markersize=5, label = 'Anneal 2, 6.5 K')
plt.plot(blinW2, HfitW2, color = 'k')
plt.errorbar(obC2, ohC2, ohsiC2, fmt='x', color='b', markersize=5, label = 'Anneal 2, 3.5 K')
plt.plot(blinC2, HfitC2, color = 'k')

plt.xlabel('|B| (T)')
plt.ylabel(r'$R_{xy}$ ($\Omega$)')
plt.legend(loc = 'best', prop={'size':10})
#plt.savefig('/home/sam/Documents/DriftTests/HallPlot.pdf', bbox_inches="tight")
plt.show()

plt.errorbar(obW1, ohW1-HfitW1, ohsiW1, fmt='o', color='r', markersize=5, label = 'Anneal 1, 6.5 K')
plt.xlabel('|B| (T)')
plt.ylabel(r'$\delta R_{xy}$ ($\Omega$)')
plt.legend(loc = 'best', prop={'size':10})
#plt.savefig('/home/sam/Documents/DriftTests/HallPlot.pdf', bbox_inches="tight")
plt.show()
plt.errorbar(obC1, ohC1-HfitC1, ohsiC1, fmt='x', color='r', markersize=5, label = 'Anneal 1, 3.5 K')
plt.xlabel('|B| (T)')
plt.ylabel(r'$\delta R_{xy}$ ($\Omega$)')
plt.legend(loc = 'best', prop={'size':10})
#plt.savefig('/home/sam/Documents/DriftTests/HallPlot.pdf', bbox_inches="tight")
plt.show()

plt.errorbar(obW2, ohW2-HfitW2, ohsiW2, fmt='o', color='b', markersize=5, label = 'Anneal 2, 6.5 K')
plt.xlabel('|B| (T)')
plt.ylabel(r'$\delta R_{xy}$ ($\Omega$)')
plt.legend(loc = 'best', prop={'size':10})
#plt.savefig('/home/sam/Documents/DriftTests/HallPlot.pdf', bbox_inches="tight")
plt.show()
plt.errorbar(obC2, ohC2-HfitC2, ohsiC2, fmt='x', color='b', markersize=5, label = 'Anneal 2, 3.5 K')

plt.xlabel('|B| (T)')
plt.ylabel(r'$\delta R_{xy}$ ($\Omega$)')
plt.legend(loc = 'best', prop={'size':10})
#plt.savefig('/home/sam/Documents/DriftTests/HallPlot.pdf', bbox_inches="tight")
plt.show()