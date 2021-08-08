#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:18:31 2021

@author: sam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate
from scipy.optimize import curve_fit

fnameA, startcutA, label6 = '/home/sam/Documents/4_10_cooldown/4_10_21_t1.txt', 40, 'Anneal 2, 6.5 K'
fnameA4, startcutA4, label4 = '/home/sam/Documents/4_10_cooldown/4_11_21_pumpB.txt', 200, 'Anneal 2, 3.5 K'

fname, startcut, labelhr = '/home/sam/Documents/3_18_cooldown/3_17_21_t1.txt', 75, 'Anneal 1, 6.5 K'
fname4, startcut4, label4hr = '/home/sam/Documents/3_18_cooldown/3_18_21_pump1.txt', 140, 'Anneal 1, 4 K'

fnameT = '/home/sam/Documents/HallCool/6_10_4K.txt'
fnameT6 = '/home/sam/Documents/HallCool/6_10_6K.txt'

def makePlot(fname, startcut):
    Fdat = np.loadtxt(fname, delimiter="\t")
    startcut *= 10
    endcut =-1
    currs = Fdat[startcut:endcut, 0]*.875/(5)
    amps = Fdat[startcut:endcut, 1]
    Phs = Fdat[startcut:endcut, 2]*np.pi/180
    amps = amps/np.cos(Phs)
    ts = Fdat[startcut:endcut, -1]
    runInds = np.linspace(0, len(amps)-1, len(amps))
    
    lowBm = (currs == 0)
    a0 = np.mean(amps[lowBm])
    amps = (amps - a0)/a0
    
    def Tfit(x, a, b, c, d, e):
        return a*np.tanh(x**2/b) + c + d*x**2 + e*x**4
    mask = ts > 1.0
    maskoff = ts < 1.0
    cf = currs[mask]
    tf = ts[mask]
    ri = runInds[mask]

    Tps = Tfit(currs, -.2, .1, 7.55, -.03, .00025)
    popt,pcov=curve_fit(Tfit, cf, tf, p0=(-.2, .1, 7.55, -.03, .00025))
    Tps = Tfit(cf, popt[0], popt[1], popt[2], popt[3], popt[4])
    
    delT = tf - Tps
    window = 300
    smooT = savgol_filter(delT, 2*window + 1, 2)
    
    tpf = interpolate.interp1d(ri, smooT, fill_value="extrapolate")
    delTPred = tpf(runInds)
    
    X = np.vstack((np.abs(currs), delTPred))
    def Rfit(x, a, b, c, d, e, f, g):
        cu = x[0, :]
        t = x[1, :]
        ff = a*cu + b*cu**2 + c*cu**4 + d*cu**6 + e*cu**8 + f*t + g*t**2
        return ff
    pr,prcov=curve_fit(Rfit, X, amps, p0=(0, 0, 0, 0, 0, 0, 0))
    rf = Rfit(X, pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6])
    
    Tpart = Rfit(X, 0, 0, 0, 0, 0, pr[5], pr[6])
    Rpart = Rfit(X, pr[0], pr[1], pr[2], pr[3], pr[4], 0, 0)
    
    delR = amps - Tpart
    
#    plt.scatter(currs, delR, c= colors, s = 5, marker = '.')
#    plt.ylabel(r'$\delta R/R_{0}$')
#    plt.xlabel('|B| (T)')
#    plt.show()
    
#    plt.scatter(delTPred, amps - Rpart, c = colors, s= 5, marker = '.')
#    plt.ylabel('R Fit Residaul')
#    plt.xlabel(r'$\Delta$ T')
#    plt.show()
#    
#    plt.scatter(runInds, amps - Rpart, c = colors, s= 5, marker = '.')
#    plt.ylabel('R Fit Residaul')
#    plt.xlabel('Data Order')
#    plt.show()
    
    return currs, delR

def getData(fnameT):
    Fdat = np.loadtxt(fnameT)
    currsAng = Fdat[0, :]
    delRAng = Fdat[1, :]
    tol = .001
    maskB = (np.abs(currsAng) < tol)
    offy = 0
    delRAng = delRAng + offy
    r0 = np.mean(delRAng[maskB])
    delRAng = (delRAng - r0)/r0
    
    keep = [True]*len(delRAng)
    tolly = .001
    for i in range(1, len(delRAng)-1):
        if(np.abs(delRAng[i] - delRAng[i-1]) > tolly):
            if(np.abs(delRAng[i] - delRAng[i+1]) > tolly):
                keep[i] = False
    currsAng = currsAng[keep]
    delRAng = delRAng[keep]
    return currsAng, delRAng

def groupData(Bs, amps, nbins):
        n, _ = np.histogram(Bs, bins=nbins)
        sy, _ = np.histogram(Bs, bins=nbins, weights=amps)
        sy2, binni = np.histogram(Bs, bins=nbins, weights=amps*amps)
        mean = sy/n
        std = np.sqrt(sy2/n - mean*mean)
        return (binni[1:] + binni[:-1])/2, mean, std
    
def discData(Bs, Amps):
    AllBs = []
    AllAmps = []
    AllDevs = []
    startVal = Bs[0]
    start = 0
    for i in range(0, len(Bs)):
        if(Bs[i] != startVal):
            AllBs = AllBs + [startVal]
            AllAmps = AllAmps + [np.mean(Amps[start:i])]
            AllDevs = AllDevs + [np.std(Amps[start:i])/np.sqrt(i-start)]
            start = i
            startVal = Bs[i]
    AllBs = AllBs + [startVal]
    AllAmps = AllAmps + [np.mean(Amps[start:])]
    AllDevs = AllDevs + [np.std(Amps[start:])/np.sqrt(len(Bs)-start)]
    return np.array(AllBs), np.array(AllAmps), np.array(AllDevs)


currs, delR = makePlot(fnameA, startcutA)
currs4, delR4 = makePlot(fnameA4, startcutA4)

currs, delR, delRDev = discData(currs, delR)
currs4, delR4, delR4Dev = discData(currs4, delR4)

currsAng, delRAng = getData(fnameT6)
currs4Ang, delR4Ang = getData(fnameT)

currsAng, delRAng, devAng = groupData(currsAng, delRAng, 200)
currs4Ang, delR4Ang, dev4Ang = groupData(currs4Ang, delR4Ang, 200)

currshr, delRhr = makePlot(fname, startcut)
currs4hr, delR4hr = makePlot(fname4, startcut4)

currshr, delRhr, delRhrDev = discData(currshr, delRhr)
currs4hr, delR4hr, delR4hrDev = discData(currs4hr, delR4hr)

plt.scatter(np.abs(currshr), delRhr, c = 'green', s=3, label='Anneal 1')
plt.scatter(np.abs(currs), delR, c = 'red', s=3, label='Anneal 2, Vertical')
plt.scatter(np.abs(currsAng), delRAng, c = 'blue', s=1, label=r'Anneal 2, 20$^{\circ}$ Angle')
plt.xlabel('|B| (T)')
plt.ylabel(r'$\delta R/R_{0}$')
plt.legend(loc='best', title = '6.5 K Magnetoresistance', prop={'size':10}).get_title().set_fontsize('12')
plt.savefig('/home/sam/Documents/DriftTests/6Ks.pdf', bbox_inches="tight")
plt.show()

plt.scatter(np.abs(currs4hr), delR4hr, c = 'green', s=3, label='Anneal 1')
plt.scatter(np.abs(currs4), delR4, c = 'red', s=3, label='Anneal 2, Vertical')
plt.scatter(np.abs(currs4Ang), delR4Ang, c = 'blue', s=1, label=r'Anneal 2, 20$^{\circ}$ Angle')
plt.xlabel('|B| (T)')
plt.ylabel(r'$\delta R/R_{0}$')
plt.legend(loc='best', title = '3.5 K Magnetoresistance', prop={'size':10}).get_title().set_fontsize('12')
plt.savefig('/home/sam/Documents/DriftTests/4Ks.pdf', bbox_inches="tight")
plt.show()