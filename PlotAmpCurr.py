
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

def FF(x, a, b, c, d, e, f):
    y = a + b*(np.exp(x/f))*np.sin(-2*c*d*np.pi*(np.exp(-x/d) - 1) + e)
    return y

fname, startcut = '/home/sam/Documents/3_18_cooldown/3_17_21_t1.txt', 75
#fname = '/home/sam/Documents/3_18_cooldown/3_17_21_t3.txt'
fname4, startcut4 = '/home/sam/Documents/3_18_cooldown/3_18_21_pump1.txt', 140
#startcut = 75#20*10

def makePlot(fname, startcut):
    Fdat = np.loadtxt(fname, delimiter="\t")
    startcut *= 10
    endcut =-1
    currs = Fdat[startcut:endcut, 0]*.875/(5)
    currs = np.abs(currs)
    amps = Fdat[startcut:endcut, 1]
    Phs = Fdat[startcut:endcut, 2]*np.pi/180
    amps = amps/np.cos(Phs)
    ts = Fdat[startcut:endcut, -1]
    
    colors = cm.jet(np.linspace(0, 1, len(currs)))    
    def Tfit(x, a, b, c, d, e):
        return a*np.tanh(x**2/b) + c + d*x**2 + e*x**4
    Tps = Tfit(currs, -.2, .1, 7.55, -.03, .00025)
    popt,pcov=curve_fit(Tfit, currs, ts, p0=(-.2, .1, 7.55, -.03, .00025))
    Tps = Tfit(currs, popt[0], popt[1], popt[2], popt[3], popt[4])
    
    delT = ts - Tps
    delR = (amps - np.amax(amps))/np.amax(amps)
    
    def Rfit(x, a, b, c, d):
        return a*x**2 + b*x**3 + c*x**4 + d*x**5
    Rps = Rfit(currs, -.007, .0001, 0, 0)
    pr,pcr=curve_fit(Rfit, currs, delR, p0=( -.005, .00005, 0, 0))
    Rps = Rfit(currs, pr[0], pr[1], pr[2], pr[3])
    
    plt.scatter(delT, delR - Rps)
    plt.show()
    
    def Tprop(x, a):
        return x*a
    Tsub = Tprop(delT, 0)
    pt,ptr=curve_fit(Tprop, delT, delR - Rps, p0=(0))
    Tsub = Tprop(delT, pt[0])
    
    return currs, delR, Tsub, (delT + popt[2]), colors

currs, delR, Tsub, TrueT, colors = makePlot(fname, startcut)
currs4, delR4, Tsub4, TrueT4, colors4 = makePlot(fname4, startcut4)

plt.scatter(currs, TrueT, c = colors, s=10, linewidth = .5,  zorder=1, edgecolors = 'k')
plt.xlabel('B Field (T)')
plt.ylabel('T (K)')
plt.show()

plt.scatter(currs4, TrueT4, c = colors4, s=10, linewidth = .5,  zorder=1, edgecolors = 'k')
plt.xlabel('B Field (T)')
plt.ylabel('T (K)')
plt.show()

plt.scatter(currs, delR - Tsub, s=10, linewidth = .5,  zorder=1, edgecolors = 'k', label = '7.5K')
plt.scatter(currs4, delR4 - Tsub4, s=10, linewidth = .5,  zorder=1, edgecolors = 'k', label = '4.8K')
plt.xlabel('B Field (T)')
plt.ylabel('delta R/R_{0}')
plt.legend(loc = 'best')
plt.show()