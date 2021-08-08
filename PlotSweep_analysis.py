
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:09:11 2018

@author: KGB
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import interpolate
from scipy.signal import savgol_filter

base = '/home/sam/Documents/HallCool/'
if(True):
    fname = base + '6_9_LowT_Bsweep_v1.txt'#'6_10_LowT_Bsweep_v4.txt'#'6_9_LowT_Bsweep_v1.txt'
    startcut = 0
    endcut =-5000
    Bstart = 2
    offy = .015#.02#.015
    StartCenter = True
else:
    fname = base + '6_10_LowT_Bsweep_v4.txt'#'6_9_LowT_Bsweep_v1.txt'
    startcut = 6400
    endcut =-1
    Bstart = 2
    offy = .0225#.015
    StartCenter = True

Fdat = np.loadtxt(fname)
runs = Fdat[startcut:endcut, 0]
Bs = Fdat[startcut:endcut, 1]*.875/(5)
amps = Fdat[startcut:endcut, 3]
Phs = Fdat[startcut:endcut, 4]*np.pi/180
amps = amps/np.cos(Phs)
ampsH = Fdat[startcut:endcut, 5]
PhsH = Fdat[startcut:endcut, 6]*np.pi/180
#ampsH = Fdat[startcut:endcut, 3]/np.cos(PhsH)
ts = Fdat[startcut:endcut, 2]
amps = amps/np.mean(amps)

mask = (np.abs(Phs + .018) < .01)
tol = 5E-3
for i in range(1, len(runs)-1):
    if(np.abs(amps[i] - amps[i-1]) > tol):
        if(np.abs(amps[i] - amps[i+1]) > tol):
            mask[i] = False
runs = runs[mask]
Bs = Bs[mask]
amps = amps[mask]
Phs = Phs[mask]
ts = ts[mask]

lowBmask = (np.abs(Bs) < .1)
amps = amps/(np.mean(amps[lowBmask]))

colors = cm.jet(np.linspace(0, 1, len(Bs)))
    
plt.scatter(runs, ts, s=10, linewidth = .5,  zorder=1, color=colors, marker = '.')
plt.xlabel('Run Index')
plt.ylabel('T (K)')
#plt.title('Amplitude vs Run Index')
plt.show()

space = np.insert(np.diff(Bs), 0, 0)
Up = (space > 0)
Dw = (space <= 0)
baseBmax = np.amax(np.abs(Bs))
Bs[Up] -= offy
Bs[Dw] += offy
plt.scatter(np.abs(Bs[Up]), ts[Up], s=10, linewidth = .5,  zorder=1, marker = '.', color = 'red')
plt.scatter(np.abs(Bs[Dw]), ts[Dw], s=10, linewidth = .5,  zorder=1, marker = '.', color = 'blue')
plt.axvline(0)
plt.xlim((0, .1))
plt.ylim((3.3, 3.55))
plt.xlabel('B (T)')
plt.ylabel('T (K)')
#plt.title('Amplitude vs Run Index')
plt.show()

Bmask = (np.abs(np.abs(Bs) - baseBmax) > offy)
Bs = Bs[Bmask]
ts = ts[Bmask]
amps = amps[Bmask]
runs = runs[Bmask]
colors = colors[Bmask]

plt.scatter(np.abs(Bs), amps, s=3, linewidth = .5,  zorder=1, color=colors, marker = '.')
plt.xlabel('|B| (T)')
plt.ylabel(r'R/$R_{0}$')
#plt.ylim((.0148, .0156))
#plt.xlim((0, 2))
#plt.title('Amplitude vs Run Index')
plt.show()

tolo=.001
Bfp = np.amax(Bs)-tolo
if(StartCenter):
    start = next(ind for ind, val in enumerate(Bs) if np.abs(val) > Bfp)
else:
    start = next(ind for ind, val in enumerate(Bs) if np.abs(val) < tolo)
print(start)
Tpredict = interpolate.interp1d(np.abs(Bs[:start]), ts[:start], kind = 'linear', fill_value='extrapolate')
Apredict = interpolate.interp1d(np.abs(Bs[:start]), amps[:start], kind = 'linear', fill_value='extrapolate')
delT = ts - Tpredict(np.abs(Bs))
delA = amps - Apredict(np.abs(Bs))

plt.plot(runs, delA)
plt.xlabel('Run Index')
plt.ylabel(r'R/$R_{0}$')
plt.show()

window = 2000
smooT = savgol_filter(delT, 2*window + 1, 2)

plt.scatter(runs, delT, s=10, linewidth = .5,  zorder=1, color=colors, marker = '.')
plt.plot(runs, smooT)
plt.xlabel('Run Index')
plt.ylabel(r'$\delta$ T (K)')
#plt.title('Amplitude vs Run Index')
plt.show()

p = np.polyfit(smooT[start:], delA[start:], 2)
fitAT = np.poly1d(p)

plt.scatter(smooT[start:], delA[start:], s=3, linewidth = .5,  zorder=1, color=colors[start:], marker = '.')
plt.plot(smooT[start:], fitAT(smooT[start:]))
plt.xlabel(r'$\delta$ T (K)')
plt.ylabel('R/$R_{0}$')
#plt.ylim((.0148, .0156))
#plt.xlim((0, 2))
#plt.title('Amplitude vs Run Index')
plt.show()

AAdj = amps - fitAT(smooT)

plt.scatter(np.abs(Bs), AAdj, s=3, linewidth = .5,  zorder=1, color=colors, marker = '.')
plt.xlabel('|B| (T)')
plt.ylabel(r'R/$R_{0}$')
plt.savefig('6_10_lowsweep_overlay.pdf')
plt.show()

#write = np.vstack((Bs, AAdj))
#np.savetxt('6_10_RepSweep.txt', write)

if(False):
    amps = AAdj
    nbins = 70
    def groupData(Bs, amps, nbins):
        n, _ = np.histogram(Bs, bins=nbins)
        sy, _ = np.histogram(Bs, bins=nbins, weights=amps)
        sy2, binni = np.histogram(Bs, bins=nbins, weights=amps*amps)
        mean = sy/n
        std = np.sqrt(sy2/n - mean*mean)
        return (binni[1:] + binni[:-1])/2, mean, std
    
    
    bins, mean, _ = groupData(Bs, amps, nbins)
    smooAmps = savgol_filter(amps, 9, 2)
    _, _, std = groupData(Bs, amps-smooAmps, nbins)
    
    start = next(ind for ind, val in enumerate(Bs) if np.abs(val) > Bstart)
    startMask = [True]*start + [False]*(len(Bs)- start)
    upMask = [False]*len(Bs)
    dwMask = [False]*len(Bs)
    for i in range(start, len(Bs)):
        if(Bs[i] < Bs[i-1]):
            dwMask[i] = True
            upMask[i] = False
        else:
            dwMask[i] = False
            upMask[i] = True
    
    upBs = Bs[upMask]
    dwBs = Bs[dwMask]
    stBs = Bs[startMask]
    upAmps = amps[upMask]
    dwAmps = amps[dwMask]
    stAmps = amps[startMask]
    smUpAmps = smooAmps[upMask]
    smDwAmps = smooAmps[dwMask]
    smStAmps = smooAmps[startMask]
    
    upSmoother = interpolate.interp1d(upBs, upAmps, kind = 'linear', fill_value='extrapolate')
    dwSmoother = interpolate.interp1d(dwBs, dwAmps, kind = 'linear', fill_value='extrapolate')
    
    plt.scatter(stBs, stAmps-(upSmoother(stBs)+dwSmoother(stBs))/2, s=3, color='green', marker = '.', label = 'Start Sweep')
    plt.scatter(upBs, upAmps-dwSmoother(upBs), s=3, color='red', marker = '.', label = 'Up Sweep')
    plt.scatter(dwBs, dwAmps-upSmoother(dwBs), s=3, color='blue', marker = '.', label = 'Down Sweep')
    plt.xlabel('|B| (T)')
    plt.ylabel(r'R/$R_{0}$')
    plt.legend(loc='best')
    plt.savefig('Hist_6_10.pdf')
    plt.show()
    
    # plt.plot(stAmps, color='green', marker = '.', label = 'Start Sweep')
    # plt.plot(upAmps, color='red', marker = '.', label = 'Up Sweep')
    # plt.plot(dwAmps, color='blue', marker = '.', label = 'Down Sweep')
    # plt.xlabel('Data Order')
    # plt.ylabel(r'R/$R_{0}$')
    # plt.legend(loc='best')
    # plt.show()
    
    upbins, upmean, _ = groupData(upBs, upAmps, nbins)
    _, _, upstd = groupData(upBs, smUpAmps-upAmps, nbins)
    
    dwbins, dwmean, _ = groupData(dwBs, dwAmps, nbins)
    _, _, dwstd = groupData(dwBs, smDwAmps-dwAmps, nbins)
    
    stbins, stmean, _ = groupData(stBs, stAmps, nbins)
    _, _, ststd = groupData(stBs, smStAmps-stAmps, nbins)
    
    upPreds = dwSmoother(upbins)
    dwPreds = upSmoother(dwbins)
    stPreds = (dwSmoother(stbins) + upSmoother(stbins))/2
    
    #plt.errorbar(stbins, (stmean-stPreds), yerr = ststd, fmt ='g-', label = 'Start Sweep')
    plt.errorbar(upbins, (upmean-upPreds), yerr = upstd, fmt ='r.', label = 'Up Sweep')
    plt.errorbar(dwbins, (dwmean-dwPreds), yerr = dwstd, fmt ='b.', label = 'Down Sweep')
    plt.legend(loc='best')
    plt.ylabel(r'R/$R_{0}$')
    plt.xlabel('B (T)')
    plt.savefig('BinnedHist_6_10.pdf')
    plt.show()
    
    Writer = np.vstack((upbins, (upmean-upPreds), upstd, dwbins, (dwmean-dwPreds), dwstd))
    np.savetxt('Hist_6_10.txt', Writer)