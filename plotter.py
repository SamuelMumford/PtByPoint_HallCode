#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:46:57 2020
@author: tiffanypaul
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({'font.size': 18})

sigma2d = np.array([-1.8*10**(-8), .7*10**(-9), -11*10**(-9), -8*10**(-8)])

sigma = sigma2d/(50*10**(-9))

sigmaerr2d = np.array([4*10**(-9), 2*10**(-9), 7*10**(-9), 12*10**(-9)])

sigmaerr = sigmaerr2d/(50*10**(-9))

sigmaOld = np.array([2*10**(-7)])
sigmaErrOld = np.array([.1*10**(-7)])
rho2dOld = np.array([300-100])*2*np.pi/np.log(75/40)
print(rho2dOld)

tr = 25000*(50*10**(-9))

#fitco = [30.7, -10, .4]

#fiterr = [1.1, 2, 1]

sigmaBD = 4*np.array([2.97E-7])/3
simgaErrDB = 4*np.array([1.3E-7])/3
rho2dBD = np.array([8500])

HBSig = [3.3E-7, 2.7E-7, 1.29E-6, 1.2E-6]
HBSigErr = [3.3E-7*.002, 2.7E-7*.002, 1.29E-6*.004, 1.2E-6*.004]
HBRxx = np.array([13.5*1000, 15*1000, 4.4*1000, 4.6*1000])

rhoxx2D = (np.array([46000, 330000, 18700, 5000]) - 200)*2*np.pi/np.log(75/40)

rhoxx2DWiggle = (np.array([1900, 3450]) - 200)*2*np.pi/np.log(75/40)
print(rhoxx2DWiggle)
I = 3E-18
sigmaWiggle = np.array([-1.67E-6*(.2/.3), 1.08E-6*(.2/.3)])*(I/1E-17)
sigmaErrWiggle = np.array([sigmaWiggle[0]*.012/.057, sigmaWiggle[1]*.004/.037])

print(rhoxx2D)
rhoxx = rhoxx2D*(50*10**(-9))



rhoxy = -sigmaBD*(rho2dBD)**2
uncrhoxy = simgaErrDB*(rho2dBD)**2
print('rho_xy_BD')
print(rhoxy)
print(uncrhoxy)
cf = 5/(1.6E-19 * 50E-9 * 1E6)
den = cf/rhoxy
print(rhoxx2D)
print('densities bd')
print(den)
uncden = cf*uncrhoxy/(rhoxy**2)
print(uncden)


rhoxy = -sigmaOld*(rho2dOld)**2
uncrhoxy = sigmaErrOld*(rho2dOld)**2
print('rho_xy_old')
print(rhoxy)
print(uncrhoxy)
cf = 5/(1.6E-19 * 50E-9 * 1E6)
den = cf/rhoxy
print(rhoxx2D)
print('densities old')
print(den)
uncden = cf*uncrhoxy/(rhoxy**2)
print(uncden)


rhoxy = -sigma2d*rhoxx2D**2
uncrhoxy = sigmaerr2d*rhoxx2D**2
print('rho_xy')
print(rhoxy)
print(uncrhoxy)
cf = 5/(1.6E-19 * 50E-9 * 1E6)
den = cf/rhoxy
print(rhoxx2D)
print('densities')
print(den)
uncden = cf*uncrhoxy/(rhoxy**2)
print(uncden)

rhoxy = -sigmaWiggle*rhoxx2DWiggle**2
uncrhoxy = sigmaErrWiggle*rhoxx2DWiggle**2
print('rho_xy wiggle')
print(rhoxy)
print(uncrhoxy)
cf = 5/(1.6E-19 * 50E-9 * 1E6)
den = cf/rhoxy
print(rhoxx2D)
print('densities wiggle')
print(den)
uncden = cf*uncrhoxy/(rhoxy**2)
print(uncden)


plt.errorbar(rhoxx, sigma, yerr = sigmaerr, marker = '.', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.axvline(x=tr)
#plt.ylim(-2, 5)
plt.ylabel(r'$\sigma_{xy}$ ($\Omega^{-1}$ cm$^{-1}$)')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$ cm)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare1.pdf', bbox_inches="tight")

plt.show()

plt.errorbar(rhoxx, np.abs(sigma*rhoxx), yerr = sigmaerr*rhoxx, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.ylabel(r'$|\frac{\rho_{xy}}{\rho_{xx}}| \propto |\mu|$ ')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$ cm)')
#plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare2.pdf', bbox_inches="tight")

plt.show()

sq = '\u25A1'
print(sq)
plt.errorbar(rhoxx2D, sigma2d, yerr = sigmaerr2d, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = 'Resistive ITO, Cantilever')
plt.errorbar(rhoxx2DWiggle, sigmaWiggle, sigmaErrWiggle, marker = '^', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'green', label = 'Magnetic ITO, Cantilever')
plt.errorbar(HBRxx, -1*np.array(HBSig), HBSigErr, marker = 'x', linewidth = 0, elinewidth=1, capsize=2,  markersize=5, color = 'red', label = 'ITO Hall Bar')
plt.errorbar(rho2dBD, -1*np.array(sigmaBD), simgaErrDB, marker = '^', linewidth = 0, elinewidth=1, capsize=2,  markersize=5, color = 'green')

#plt.errorbar(rho2dOld, sigmaOld, sigmaErrOld, marker = 'd', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'orange', label = 'Deposited Conductive ITO')
#plt.plot(EFs, FE, c = 'red')
plt.axvline(x=26000)
#plt.ylim(-1E-7, 2.5E-7)
plt.ylabel(r'$\sigma_{xy}$ ([$\Omega/$' + sq + r']$^{-1}\times 10^{-6}$)')
plt.xlabel(r'$\rho_{xx}$ ($\Omega/$' + sq + ')')
plt.axhline(y=0, color='k', linestyle='-')
plt.xscale('log')
plt.legend(loc = 'best', prop={'size':12})
plt.savefig('Cooldown_compareAll.pdf', bbox_inches="tight")
#plt.grid()
plt.show()

plt.errorbar(rhoxx2D, sigma2d*1E6, yerr = sigmaerr2d*1E6, marker = '.', linewidth = 0, elinewidth=1, capsize=2, color = 'blue', label = 'Resistive ITO, Cantilever')
#plt.errorbar(rhoxx2DWiggle, sigmaWiggle, sigmaErrWiggle, marker = '^', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'green', label = 'Magnetic ITO, Cantilever')
#plt.errorbar(HBRxx, HBSig, HBSigErr, marker = 'x', linewidth = 0, elinewidth=1, capsize=2,  markersize=5, color = 'red', label = 'Annealed ITO, Hall Bar')
#plt.errorbar(rho2dOld, sigmaOld, sigmaErrOld, marker = 'd', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'orange', label = 'Deposited Conductive ITO')
#plt.plot(EFs, FE, c = 'red')
#plt.axvline(x=26000)
#plt.ylim(-1E-7, 2.5E-7)
plt.ylabel(r'$\sigma_{xy}$ ([$\Omega/$' + sq + r']$^{-1}\times 10^{-6}$)')
plt.xlabel(r'$\rho_{xx}$ ($\Omega/$' + sq + ')')
plt.axhline(y=0, color='k', linestyle='-')
plt.xscale('log')
#plt.legend(loc = 'best', prop={'size':12})
plt.savefig('Cooldown_compareHighR.pdf', bbox_inches="tight")
#plt.grid()
plt.show()



plt.errorbar(rhoxx2D, np.abs(sigma2d*rhoxx2D), yerr = sigmaerr2d*rhoxx2D, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.ylabel(r'$|\frac{\rho_{xy}}{\rho_{xx}}| \propto |\mu|$ ')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$)')
#plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare2.pdf', bbox_inches="tight")

plt.show()

cf = 1
mob = sigma2d*rhoxx2D/5
mobE = sigmaerr2d*rhoxx2D/5

mobO = sigmaOld*rho2dOld/5
mobEO = sigmaErrOld*rho2dOld/5

mobWigg = sigmaWiggle*rhoxx2DWiggle/5
mobWiggE = sigmaErrWiggle*rhoxx2DWiggle/5

mobHB = cf*np.array(HBSig*HBRxx)/5
mobHBErr = cf*np.array(HBSigErr*HBRxx)/5

mobBD = cf*np.array(sigmaBD*rho2dBD)/5
mobBDErr = cf*np.array(simgaErrDB*rho2dBD)/5

weights = 1/(mobE**2)
weights = weights/np.sum(weights)
mean = np.sum(mob*weights)
print(mean)
plt.errorbar(rhoxx2D,cf*mob, yerr = cf*mobE, marker = '.', linewidth = 0, elinewidth=1, capsize=2, c = 'blue')
plt.errorbar(rhoxx2DWiggle, cf*mobWigg, yerr = cf*mobWiggE, marker = '^', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'green', label = 'Magnetic ITO, Cantilever')
plt.errorbar(rho2dBD, -cf*mobBD, yerr = cf*mobBDErr, marker = '^', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'green')
plt.errorbar(HBRxx, -mobHB, yerr = mobHBErr, marker = 'x', linewidth = 0, elinewidth=1, capsize=2,  markersize=5, color = 'red', label = 'ITO Hall Bar')
#plt.errorbar(rho2dOld, -cf*mobO, yerr = cf*mobEO, marker = 'd', linewidth = 0, elinewidth=1, capsize=2, markersize=5, color = 'orange', label = 'Deposited Conductive ITO')
#plt.plot(EFs, FE, c = 'red')
plt.axhline(y=mean, color = 'r')
plt.axhline(y=0, color = 'k')
plt.ylabel(r'$\sigma_{xy}\rho_{xx}$ ')
plt.xlabel(r'$\rho_{xx}$ ($\Omega/$' + sq + ')')
plt.axvline(x=26000)
#plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
plt.savefig('rat.pdf', bbox_inches="tight")

plt.show()