#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:46:49 2020

@author: tiffanypaul
"""



import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({'font.size': 18})

sigma2d = np.array([2*10**(-7), -1.8*10**(-8), .7*10**(-9), -11*10**(-9), -8*10**(-8)])

sigma = sigma2d/(50*10**(-9))

sigmaerr2d = np.array([.1*10**(-7), 4*10**(-9), 2*10**(-9), 7*10**(-9), 12*10**(-9)])

sigmaerr = sigmaerr2d/(50*10**(-9))

tr = 25800*(50*10**(-9))

#fitco = [30.7, -10, .4]

#fiterr = [1.1, 2, 1]

rhoxx2D = np.array([400, 46000, 330000, 18700, 5000])*2*np.pi/np.log(75/40)#np.array([0.02, 2.3, 16.5, .94, .2])
print('rhoxx2D (Ohms)')
print(rhoxx2D)
rhoxx = rhoxx2D*(50*10**(-9))
print('rhoxx (Ohm m)')
print(rhoxx)

plt.errorbar(rhoxx*100, sigma/100, yerr = sigmaerr/100, marker = '.', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.vlines(tr*100, -.02, .05)
plt.ylim(-.02, .05)
plt.ylabel(r'$\sigma_{xy}$ ($\Omega^{-1}$ cm$^{-1}$)')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$ cm)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare1.pdf', bbox_inches="tight")

plt.show()

plt.errorbar(rhoxx*100, np.abs(sigma*rhoxx), yerr = sigmaerr*rhoxx, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.ylabel(r'$|\frac{\rho_{xy}}{\rho_{xx}}| \propto |\mu|$ ')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$ cm)')
#plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare2.pdf', bbox_inches="tight")

plt.show()

plt.errorbar(rhoxx2D, sigma2d, yerr = sigmaerr2d, marker = '.', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.vlines(25800, -1E-7, 2.5E-7)
plt.ylim(-1E-7, 2.5E-7)
plt.ylabel(r'$\sigma_{xy}$ ($\Omega^{-1}$)')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare1.pdf', bbox_inches="tight")

plt.show()

plt.errorbar(rhoxx2D, np.abs(sigma2d*rhoxx2D), yerr = sigmaerr2d*rhoxx2D, marker = 'o', linewidth = 0, elinewidth=1, capsize=2, c = 'black')
#plt.plot(EFs, FE, c = 'red')
plt.ylabel(r'$|\frac{\rho_{xy}}{\rho_{xx}}| \propto |\mu|$ ')
plt.xlabel(r'$\rho_{xx}$ ($\Omega$)')
#plt.axhline(y=0, color='r', linestyle='-')
plt.xscale('log')
#plt.savefig('/Users/tiffanypaul/Desktop/Cooldown_compare2.pdf', bbox_inches="tight")

plt.show()