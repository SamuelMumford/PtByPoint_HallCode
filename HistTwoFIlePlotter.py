#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:54:34 2021

@author: sam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

F610 = np.loadtxt('Hist_6_10.txt')
F410Up = np.loadtxt('Hist_4_10_up.txt')
F410Dw = np.loadtxt('Hist_4_10_dw.txt')

upB6 = F610[0, :]
upR6 = F610[1, :]
upU6 = F610[2, :]
dwB6 = F610[3, :]
dwR6 = F610[4, :]
dwU6 = F610[5, :]

upB4 = F410Up[0, :]
upR4 = F410Up[1, :]
upU4 = F410Up[2, :]

dwB4 = F410Dw[0, :]
dwR4 = F410Dw[1, :]
dwU4 = F410Dw[2, :]

fBs = (np.abs(upB4) < 4)
dBs = (np.abs(dwB4) < 4)

smu6 = savgol_filter(upR6, 3, 1)
smd6 = savgol_filter(dwR6, 3, 1)
smu4 = savgol_filter(upR4[fBs], 1, 0)
smd4 = savgol_filter(dwR4[dBs], 1, 0)

plt.errorbar(upB6, upR6, yerr = upU6, fmt ='mx', label = r'20$^\circ$ Up Sweep', alpha = .5)
plt.plot(upB6, smu6, color = 'm')
plt.errorbar(dwB6, dwR6, yerr = dwU6, fmt ='cx', label = r'20$^\circ$ Down Sweep', alpha = .5)
plt.plot(dwB6, smd6, color = 'c')
plt.errorbar(upB4, upR4, yerr = upU4, fmt ='r.', label = r'Vert. Up Sweep', alpha = .5)
plt.plot(upB4[fBs], smu4, color = 'r')
plt.errorbar(dwB4, dwR4, yerr = dwU4, fmt ='b.', label = r'Vert. Down Sweep', alpha = .5)
plt.plot(dwB4[dBs], smd4, color = 'b')
plt.legend(loc='best',prop={'size':8})
plt.xlim(np.amin(dwB6), np.amax(upB6))
plt.ylabel(r'$\delta$R/$R_{0}$')
plt.xlabel('B (T)')
plt.savefig('Hist_2file.pdf', bbox_inches='tight')
plt.show()