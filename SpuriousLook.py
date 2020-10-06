import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt

def main():
    
    #Read in the file
    fname = 'C:/Users/sammy/Downloads/Cooldown_all_10_2_20.txt'
    f2 = 'C:/Users/sammy/Downloads/Cooldown_all_10_2_20L.txt'
    fB = 'C:/Users/sammy/Downloads/Cooldown_all_10_2_20.txt'
    
    Fdat = np.loadtxt(fname, delimiter="\t")
    F2 = np.loadtxt(f2, delimiter="\t")
    FdatB = np.loadtxt(fB, delimiter="\t")
    startcut = 350
    cutEnd = False
    if(cutEnd):
        #Two endcuts have been used. 5600 gives you more or less the whole file
        #Before the temperature gets out of hand. 4520 is all the data before
        #the .2K rise
        endcut = 5400
        freqs = Fdat[startcut:endcut, 2]
        runs = Fdat[startcut:endcut, 3]
        temps = Fdat[startcut:endcut, 4]
        currs = Fdat[startcut:endcut, 5]
        dc = Fdat[startcut:endcut, 0]
        amp = F2[startcut:endcut, 6]
        Q = F2[startcut:endcut, 3]
    else:
        freqs = Fdat[startcut:, 2]
        runs = Fdat[startcut:, 3]
        temps = Fdat[startcut:, 4]
        currs = Fdat[startcut:, 5]
        dc = Fdat[startcut:, 0]
        amp = F2[startcut:, 6]
        Q = F2[startcut:, 3]
    #Convert current to magnetic field and make Qs positive
    Bs = currs*.85/5 
    Q = np.abs(Q)
    
    scB = 0
    cutEndB = False
    if(cutEndB):
        #Two endcuts have been used. 5600 gives you more or less the whole file
        #Before the temperature gets out of hand. 4520 is all the data before
        #the .2K rise
        endcutB = 4520
        rb = FdatB[scB:endcutB, 3]
        tb = FdatB[scB:endcutB, 4]
        cb = FdatB[scB:endcutB, 5]
    else:
        rb = FdatB[scB:, 3]
        tb = FdatB[scB:, 4]
        cb = FdatB[scB:, 5]
    Bsb = cb*.85/5
    
    #We need to calculate the temperature by getting rid of the magnetoresistance.
    #To do that, we need to get rid of any runs that were at a different temperature
    DelOutliersTemp = False
    #We may or may not want to use those different-temperature data points in the 
    #full analysis. I still cut them out, it doesn't make a real difference now though
    DelOutliersFit = True
    #If you're cutting out data sections, enter this loop
    if(DelOutliersFit):
        #Run 17 is the weird run f
        Outliers = []
        #We're going to make arrays with the data (f, Q, T, etc.) from each part
        #of the dataset. If we want to keep that part of the dataset, we append
        #the new data to the running list of all data
        
        #Initialize indexes of where data sections start and how many fields we
        #have looked at
        startIndex = 0
        index = 0
        #Initialize data arrays
        fs, rs, ts, bs, ds, ams, qs = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        for j in range(len(runs)-1):
            #If the current changed, you have found the end of a section of data
            if(currs[j] != currs[j+1] or j ==len(runs)-2):
                #If you don't want this section of data, don't add it to the list
                #Just change the indexes to start looking at the next section
                if(index in Outliers):    
                    startIndex = j+1
                    index += 1
                #If you want to analyze this section of data, add it to the list 
                #of all data
                else:
                    rs = np.append(rs, runs[startIndex:j])
                    fs = np.append(fs, freqs[startIndex:j])
                    ts = np.append(ts, temps[startIndex:j])
                    bs = np.append(bs, Bs[startIndex:j])
                    ds = np.append(ds, dc[startIndex:j])
                    ams = np.append(ams, amp[startIndex:j])
                    qs = np.append(qs, Q[startIndex:j])
                    startIndex = j+1
                    index += 1
        #If you're cutting outlier sections from all analysis, just redefine the
        #base data arrays to only have the good data
        if DelOutliersFit:
            runs = rs
            freqs = fs
            temps = ts
            Bs = bs
            dc = ds
            amp= ams
            Q = qs
    

    runsTemp = rb
    tempsTemp = tb
    BsTemp = Bsb
    
    #Define a polynomial fit in B and the run index for the temperature value
    #that we read out    
    def TFit(data, Bcs, Rcs):
        runs = data[:, 0]
        Bs = data[:, 1]
        
        fit = np.zeros(len(runs))
        i = 2
        #The m's are the fit coefficients, doing only an even fit in B
        for m in zip(Bcs):
            fit += m*np.power(Bs, i)
            i += 2
        #n's are fit coefficients, allow for an offset term in the run index fit
        j = 0
        rus = runs/np.max(runs)
        for n in zip(Rcs):
            fit += n*np.power(rus, j)
        return fit
    
    #When you call the curve_fit code, it can ONLY take things of the form 
    #function(x, args). We need to make a wrapping function to put TFit in 
    #that form.        
    def wrapT(x, *args):
        #Note that NTs is a global variable telling which indexes separate
        #the B and run fit coefficients
        polyBs = list(args[:NTs[0]])
        polyRs = list(args[NTs[0]:])
        return TFit(x, polyBs, polyRs)
    
    #The order of the even in B polynomial fit and run index fits
    BOrd = 5
    ROrd = 2
    #We use NTs in wrapT, so we make it a global variable. Gives the index
    #of fit variables of different types
    global NTs
    NTs = ([BOrd, BOrd + ROrd])
    pT = np.zeros(BOrd + ROrd)
    #Have a run index after which temperature rises
    tendCut = max(runsTemp)
    tstartCut = 400
    #Find that index in the data arrays
    place = np.where(runsTemp == tendCut)[0]
    print(runsTemp)
    #Keep only data before the cutoff index
    pT[BOrd] = np.mean(tempsTemp[tstartCut:place[0]])
    rsStart = runsTemp[tstartCut:place[0]]
    BsStart = BsTemp[tstartCut:place[0]]
    tempsStart = tempsTemp[tstartCut:place[0]]
    
    def tdmaker(Bs, trueTs):
        td = np.zeros(0)
        startIndex = 0
        for i in range(0, len(Bs)-1):
            if(Bs[i] != Bs[i+1]):
                td = np.append(td, np.gradient(trueTs[startIndex:i+1], 1))
                startIndex = i+1
        td = np.append(td, np.gradient(trueTs[startIndex:], 1))
        return td
    
    tSl = np.abs(tdmaker(BsStart, tempsStart))
    plt.plot(rsStart, tSl)
    plt.ylim(0, .002)
    plt.show()
    limmy = 1
    crit = (tSl < limmy)
    rsFin = rsStart[crit]
    BsFin = BsStart[crit]
    tsFin = tempsStart[crit]
    #Stack the runs and B data so that it can be passed as one variable to the fit function
    RandB = np.vstack((rsFin, BsFin)).T
    #Call the fit function, this syntax is terrible but works. lambda essentially 
    #defines a new function that we can pass as our fit function
    popt, pcov = curve_fit(lambda RandB, *pT: wrapT(RandB, *pT), RandB, tsFin, p0=pT)
    #if you want to see fit results for temperature, uncomment here
    print(popt)
    print(popt/np.sqrt(np.diag(pcov)))
    
    #In order to get the true temperature, we need to subtract off the magnetic 
    #field part. So define a parameter array where the run-dependent fit parts are 0
    #and the magnetic field dependent parts are the results of the fit
    pSub = np.zeros(len(pT))
    pSub[:BOrd] = popt[:BOrd]
    
    
    #I made a fit for f0 as a function of temperature, Q, and magnetic field.
    #In retrospect, somewhat unnecessary, but it works and helps with later smoothing
    #to have big jumps taken out

    
    plt.plot(rsFin, tsFin)
    plt.plot(rsFin, wrapT(RandB, *popt))
    plt.show()
    
    #define the true temperature calling the fit function used earlier
    #but with the run terms set to 0
    TandBFull = np.vstack((runs, Bs)).T
    trueTs = temps - wrapT(TandBFull, *pSub)
    
    def secSmoother(Bs, dat, sz):
        td = np.zeros(0)
        startIndex = 0
        for i in range(0, len(Bs)-1):
            if(Bs[i] != Bs[i+1]):
                td = np.append(td, signal.savgol_filter(dat[startIndex:i+1], sz, 1))
                startIndex = i+1
        td = np.append(td, signal.savgol_filter(dat[startIndex:], sz, 1))
        return td
    
    print(min(dc))
    print(max(dc))
    mini = min(.015, min(dc))
    maxi = max(.095, max(dc))
    interf = np.sin(np.pi*((dc - mini)/(maxi - mini)))
    trueAmp = np.sqrt(amp/((Q*interf)**2))
    smAmp = secSmoother(Bs, trueAmp, 61)
    mod = np.mean(smAmp[0:4500])
    smAmp = smAmp/mod
    trueAmp = trueAmp/mod
    
    tdiff = tdmaker(Bs, trueTs)
    plt.plot(np.abs(tdiff))
    plt.ylim(.0001, .001)
    plt.yscale('log')
    plt.show()
    mask = (np.abs(tdiff) < .0004)
    
    
    runs = runs[mask]
    trueTs = trueTs[mask]
    trueAmp = trueAmp[mask]
    Q = Q[mask]
    freqs = freqs[mask]
    Bs = Bs[mask]
    
    plt.plot(runs, freqs)
    plt.show()
    
    plt.plot(runs, trueTs)
    plt.show()
    
    plt.plot(runs, trueAmp)
    plt.show()
    
    def secSplit(Bs, data):
        dm = np.zeros(0)
        ds = np.zeros(0)
        startIndex = 0
        for i in range(0, len(Bs)-1):
            if(Bs[i] != Bs[i+1]):
                dm = np.append(dm, np.mean(data[startIndex:i+1]))
                ds = np.append(ds, np.std(data[startIndex:i+1])/np.sqrt(len(data[startIndex:i+1])))
                startIndex = i+1
        dm = np.append(dm, np.mean(data[startIndex:]))
        ds = np.append(ds, np.std(data[startIndex:])/np.sqrt(len(data[startIndex:])))
        return dm, ds*np.sqrt(2)
    
    delF = freqs - secSmoother(Bs, freqs, 11)#21)
    plt.plot(runs, delF)
    plt.show()
    EO = 2*(runs%2 - .5)
    shift = EO*delF
    plt.plot(runs, shift)
    plt.show()
    
    shifts, shiftDevs = secSplit(Bs, shift)
    Ts, TDevs = secSplit(Bs, trueTs)
    Amps, AmpDevs = secSplit(Bs, trueAmp)
    Qs, QDevs = secSplit(Bs, Q)
    Blist, _ = secSplit(Bs, Bs)
    plt.errorbar(Blist, shifts, yerr = shiftDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    def linfit(bs, a, b):
        return a + bs*b
    def quadfit(bs, a, b, c):
        return a + bs*b + bs*bs*c
    pLin, pcovLin = curve_fit(linfit, Blist, shifts, p0 = [0,0], sigma = shiftDevs)
    pQuad, pcovQuad = curve_fit(quadfit, Blist, shifts, p0 = [0,0,0], sigma = shiftDevs)
    predLin = linfit(Blist, pLin[0], pLin[1])
    predQuad = quadfit(Blist, pQuad[0], pQuad[1], pQuad[2])
    diffLin = shifts - predLin
    diffQuad = shifts - predQuad
    
    order = np.linspace(0, len(shifts)-1, len(shifts))
    plt.errorbar(Blist, diffQuad, yerr = shiftDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    plt.errorbar(Ts, diffQuad, yerr = shiftDevs, xerr = TDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    plt.errorbar(Qs, diffQuad, yerr = shiftDevs, xerr = QDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    plt.errorbar((Amps-1)*Blist, diffQuad, yerr = shiftDevs, xerr = Blist*AmpDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    power = 1
    plt.errorbar(((Ts - np.mean(Ts))**power)*Blist, diffQuad, yerr = shiftDevs, xerr = TDevs*Blist, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    plt.errorbar((Qs - np.mean(Qs))*Blist, diffQuad, yerr = shiftDevs, xerr = QDevs*Blist, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    newParam = ((Ts - np.mean(Ts))**power)*Blist
    data = np.vstack((Blist, newParam)).T
    def quadNew(xs, a, b, c, n):
        bs = xs[:, 0]
        newp = xs[:, 1]
        return a + bs*b + bs*bs*c + n*newp
    npa, ncov = curve_fit(quadNew, data, shifts, p0 = [pQuad[0],pQuad[1],pQuad[2], 0], sigma = shiftDevs)
    print(npa)
    print(np.sqrt(np.diag(ncov)))
    print(npa/np.sqrt(np.diag(ncov)))
    preds = quadNew(data, npa[0], npa[1], npa[2], npa[3])
    diffNew = shifts - preds
    plt.errorbar(Blist, diffNew, yerr = shiftDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.show()
    
    plt.plot(order, diffNew/shiftDevs, marker = 'o', linewidth = 0)
    plt.show()
    
    compData = shifts - npa[-1]*newParam
    Bs = np.linspace(min(Blist), max(Blist), 100)
    plt.errorbar(Blist, compData, yerr = shiftDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(Bs, quadfit(Bs, npa[0], npa[1], npa[2]))
    plt.show()
    
    runBs = Blist
    runShifts = compData
    runDevs = shiftDevs
    
    allBs = np.zeros(0)
    avgVal = np.zeros(0)
    weight = np.zeros(0)
    for i in range(0, len(runBs)):
        b=np.where(np.abs(allBs - runBs[i]) < .01)[0]
        if(len(b) == 0):
            print(runBs[i])
            allBs = np.append(allBs, runBs[i])
            avgVal = np.append(avgVal, runShifts[i]/(runDevs[i]**2))
            weight = np.append(weight, 1/(runDevs[i]**2))
        else:
            avgVal[b] += runShifts[i]/(runDevs[i]**2)
            weight[b] += 1/(runDevs[i]**2)
    avgVal = avgVal/weight
    sigm = 1/np.sqrt(weight)
    
    plt.errorbar(allBs, avgVal, yerr = sigm, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(Bs, quadfit(Bs, npa[0], npa[1], npa[2]))
    plt.ylabel('Data (Hz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
    eBs = np.zeros(0)
    eaV = np.zeros(0)
    eVar = np.zeros(0)
    for i in range(0, len(allBs)):
        b=np.where(np.abs(eBs - np.abs(allBs[i])) < .01)[0]
        if(len(b) == 0):
            eBs = np.append(eBs, np.abs(allBs[i]))
            if(runBs[i] == 0):
                eaV = np.append(eaV, avgVal[i])
                eVar = np.append(eVar, sigm[i]**2)
            else:
                eaV = np.append(eaV, avgVal[i]/2)
                eVar = np.append(eVar, sigm[i]**2/4)
        else:
            eaV[b] += avgVal[i]/2
            eVar[b] += sigm[i]**2/4
    eSig = np.sqrt(eVar)
    
    def ffE(x, a, b):
        return a + b*x**2
    
    def fE(x, a):
        return a*np.ones(len(x))
    p0 = [.56, 0]
    pE, pcovE = curve_fit(ffE, eBs, eaV, p0 = p0, sigma = eSig)
    pEBase, pcovEbase = curve_fit(fE, eBs, eaV, p0 = [.56], sigma = eSig)
    EFs = np.linspace(0, max(eBs), 100)
    FE = ffE(EFs, pE[0], pE[1])
    FE2 = fE(EFs, .56)
    preds = ffE(eBs, pE[0], pE[1])
    diff = (preds - eaV)/eSig
    print(np.mean(np.abs(diff)))
    preds = fE(eBs, pEBase[0])
    diff = (preds - eaV)/eSig
    print(np.mean(np.abs(diff)))
    
    print('Even Fit Parameters')
    print(pE)
    print('Even Fit Sig')
    print(np.sqrt(np.diag(pcovE)))
    print('Even Fit Parameters in Std Dev')
    print(pE/np.sqrt(np.diag(pcovE)))
    plt.errorbar(eBs, eaV, yerr = eSig, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(EFs, FE)
    #plt.plot(EFs, FE2)
    plt.ylabel('Even Data (mHz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
if __name__=="__main__":
    main() 
#popt, pcov = curve_fit(lambda RunData, *p0: wrapper(RunData, Ns *p0), RunData, fs, p0=p0)