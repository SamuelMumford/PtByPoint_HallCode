import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import matplotlib.pyplot as plt

def main():
    
    #Read in the file
    fname = 'C:/Users/sammy/Downloads/Edited_cooldown_all_8_16_20.txt'
    f2 = 'C:/Users/sammy/Downloads/Edited_cooldown_all_8_16_20L.txt'
    fB = 'C:/Users/sammy/Downloads/Edited_cooldown_all_7_25.txt'
    
    Fdat = np.loadtxt(fname, delimiter="\t")
    F2 = np.loadtxt(f2, delimiter="\t")
    FdatB = np.loadtxt(fB, delimiter="\t")
    startcut = 150
    cutEnd = False
    if(cutEnd):
        #Two endcuts have been used. 5600 gives you more or less the whole file
        #Before the temperature gets out of hand. 4520 is all the data before
        #the .2K rise
        endcut = 5600
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
    
    scB = 50
    cutEndB = True
    if(cutEndB):
        #Two endcuts have been used. 5600 gives you more or less the whole file
        #Before the temperature gets out of hand. 4520 is all the data before
        #the .2K rise
        endcutB = -70
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
        Outliers = [22]
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
        for n in zip(Rcs):
            fit += n*np.power(runs, j)
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
    ROrd = 1
    #We use NTs in wrapT, so we make it a global variable. Gives the index
    #of fit variables of different types
    global NTs
    NTs = ([BOrd, BOrd + ROrd])
    pT = np.zeros(BOrd + ROrd)
    #Have a run index after which temperature rises
    tendCut = 3570
    tstartCut = 170
    #Find that index in the data arrays
    place = np.where(runsTemp == tendCut)[0]
    print(runsTemp)
    #Keep only data before the cutoff index
    pT[BOrd] = np.mean(tempsTemp[tstartCut:place[0]])
    #Stack the runs and B data so that it can be passed as one variable to the fit function
    RandB = np.vstack((runsTemp[tstartCut:place[0]], BsTemp[tstartCut:place[0]])).T
    #Call the fit function, this syntax is terrible but works. lambda essentially 
    #defines a new function that we can pass as our fit function
    popt, pcov = curve_fit(lambda RandB, *pT: wrapT(RandB, *pT), RandB, tempsTemp[tstartCut:place[0]], p0=pT)
    #if you want to see fit results for temperature, uncomment here
#    print(popt)
#    print(popt/np.sqrt(np.diag(pcov
    
    #In order to get the true temperature, we need to subtract off the magnetic 
    #field part. So define a parameter array where the run-dependent fit parts are 0
    #and the magnetic field dependent parts are the results of the fit
    pSub = np.zeros(len(pT))
    pSub[:BOrd] = popt[:BOrd]
    
    
    #I made a fit for f0 as a function of temperature, Q, and magnetic field.
    #In retrospect, somewhat unnecessary, but it works and helps with later smoothing
    #to have big jumps taken out
    
    #define the true temperature calling the fit function used earlier
    #but with the run terms set to 0
    TandBFull = np.vstack((runs, Bs)).T
    trueTs = temps - wrapT(TandBFull, *pSub)
    #Define the data for f0 fitting
    tdiff = np.gradient(trueTs, 1)
    
    plt.plot(runs, trueTs)
    plt.show()
    
    def tdmaker(Bs, trueTs):
        td = np.zeros(0)
        startIndex = 0
        for i in range(0, len(Bs)-1):
            if(Bs[i] != Bs[i+1]):
                td = np.append(td, np.gradient(trueTs[startIndex:i+1], 1))
                startIndex = i+1
        td = np.append(td, np.gradient(trueTs[startIndex:], 1))
        return td
    
    tdiff = tdmaker(Bs, trueTs)
    
    RunData = np.vstack((runs, trueTs, Q, Bs, tdiff)).T
    #Define order of polynomial fits. NBSigm must be 2, others can be changed to anything
    #Also one sigmoid in B^2
#    N_fbase = 3
#    N_temp = 3
#    N_Q = 0
#    NBSigm = 2
#    N_BPoly = 7
    N_fbase = 3
    N_temp = 3
    N_Q = 0
    NBSigm = 2
    N_BPoly = 7
    N_tderiv = 1
    #Make a global variable for the number of each fit variable type
    global Ns
    Ns = [N_fbase, N_temp, N_Q, NBSigm, N_BPoly, N_tderiv]
    #Initialize fit parameters
    p0 = np.zeros(sum(Ns))
    p0[0] = np.mean(freqs)
    #Convert the Ns array to the indexes separating varaible types
    for i in range(1, len(Ns)):
        Ns[i] = Ns[i] + Ns[i-1]
      
    #Define a sigmoid for ease later
    def sigmoid(v):
        return 1/(1 + np.exp(-v))
    
    #Call a wrapper for the f0 fitting. Same idea as the run and B-field fit
    def wrapper(x, *args): 
        fCoeff = list(args[:Ns[0]])
        tCoeff = list(args[Ns[0]:Ns[1]])
        QCoeff = list(args[Ns[1]:Ns[2]])
        BSigmCoeff = list(args[Ns[2]:Ns[3]])
        BCoeff = list(args[Ns[3]:Ns[4]])
        TDCoeff = list(args[Ns[4]:Ns[5]])
        return fit_func(x, fCoeff, tCoeff, QCoeff, BSigmCoeff, BCoeff, TDCoeff)
    
    #The fit function. Polynomials and one sigmoid
    def fit_func(data, fCoeff, tCoeff, QCoeff, BSigmCoeff, BCoeff, TDCoeff):
        runs = data[:, 0]
        temps = data[:, 1]
        Qs = data[:, 2]
        Bs = data[:, 3]
        TDs = data[:, 4]
        fit = np.zeros(len(runs))
        i = 0
        for a in zip(fCoeff):
            fit += a*np.power(runs, i)
            i += 1
        j = 1
        for a in zip(tCoeff):
            fit += a*np.power(temps, j)
            j += 1
        j = 1
        for a in zip(QCoeff):
            fit += a*np.power(Qs, j)
            j += 1
        j = 1
        for a in zip(BCoeff):
            fit += a*np.power(Bs, j)
            j += 1
        j = 1
        for a in zip(TDCoeff):
            fit += a*np.power(TDs, j)
            j += 1
        fit += BSigmCoeff[0]*(sigmoid((Bs/BSigmCoeff[1])**2) -.5)
        return fit
    #Initialize p0 to some reasonable values in base f0 and B^2 terms
    p0[0] = np.min(freqs)
    p0[Ns[2]] = .5
    p0[Ns[2]+1] = 1
    #Call the f0 fit function and get parameters
    popt2, pcov2 = curve_fit(lambda RunData, *p0: wrapper(RunData, *p0), RunData, freqs, p0=p0)
    print('f0 fit parameters in order: run index polynomial, temperature polynomial, Q polynomial, B^2 sigmoid, and B-polynomial')
    print(popt2)
#    print('fit coeff in std. dev.')
#    print(pcov2)
    
    #Uncomment this if you want to see your f0 and fit of f0
    plt.plot(runs, freqs)
    plt.plot(runs, wrapper(RunData, *popt2))
    plt.ylabel("Frequency and Fit")
    plt.show()
    
    #Note we did not have any even/odd or voltage dependent parts in the f0 fit.
    #So the leftover parts after you subtract the fit should have that info, but
    #without the messy offsets
    leftover =  freqs - wrapper(RunData, *popt2)
    #Make a smoothed version of the residual/v-dependent part
    def secAvg(left, Bs):
        sa = np.zeros(0)
        startIndex = 0
        for i in range(0, len(Bs)-1):
            if(Bs[i] != Bs[i+1]):
                sa = np.append(sa, np.mean(left[startIndex:i+1])*np.ones(len(left[startIndex:i+1])))
                startIndex = i+1
        sa = np.append(sa, np.mean(left[startIndex:])*np.ones(len(left[startIndex:])))
        return sa
    secA = secAvg(leftover, Bs)
    size = 21
    sm = signal.savgol_filter(leftover - secA, size, 1)
    
    #Uncomment this if you want to see plots of how well our fits for f0 are doing
    #without accounting for V. I used this to justify cutting out run 17.
    plt.plot(runs, leftover)
    size0 = 1
    plt.plot([min(runs), max(runs)], [.002, .002], 'y')
    plt.plot([min(runs), max(runs)], [-.002, -.002], 'y')
    plt.plot([min(runs), max(runs)], [.004, .004], 'r')
    plt.plot([min(runs), max(runs)], [-.004, -.004], 'r')
    plt.ylabel("Frequency Fit Residual")
    plt.show()
    plt.plot(runs, leftover - secA - sm)
    plt.plot([min(runs), max(runs)], [.002, .002], 'y')
    plt.plot([min(runs), max(runs)], [-.002, -.002], 'y')
    plt.plot([min(runs), max(runs)], [.004, .004], 'r')
    plt.plot([min(runs), max(runs)], [-.004, -.004], 'r')
    plt.show()
    
    #Decide if you want to subtract off offsets in the leftover data as you know any
    #linear in V terms will avearge out to 0
    ZeroOut = True
    if(ZeroOut):
        leftover = leftover - secA - sm
        
    # a = 2040
    # b = 2375
    
    # tdiff = np.gradient(trueTs, 1)
    # plt.plot(runs[a:b], leftover[a:b])
    # plt.plot(runs[a:b], -.4*tdiff[a:b])
    # plt.ylabel("Frequency and Deriv Fit")
    # plt.show()
    
    #Convert the even/odd data to -1 and 1
    EO = 2*(runs%2 - .5)
    
    #Get a sense of the std. dev. on the leftover data
    size2 = 81
    distStart = np.sqrt(leftover**2)
    smDist = np.sqrt(signal.savgol_filter(leftover**2, size2, 1))

    #If CutOutliers is true, cut out all points from the dataset which are 
    #more than 'cut' standard deviations from 0.
    CutOutliers = True
    if(CutOutliers):
        cut = 2
        pts = len(leftover)
        mask = np.ones(pts, dtype=bool)
        for i in range(0, pts):
            if(distStart[i] > cut*smDist[i]):
                mask[i] = False
        leftover = leftover[mask]
        trueTs = trueTs[mask]
        Q = Q[mask]
        Bs = Bs[mask]
        EO = EO[mask]
        runs = runs[mask]
        plt.plot(runs, leftover)
        plt.plot([min(runs), max(runs)], [.002, .002], 'y')
        plt.plot([min(runs), max(runs)], [-.002, -.002], 'y')
        plt.plot([min(runs), max(runs)], [.004, .004], 'r')
        plt.plot([min(runs), max(runs)], [-.004, -.004], 'r')
        plt.show()
        distStart = np.sqrt(leftover**2)
        smDist = np.sqrt(signal.savgol_filter(leftover**2, size2, 1))
        #Uncomment if you want to see how many points we are cutting out
        plt.plot(distStart)
        plt.plot(cut*smDist)
        plt.show()
    
        
    #VERY IMPORTANT STEP. I multiply the remainder data by the +-1 even/odd
    #index, or the sign of voltage applied. This makes it so that signal looks
    #like an offset instead of noise. Very helpful in determining if fits are 
    #working
    leftover = leftover*EO
    
    #Do the same procedure we did on the f0 fit, now just fitting the voltage-dependent data
    #Don't change the offset N_V from 1, others can be changed at will
    global NEO
    N_V = 1
    N_VT = 3
    N_VQ = 2
    N_VB = 2
    NEO = [N_V, N_VT, N_VQ, N_VB]
    p0 = np.zeros(sum(NEO))
    for i in range(1, len(NEO)):
        NEO[i] = NEO[i] + NEO[i-1]
    EOData = np.vstack((EO, trueTs, Q, Bs)).T
    
    #Same procedure as the f0 data, split up the fit coefficients and pass them
    #to a fitting function
    def wrapEO(x, *args):
        V_Coeff = list(args[:NEO[0]])
        VT_Coeff = list(args[NEO[0]:NEO[1]])
        VQ_Coeff = list(args[NEO[1]:NEO[2]])
        VB_Coeff = list(args[NEO[2]:NEO[3]])
        return fit_EO(x, V_Coeff, VT_Coeff, VQ_Coeff, VB_Coeff)
    #the actual fit function
    def fit_EO(data, V_Coeff, VT_Coeff, VQ_Coeff, VB_Coeff):
        eo = data[:, 0]
        ts = data[:, 1] - np.mean(data[:, 1])
        Qs = data[:, 2]
        Qs = (Qs - np.mean(Qs))/np.std(Qs)
        Bs = data[:, 3]
        plt.show()
        fit = np.zeros(len(eo))
        for a in zip(V_Coeff):
            fit += a       
        j = 1
        for a in zip(VT_Coeff):
            fit += a*np.power(ts, j)
            j += 1        
        j = 1
        for a in zip(VQ_Coeff):
            fit += a*np.power(Qs, j)
            j += 1            
        j = 1
        for a in zip(VB_Coeff):
            fit += a*np.power(Bs, j)
            j += 1
        return fit
    
    #Call the fit function
    poptV, pcovV = curve_fit(lambda EOData, *p0: wrapEO(EOData, *p0), EOData, leftover, p0=p0)
    print('Fit coefficients in the order: 0 field offset, temperature polynomial, Q polynomial, B polynomial')
    print(poptV)
    print('Fit coeff. std. dev')
    print(np.sqrt(np.diag(pcovV)))
    print('Fit coeff. in std. dev. from 0')
    print(poptV/np.sqrt(np.diag(pcovV)))
    
    #Make a plot a fit
    pred = wrapEO(EOData, *poptV)
    plt.plot(leftover)
    plt.plot(pred)
    plt.xlabel('Run Index')
    plt.ylabel('Voltage dependent signal')
    plt.show()
    #Plot residuals to the fit, show that they average to 0
    plt.plot(leftover - pred)
    plt.plot(signal.savgol_filter(leftover-pred, size, 1))
    plt.xlabel('Run Index')
    plt.ylabel('Fit Residual')
    plt.show()
    #Plot the std. dev. of the resisduals. Note that it seems about right, 
    #a 2mHz or so uncertainty for most of the data.
    plt.plot(np.sqrt(signal.savgol_filter((leftover - pred)**2, size, 1)))
    plt.xlabel('Run Index')
    plt.ylabel('Fit std. dev.')
    plt.show()
    
    param_noB = np.zeros(len(poptV))
    param_noB[1:NEO[-2]] = poptV[1:NEO[-2]]
    param_B = np.zeros(len(poptV))
    param_B[0] = poptV[0]
    param_B[NEO[-2]:] = poptV[NEO[-2]:]
    print('B parameters')
    print(param_B)
    OnlyBData = leftover - wrapEO(EOData, *param_noB)
    Bpred = wrapEO(EOData, *param_B)
    
    index = 0
    startIndex = 0
    runInds = np.zeros(0)
    runBs = np.zeros(0)
    runShifts = np.zeros(0)
    runDevs = np.zeros(0)
    smData = np.zeros(0)
    sz = 21
    for j in range(0, len(Bs)-1):
        if(Bs[j] != Bs[j+1] or j ==len(Bs)-2):
            subData = OnlyBData[startIndex:j]
            smData = np.append(smData, signal.savgol_filter(subData, sz, 1))
            runBs = np.append(runBs, Bs[j])
            runShifts = np.append(runShifts, np.mean(subData))
            runDevs = np.append(runDevs, np.std(subData)/np.sqrt(len(subData)))
            runInds = np.append(runInds, index)
            startIndex = j+1
            index += 1
    
    plt.plot(OnlyBData)
    plt.plot(smData)
    plt.plot(Bpred)
    plt.show()
    
    runShifts *= 1000
    runDevs *= 1000
    plt.errorbar(runBs, runShifts, yerr = runDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    xs = np.linspace(min(runBs), max(runBs))
    ys = np.ones(len(xs))*poptV[0]
    for i in range(NEO[-2], len(poptV)):
        ys += poptV[i]*np.power(xs, i - NEO[-2] + 1)
    ys *= 1000
    plt.plot(xs, ys)
    plt.ylabel('Shift in f_{0} (mHz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
    preds = np.ones(len(runBs))*poptV[0]
    for i in range(NEO[-2], len(poptV)):
        preds += poptV[i]*np.power(runBs, i - NEO[-2] + 1)
    preds *= 1000
    plt.errorbar(runBs, runShifts - preds, yerr = runDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(runBs, np.zeros(len(runBs)))
    plt.ylabel('Fit Residual (mHz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
    plt.errorbar(runInds, runShifts - preds, yerr = runDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(runInds, np.zeros(len(runInds)))
    plt.ylabel('Fit Residual (mHz)')
    plt.xlabel('Run Index')
    plt.show()
    
    eYs = np.ones(len(xs))*poptV[0]
    ErunShifts = np.copy(runShifts)
    for i in range(NEO[-2], len(poptV)):
        if((i - NEO[-2] + 1)%2 == 1):
            ErunShifts -= 1000*poptV[i]*np.power(runBs, i - NEO[-2] + 1)
        else:
            eYs += poptV[i]*np.power(xs, i - NEO[-2] + 1)
    eYs = 1000*eYs
    plt.errorbar(runBs, ErunShifts, yerr = runDevs, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(xs, eYs)
    plt.ylabel('Even Component of Fit (mHz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
    allBs = np.zeros(0)
    avgVal = np.zeros(0)
    weight = np.zeros(0)
    for i in range(0, len(runBs)):
        b=np.where(allBs == runBs[i])[0]
        if(len(b) == 0):
            allBs = np.append(allBs, runBs[i])
            avgVal = np.append(avgVal, runShifts[i]/(runDevs[i]**2))
            weight = np.append(weight, 1/(runDevs[i]**2))
        else:
            avgVal[b] += runShifts[i]/(runDevs[i]**2)
            weight[b] += 1/(runDevs[i]**2)
    avgVal = avgVal/weight
    sigm = 1/np.sqrt(weight)
    
    plt.errorbar(allBs, avgVal, yerr = sigm, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.ylabel('Data (mHz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
    eBs = np.zeros(0)
    eaV = np.zeros(0)
    eVar = np.zeros(0)
    
    for i in range(0, len(allBs)):
        b=np.where(eBs == np.abs(allBs[i]))[0]
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
    pE, pcovE = curve_fit(ffE, eBs, eaV, p0 = [1.1, 0], sigma = eSig)
    EFs = np.linspace(0, max(eBs), 100)
    FE = ffE(EFs, pE[0], pE[1])
    
    print('Even Fit Parameters')
    print(pE)
    print('Even Fit Sig')
    print(np.sqrt(np.diag(pcovE)))
    print('Even Fit Parameters in Std Dev')
    print(pE/np.sqrt(np.diag(pcovE)))
    plt.errorbar(eBs, eaV, yerr = eSig, marker = 'o', linewidth = 0, elinewidth=1, capsize=2)
    plt.plot(EFs, FE)
    plt.ylabel('Even Data (mHz)')
    plt.xlabel('Magnetic Field (T)')
    plt.show()
    
if __name__=="__main__":
    main() 
#popt, pcov = curve_fit(lambda RunData, *p0: wrapper(RunData, Ns *p0), RunData, fs, p0=p0)