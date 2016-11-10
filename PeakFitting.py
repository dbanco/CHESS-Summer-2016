"""
Created on Mon Feb 23 06:34:55 2015
Modified on Sun Apr 12, 2015

@author: abeaudoi, Chris Budrow

Peak fitting routines are adapted from the fitting example given in:

    http://mesa.ac.nz/?page_id=2195
    
Gaussian and Pseudo-Voight functions are addapted from examples in APEXRD

"""
import numpy as np
import numpy.linalg as la
from scipy.optimize import leastsq           # Levenberg-Marquadt Algorithm #
from scipy.interpolate import interp1d

####################################################################################
####################################################################################

def lorentzian(x, param):       # two sided Lorentz function

    peakCtr, fwhmL, fwhmR, amp = param

    xL       = x[x<=peakCtr]  
    NLL      = fwhmL**2
    DLL      = (xL-peakCtr)**2 + fwhmL**2
    LL       = amp*(NLL/DLL)
    
    xR       = x[x>peakCtr]
    NRL      = fwhmR**2
    DRL      = (xR-peakCtr)**2 + fwhmR**2
    RL       = amp*(NRL/DRL)
    
    return np.hstack([LL,RL])
    
def residualsL(param, x, y):
    return y - lorentzian(x, param)     
   
####################################################################################
    
def gaussian(x, param):         # two sided gaussian function

    peakCtr, fwhmL, fwhmR, amp = param
    
    xL       = x[x<=peakCtr]  
    NLG      = (xL-peakCtr)**2
    DLG      = 2*(fwhmL)**2
    LG       = amp*np.exp(-NLG/DLG)
    
    xR       = x[x>peakCtr]
    NRG      = (xR-peakCtr)**2
    DRG      = 2*(fwhmR)**2
    RG       = amp*np.exp(-NRG/DRG)
    
    return np.hstack([LG,RG])

def residualsG(param, x, y):
    return y - gaussian(x, param)

####################################################################################

def pseudovoigt(x, param, n):    # two sided Pseudo-Voight function 

    peakCtr, fwhmL, fwhmR, amp = param
    
    xL       = x[x<=peakCtr]  
    
    gL       = fwhmL/2
    NLL      = gL**2
    DLL      = (xL-peakCtr)**2 + gL**2
    LL       = amp*(NLL/DLL)
    
    sL       = fwhmL/(2*np.sqrt(2*np.log(2)))
    NLG      = -(xL-peakCtr)**2
    DLG      = 2*(sL)**2
    LG       = amp*np.exp(NLG/DLG)
    
    xR       = x[x>peakCtr]
    
    gR       = fwhmR/2
    NRL      = gR**2
    DRL      = (xR-peakCtr)**2 + gR**2
    RL       = amp*(NRL/DRL)
    
    sR       = fwhmR/(2*np.sqrt(2*np.log(2)))
    NRG      = -(xR-peakCtr)**2
    DRG      = 2*(sR)**2
    RG       = amp*np.exp(NRG/DRG)
    
    CombL    = np.hstack((LL,RL))
    CombG    = np.hstack((LG,RG))
    Comb     = n*CombL + (1-n)*CombG
    
    return Comb

def residualsPV(param, x, y, n):
    return y - pseudovoigt(x, param, n)

####################################################################################
####################################################################################

def get_peak_fit_indices(peak, ctr=0.5, lo=0.2, hi=0.8):
    """ function determines indices required for PeakFitting function
    
    inputs:
    peak              : 1d array of peak
    ctr, lo, hi       : location of peak center, left cutoff, and right cutoff as a ratio of the length of the peak vector
    
    outputs:
    peakCtr, loCut, hiCut : indices of peak vector corresponding to peak center, left cutoff, and right cutoff """
    
    peakCtr = int(round(len(peak)*ctr))
    loCut   = int(round(len(peak)*lo))
    hiCut   = int(round(len(peak)*hi))
    
    return peakCtr, loCut, hiCut


def fitPeak(x, y, peakCtr0, fwhm0=10, amp0=3000, FitType='Gaussian', n=1):  
    """
    Gaussian will be the default fit unless another is specified.
    For the Pseudo-Voight fit n=1 will be default unless another is specified
    n is a mixing parameter, it is not necessary for the Lorentzian or Gaussian fits
    """
    param0 = [peakCtr0, fwhm0, fwhm0, amp0]
    
    if FitType == 'Lorentzian':
        param_opt = leastsq(residualsL,  param0, args=(x, y),    full_output=1)[0]
        fit       = lorentzian(x, param_opt)
        err       = la.norm( residualsL(param_opt, x, y), 2.0) / la.norm(y, 2.0)
        
    elif FitType == 'Gaussian':
        param_opt = leastsq(residualsG,  param0, args=(x, y),    full_output=1)[0]
        fit       = gaussian(x, param_opt)
        err       = la.norm( residualsG(param_opt, x, y), 2.0) / la.norm(y, 2.0)
        
    elif FitType == 'PseudoVoigt':
        param_opt = leastsq(residualsPV, param0, args=(x, y, n), full_output=1)[0]
        fit       = pseudovoigt(x, param_opt, n)
        err       = la.norm( residualsPV(param_opt, x, y, n), 2.0) / la.norm(y, 2.0)

    return fit, param_opt, err
    
    
def RemoveBackground(x, y, loCut, hiCut):
    
    x_bg        = np.concatenate([ x[x<loCut], x[x>hiCut] ])
    y_bg        = np.concatenate([ y[x<loCut], y[x>hiCut] ])
    
    #f           = interp1d(x_bg, y_bg)
    coeff       = np.polyfit(x_bg, y_bg, 1.0)
    background  = np.polyval(coeff, x)
    
    yClean      =  y - background
    
    return yClean, background