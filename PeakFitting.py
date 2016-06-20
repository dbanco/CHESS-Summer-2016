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
#from savitzky_golay import savitzky_golay   # Smoothing algorithm ##

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
    
    xR       = x[x> peakCtr]
    
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

def residualsL(param, x, y):
    return y - lorentzian(x, param) 

def residualsG(param, x, y):
    return y - gaussian(x, param)

def residualsPV(param, x, y, n):
    return y - pseudovoigt(x, param, n)


def fitPeak(x, y, peakCtr, fwhm=10, amp=3000, FitType='Gaussian', n=1):  
    """
    Gaussian will be the default fit unless another is specified.
    For the Pseudo-Voight fit n=1 will be default unless another is specified
    n is a mixing parameter, it is not necessary for the Lorentzian or Gaussian fits
    """
    param = [peakCtr, fwhm, fwhm, amp]
    
    if FitType == 'Lorentzian':
        param_opt = leastsq(residualsL,  param, args=(x, y),    full_output=1)[0]
        fit       = lorentzian(x, param_opt)
        err       = la.norm( residualsL(param_opt, x, y), 2.0)
        
    elif FitType == 'Gaussian':
        param_opt = leastsq(residualsG,  param, args=(x, y),    full_output=1)[0]
        fit       = gaussian(x, param_opt)
        err       = la.norm( residualsG(param_opt, x, y), 2.0)
        
    elif FitType == 'PseudoVoigt':
        param_opt = leastsq(residualsPV, param, args=(x, y, n), full_output=1)[0]
        fit       = pseudovoigt(x, param_opt, n)
        err       = la.norm( residualsPV(param_opt, x, y, n), 2.0)

    return fit, param_opt, err
    
    
def RemoveBackground(x, y, loCut, hiCut):
    
    x_bg        = np.concatenate([ x[x<loCut], x[x>hiCut] ])
    y_bg        = np.concatenate([ y[x<loCut], y[x>hiCut] ])
    
    f           = interp1d(x_bg, y_bg)
    background  = f(x)
    
    yClean      =  y - background
    
    return yClean, background