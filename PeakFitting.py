"""
Created on Mon Feb 23 06:34:55 2015
Modified on Sun Apr 12, 2015

@author: abeaudoi, Chris Budrow

Peak fitting routines are adapted from the fitting example given in:

    http://mesa.ac.nz/?page_id=2195
    
Gaussian and Pseudo-Voight functions are addapted from examples in APEXRD

"""
import numpy
import math

from scipy.optimize import leastsq           # Levenberg-Marquadt Algorithm #
#from savitzky_golay import savitzky_golay   # Smoothing algorithm ##
from scipy.interpolate import interp1d

def lorentzian(x,p):       # two sided Lorentz function
    #min_step = x[1] - x[0]
    ind1     = (x<= p[1])  
    x1       = x[ind1]  
    ind2     = (x > p[1])
    x2       = x[ind2]
        
    NLL      = (p[0]**2 )
    DLL      = ( x1 - (p[1]) )**2 + p[0]**2
    NRL      = (p[2]**2 )
    DRL      = ( x2 - (p[1]) )**2 + p[2]**2
    LL       = p[3]*(NLL/DLL)
    RL       = p[3]*(NRL/DRL)
    lin_comb = numpy.hstack((LL,RL))
    
    return lin_comb
   
def gaussian(x,p):         # two sided gaussian function
    ind1     = (x<= p[1]) 
    x1       = x[ind1]  
    ind2     = (x > p[1])
    x2       = x[ind2]
    		
    NLG      = -(x1 - p[1])**2
    DLG      = 2*(p[0])**2
    NRG      = -(x2 - p[1])**2
    DRG      = 2*(p[2])**2
    LG       = p[3]*numpy.exp(NLG/DLG)
    RG       = p[3]*numpy.exp(NRG/DRG)
    lin_comb = numpy.hstack((LG,RG))
    
    return lin_comb

def pseudovoigt(x,p,n):    # two sided Pseudo-Voight function
    ind1     = (x<= p[1])  
    x1       = x[ind1]  
    ind2     = (x > p[1])
    x2       = x[ind2]
    
    gL       = p[0]/2
    gR       = p[2]/2
    sL       = p[0]/(2*math.sqrt(2*math.log(2)))
    sR       = p[2]/(2*math.sqrt(2*math.log(2)))
    
    NLL      = (gL**2 )
    DLL      = ( x1 - (p[1]) )**2 + gL**2
    NRL      = (gR**2 )
    DRL      = ( x2 - (p[1]) )**2 + gR**2
    
    LL       = p[3]*(NLL/DLL)
    RL       = p[3]*(NRL/DRL)
    CombL    = numpy.hstack((LL,RL))
    
    NLG      = -(x1 - p[1])**2
    DLG      = 2*(sL)**2
    NRG      = -(x2 - p[1])**2
    DRG      = 2*(sR)**2
    
    LG       = p[3]*numpy.exp(NLG/DLG)
    RG       = p[3]*numpy.exp(NRG/DRG)
    CombG    = numpy.hstack((LG,RG))
    
    Comb     = n*CombL + (1-n)*CombG
    
    return Comb

def residualsL(p,y,x):
    err = y - lorentzian(x,p)
    return err

def residualsG(p,y,x):
    err = y - gaussian(x,p)
    return err

def residualsPV(p,y,x,n):
    err = y - pseudovoigt(x,p,n)
    return err


def fitPeak(x, y_bg_corr, FWHM, peakCtr, Am, FitType = 'Lorentzian', n = 1):  
    """
    Lorentzian will be the default fit unless another is specified.
    For the Pseudo-Voight fit n=1 will be default unless another is specified
    n is a mixing parameter, it is not necessary for the Lorentzian or Gaussian fits
    """
    p = [FWHM,peakCtr,FWHM, Am]
    if FitType == 'Lorentzian':
        best_parameters = leastsq(residualsL, p, args=(y_bg_corr,x), full_output=1)[0]
        fit             = lorentzian(x,best_parameters)
    elif FitType == 'Gaussian':
        best_parameters = leastsq(residualsG,p,args=(y_bg_corr,x),full_output=1)[0]
        fit             = gaussian(x,best_parameters)
    elif FitType == 'PseudoVoigt':
        best_parameters = leastsq(residualsPV,p,args=(y_bg_corr,x,n),full_output=1)[0]
        fit             = pseudovoigt(x,best_parameters,n)

    return fit, best_parameters
    
# Develop background of a function
def developBackground(x,y,peakCtr,loCut,hiCut): 
    # defining the 'background' part of the spectrum #
    ind_bg_low  = (x > (peakCtr-hiCut)) & (x < (peakCtr-loCut)) 
    ind_bg_high = (x > (peakCtr+loCut)) & (x < (peakCtr+hiCut))
    
    x_bg        = numpy.concatenate((x[ind_bg_low], x[ind_bg_high]))
    y_bg        = numpy.concatenate((y[ind_bg_low], y[ind_bg_high]))
    
    # interpolating the background #
    f           = interp1d(x_bg,y_bg)
    background  = f(x)
    
    return background
    
def RemoveBackground(x,y,peakCtr,loCut,hiCut):
    background = developBackground(x, y, peakCtr, loCut, hiCut)
    yClean     = y - background
    
    return yClean, background