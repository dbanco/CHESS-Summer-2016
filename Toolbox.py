# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 00:39:04 2016

@author: Kenny Swartz
"""
import numpy as np
import scipy.optimize
import scipy.sparse
import matplotlib.pyplot as plt

def inch2mm(inch):
    return 25.4*inch
    
def mm2inch(mm):
    return mm/25.4

def D1_SOCD(n):
    
    D1           = np.diag(-1*np.ones(n-1), k=-1) + np.diag(1*np.ones(n-1), k=1)
    D1[ 0,  :2]  = np.array([-1, 1])      # apply forward differencing on left side
    D1[-1, -2:]  = np.array([-1, 1])      # apply backward differencing on right side 
    
    return D1
    

def D2_SOCD(n):
    
    D2           = np.diag(1*np.ones(n-1), k=-1) + np.diag(-2*np.ones(n), k=0) + np.diag(1*np.ones(n-1), k=1)
    D2[ 0,  :3]  = np.array([1, -2, 1])    # apply forward differencing on left side
    D2[-1, -3:]  = np.array([1, -2, 1])    # apply backward differencing on right side
        
    return D2
    

def fit_circle_nonlin_lstsq(x, y, x0=0, y0=0, r0=1):

    def residual(params, x, y):
        xc, yc, r = params
        return (x-xc)**2 + (y-yc)**2 - r**2

    def Jacobian(params, x, y):           
        xc, yc, r = params
        J         = np.zeros((x.shape[0], params.shape[0]))
        J[:,0]    = -2*(x-xc)
        J[:,1]    = -2*(y-yc)
        J[:,2]    = -2*r
        return J

    # nonlinear least squares fit
    params0 = np.array([x0, y0, r0])
    params  = scipy.optimize.leastsq(residual, params0, args=(x,y), Dfun=Jacobian)[0]
    
    # plotting
    xc, yc, r = params
    t         = np.linspace(0,2*np.pi,num=10000)
    xx        = xc + r*np.cos(t)
    yy        = yc + r*np.sin(t)
    plt.close('all')
    plt.xlabel('x'), plt.xlim([np.min(xx),np.max(xx)])
    plt.ylabel('y'), plt.ylim([np.min(yy),np.max(yy)])
    plt.plot(x,  y,  'ok', ms=10)         # input data points
    plt.plot(xx, yy, '-b', lw=2)          # nonlinear fit
    plt.plot(xc, yc, 'or', ms=10)         # nonlinear fit center
    plt.axes().set_aspect('equal')
