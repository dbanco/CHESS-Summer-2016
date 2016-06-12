"""
Created on Sat Jun 11 16:08:40 2016

@author: Kenny Swartz
"""
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


#### User Inputs (edge of circle coordinates) ####
x  = np.array([ 5.949, 5.699, 4.599, 3.25, 5.299 ])
z  = np.array([-2.0  , 0.0  , 2.0  , 3.0,  1.0   ])

   
def residual(params, x, z):
    xc, zc, r = params
    return (x-xc)**2 + (z-zc)**2 - r**2

def Jacobian(params, x, z):           
    xc, zc, r = params
    J         = np.zeros((x.shape[0],params.shape[0]))
    J[:,0]    = -2*(x-xc)
    J[:,1]    = -2*(z-zc)
    J[:,2]    = -2*r
    return J

# nonlinear least squares fit
params0 = np.array([x[0], z[0], 5.0])  # use first data point as initial guess
params  = leastsq(residual, params0, args=(x,z), Dfun=Jacobian)[0]
print('xc = '+'%11.8f'%params[0], 'zc = '+'%11.8f'%params[1], 'r = '+'%11.8f'%params[2])

# plotting
xc, zc, r = params
t         = np.linspace(0,2*np.pi,num=10000)
xx        = xc + r*np.cos(t)
zz        = zc + r*np.sin(t)
plt.close('all')
plt.plot(x,  z,  'ok', ms=10)         # input data points
plt.plot(xx, zz, '-b', lw=2)          # nonlinear fit
plt.plot(xc, zc, 'or', ms=10)         # nonlinear fit center
plt.xlim([np.min(xx),np.max(xx)])
plt.ylim([np.min(zz),np.max(zz)])
plt.axes().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('z')