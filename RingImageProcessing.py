"""
Created on Sat Jun 11 16:08:40 2016

@author: Kenny Swartz, Dan Banco
"""
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

"""
#### User Inputs (edge of circle coordinates) ####
x  = np.array([ 5.949, 5.699, 4.599, 3.25, 5.299 ])
y  = np.array([-2.0  , 0.0  , 2.0  , 3.0,  1.0   ])
"""

def fit_circle_nonlin_lstsq(x, y, plot_flag=0):

    def residual(params, x, y):
        xc, yc, r = params
        return (x-xc)**2 + (y-yc)**2 - r**2
    
    def Jacobian(params, x, y):           
        xc, yc, r = params
        J         = np.zeros((x.shape[0],params.shape[0]))
        J[:,0]    = -2*(x-xc)
        J[:,1]    = -2*(y-yc)
        J[:,2]    = -2*r
        return J
    
    # nonlinear least squares fit
    params0 = np.array([x[0], y[0], 5.0])  # use first data point as initial guess
    params  = leastsq(residual, params0, args=(x,y), Dfun=Jacobian)[0]
    xc, yc, r = params    
    print('xc = '+'%11.8f'%xc, 'zc = '+'%11.8f'%yc, 'r = '+'%11.8f'%r)
   
    if(plot_flag):
        # plotting
        t         = np.linspace(0,2*np.pi,num=10000)
        xx        = xc + r*np.cos(t)
        yy        = yc + r*np.sin(t)
        plt.close('all')
        plt.plot(x,  y,  'ok', ms=10)         # input data points
        plt.plot(xx, yy, '-b', lw=2)          # nonlinear fit
        plt.plot(xc, yc, 'or', ms=10)         # nonlinear fit center
        plt.xlim([np.min(xx),np.max(xx)])
        plt.ylim([np.min(yy),np.max(yy)])
        plt.axes().set_aspect('equal')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.show()
    
    return xc, yc, r
    
    
""" 
Thresholds image with threshold set to mean + n*standard_deviation

Thresholding with n = 3 is good for isolating points on a ring to which to fit 
circle
"""
def n_std_threshold(img,n):
    """
    inputs: img-           image
            n-             constant (threshold = mu + n*sigma)
            
    outputs: thresh_img-   binary image
    """
    mu = np.mean(img.flatten())
    sigma = np.std(img.flatten())
    
    thresh_low = mu + n*sigma

    return img > thresh_low
    
""" 
Creates binary image by thresholding pixel distance from given center
"""
def distance_threshold(img,radius_low,radius_high,center=0):
    bin_img = np.zeros(img.shape)
    n,m = img.shape
    if(center == 0):
        center = [n/2.0,m/2.0] 
        
    # limit to single ring
    for i in range(n):
        for j in range(m):
            pos = (i-center[0])**2 + (j-center[1])**2
            if ((pos > (radius_low)**2) & (pos < (radius_high)**2)):
                bin_img[i,j] = 1                         
    
    return bin_img

"""
Gets coordinates and values of img from true values in binary image
"""
def get_points(img,bin_img):
    x,y = np.nonzero(bin_img)
    f = img[x,y]
    return x,y,f

    