"""
Created on Sat Jun 11 16:08:40 2016

@author: Kenny Swartz, Dan Banco
"""
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
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


def get_points(img,bin_img):
    """
    Gets coordinates and values of img from true values in binary image
    inputs
            img an image
            bin_img a binary image
            
    outputs
            x    x coordinate of pixels
            y    y coordinate of pixels
            f    value of pixels
    """
    x,y = np.nonzero(bin_img)
    f = img[x,y]
    return x,y,f


def radial_projection(img,center,r,num_r,theta,r_in,r_out,):
    """
    Interpolates along a line in the radial direction
    
    inputs
            img     image
            center  center of the rings in the image
            r       radius of ring of interest
            num_r   number of points to sample along line
            theta   angle of line
            r_in    inside radius containing ring
            r_out   outside radius containing ring]
            
    outputs
            r_project  image values at points along line
            r_domain   domain over which image values are defined
    """
    n,m = img.shape
    if(center==0): center = [round(n/2.0),round(m/2.0)]  

    r_domain  =   np.linspace(r_in,r_out,num_r)
    r_project = np.zeros(r_domain.shape)
    
    for ridx in range(len(r_domain)):
        # identify surrounding four points
        x = r_domain[ridx]*np.cos(theta) + center[0]
        y = r_domain[ridx]*np.sin(theta) + center[1]     
        x1 = np.floor( x )
        x2 = np.ceil(  x )
        y1 = np.floor( y )
        y2 = np.ceil(  y )
  
        # make sure we are in image
        if( (x1 < n) & (x2 < n) & (x1 > 0) & (x2 > 0) &
            (y1 < m) & (y2 < m) & (y1 > 0) & (y2 > 0) ):
            if(x2-x1 == 0 & y2-y1 == 0):
                r_project[ridx] = img[x1,y1]
            elif(x2-x1 == 0):
                r_project[ridx] = img[x1,y1] + \
                                (img[x2,y2]-img[x1,y1])*(y-y1)/(y2-y1)
            elif(y2-y1 == 0):
                r_project[ridx] = img[x1,y1] + \
                                (img[x2,y2]-img[x1,y1])*(x-x1)/(x2-x1)
            else:
                
                # interpolate
                a = np.matrix([x2-x,x-x1])
                Q = np.matrix([[img[x1,y1],img[x1,y2]],
                              [img[x2,y1],img[x2,y2]]])      
                b = np.matrix([[y2-y],[y-y1]])
                r_project[ridx] = np.dot(np.dot(a,Q),b)/((x2-x1)*(y2-y1))

    return r_project, r_domain

def azimuthal_projection(img,center,r,theta_1,theta_2,num_theta):
    """
    Interpolates along a line in the radial direction
    
    inputs
            img         image
            center      center of the rings in the image
            r           radius of ring of interest     
            theta_1     inside radius containing ring
            theta_2     outside radius containing ring
            num_theta   number of samples along theta
            
    outputs
            r_project  image values at points along line
            r_domain   domain over which image values are defined
    """
    n,m = img.shape
    if(center==0): center = [round(n/2.0),round(m/2.0)]  

    theta_domain  =   np.linspace(theta_1,theta_2,num_theta)
    theta_project = np.zeros(theta_domain.shape)
    
    for tidx in range(len(theta_domain)):
        # identify surrounding four points
        x = r*np.cos(theta_domain[tidx]) + center[0]
        y = r*np.sin(theta_domain[tidx]) + center[1]     
        x1 = np.floor( x )
        x2 = np.ceil(  x )
        y1 = np.floor( y )
        y2 = np.ceil(  y )
  
        # make sure we are in image
        if( (x1 < n) & (x2 < n) & (x1 > 0) & (x2 > 0) &
            (y1 < m) & (y2 < m) & (y1 > 0) & (y2 > 0) ):
            if((x2-x1 == 0) & (y2-y1 == 0)):
                theta_project[tidx] = img[x1,y1]
            elif((x2-x1) == 0):
                theta_project[tidx] = img[x1,y1] + \
                                (img[x2,y2]-img[x1,y1])*(y-y1)/(y2-y1)
            elif((y2-y1) == 0):
                theta_project[tidx] = img[x1,y1] + \
                                (img[x2,y2]-img[x1,y1])*(x-x1)/(x2-x1)
            else:
                
                # interpolate
                a = np.matrix([x2-x,x-x1])
                Q = np.matrix([[img[x1,y1],img[x1,y2]],
                              [img[x2,y1],img[x2,y2]]])      
                b = np.matrix([[y2-y],[y-y1]])
                theta_project[tidx] = np.dot(np.dot(a,Q),b)/((x2-x1)*(y2-y1))

    return theta_project, theta_domain
    
    
def gaussian_convolution(signal,sigma,C=4):
    """
    inputs
            signal  a one dimensional signal (n,) numpy array
            sigma   a standard deviation
            C       constant to determine length of gaussian kernel
            
    outputs
            filtered signal     convolution of gaussian kernel and input signal
                                where signal length is preserved
    """
    
    M = int(round(C*sigma + 1))
    
    gaussian = np.zeros(2*M+1)
    amp = 1/(np.sqrt(2*np.pi)*sigma)
    const = 1/(2*sigma)
    
    for m in range(-M,M+1):
        gaussian[m+M] = np.exp(-const*m**2)
    
    G = amp*gaussian
    
    return np.convolve(signal,G,mode='same')
    
def find_scale_space_maxima(signal,sigma,C=4,octaves=4):
    """
    inputs
            signal  a one dimensional signal (n,) numpy array
            sigma   a standard deviation
            C       constant to determine length of gaussian kernel
            octaves number of scales over which to find maxima
            
    outputs
            maxima  list of arrays containing locations of maxima at each scale
    """
    maxima = []
    maxima.append(argrelmax(signal))
    plt.figure(0)
    plt.plot(signal)
    for i in range(octaves):
       filt_signal = gaussian_convolution(signal,2*(i+1)*sigma,C)
       maxima.append(argrelmax(filt_signal)[0])
#       plt.figure(i+1)
       plt.plot(filt_signal)
    return maxima
        
def do_peak_fit(data,param,plot_flag=0):
    amp_est                 = np.max(data)
    fwhm_est                = len(data)/2.0
    fit_domain              = np.arange(len(data))
    peakCtr, loCut, hiCut   = DA.get_peak_fit_indices(data)
    data_rm_back, back       = peak.RemoveBackground(fit_domain,data,loCut,hiCut)
    _, param_opt, err       = peak.fitPeak(fit_domain, data_rm_back, peakCtr, 
                                       fwhm=fwhm_est, amp=amp_est,
                                       FitType='Gaussian', n=1)
                                       
    mu, sigmaL, sigmaR, amp = param_opt
    
    if(plot_flag):
        plt.close(1)
        plt.figure(1)
        plt.plot(r_domain,data_rm_back,'o-b') 
        x_fit = np.linspace(fit_domain[0],fit_domain[-1],500)
        r_fit = np.linspace(r_domain[0],r_domain[-1],500)
        plt.plot(r_fit,peak.gaussian(x_fit,param_opt),'-r')   
        title = str(i) + '_err_' + str(err) + '_param_' + str(param)
        plt.title(title)
        plt.savefig(os.path.join('plots',title + '.png'))

    
    return mu, ((sigmaL+sigmaR)/2)**2, amp, err
    
    