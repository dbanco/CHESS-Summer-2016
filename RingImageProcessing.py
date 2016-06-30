"""
Created on Sat Jun 11 16:08:40 2016

@author: Kenny Swartz, Dan Banco
"""
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import DataAnalysis as DA
import PeakFitting as peak
import os


def fit_circle_nonlin_lstsq(x, y, plot_flag=0):
    """
    Fits circle to data points
    inputs
            x           x-coordinates of points to fit
            y           y-coordinates of points to fit
            plot_flag   plots circle of true
            
    outputs
            xc          x-coordinate of center of circle
            yc          y-coordinate of center of circle
            r           radius of circle
    """

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
    r_guess = np.sqrt(x[0]**2 + y[0]**2)/2
    params0 = np.array([x[0], y[0], r_guess])  # use first data point as initial guess
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

def fit_ellipse_lstsq(x,y):
    """
    Fits ellipse to data points
    inputs
            x           x-coordinates of points to fit
            y           y-coordinates of points to fit
            plot_flag   plots circle of true
            
    outputs
            a           x-axis radius parameter
            b           y-axis radius parameter
            orient      orientation angle of the ellipse
            X0_in       x-coordinate of center of ellipse
            Y0_in       y-coordinate of center of ellipse
            X0          x-coordinate of center of ellipse before rotation
            Y0          y-coordinate of center of ellipse before rotation
    """
    orientation_tol = 1e-3
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x1 = x - mean_x
    y1 = y - mean_y
    
    X = np.array([x1**2, x1*y1, y1**2, x1, y1 ])
    C = np.sum(X,axis=1)
    A = np.dot(X,np.transpose(X))
    B = np.linalg.solve(A,C)
    
    a = B[0]
    b = B[1]
    c = B[2]
    d = B[3]
    e = B[4]
    
    if( min(abs(b/a),abs(b/c)) > orientation_tol ):
        orientation_rad = 0.5*np.arctan(b/(c-a))
        cos_phi = np.cos( orientation_rad )
        sin_phi = np.sin( orientation_rad )
        a = a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2
        b = 0
        c = a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2
        d = d*cos_phi - e*sin_phi
        e = d*sin_phi + e*cos_phi
        
        mean_x = cos_phi*mean_x - sin_phi*mean_y
        mean_y = sin_phi*mean_x + cos_phi*mean_y
    else:
        orientation_rad = 0
        cos_phi = 1
        sin_phi = 0
    test = a*c
    if( test < 0 ):
        status = 'hyperbola found'
        print( status + ': did not identify ellipse')
    elif( test == 0):
        status = 'parabola found'
        print( status + ': did not identify ellipse')
    elif( test > 0):
        status = ''
        if( a<0 ): 
            a = -a
            c = -c
            d = -d
            e = -e
            
        X0 = mean_x - d/2/a
        Y0 = mean_y - e/2/c
        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
        a = np.sqrt(F/a)
        b = np.sqrt(F/c)
#        long_axis = 2*max(a,b)
#        short_axis = 2*min(a,b)
        # Need to apply rotation to center point to get actual center
        X0_in = X0*cos_phi - Y0*sin_phi
        Y0_in = X0*sin_phi + Y0*cos_phi
        
        return [a,b,orientation_rad,X0_in,Y0_in,X0,Y0]
        
    else:
        print('Something went wrong')
        return 0

def fit_ellipse_nonlin_lstsq(x, y, plot_flag=0):
    """
    Fits ellipse to data points
    inputs
            x           x-coordinates of points to fit
            y           y-coordinates of points to fit
            plot_flag   plots circle of true
            
    outputs
            xc          x-coordinate of center of ellipse
            yc          y-coordinate of center of ellipse
            a           x-axis radius parameter
            b           y-axis radius parameter
    """
    def residual(params, x, y):
        xc, yc, a, b = params
        return b**2*(x-xc)**2 + a**2*(y-yc)**2 - (a*b)**2
    
    def Jacobian(params, x, y):           
        xc, yc, a, b = params
        J         = np.zeros((x.shape[0],params.shape[0]))
        J[:,0]    = -2*(x-xc)*b**2
        J[:,1]    = -2*(y-yc)*a**2
        J[:,2]    = -2*a*b**2
        J[:,3]    = -2*b*a**2
        return J
    
    # nonlinear least squares fit
    r_guess = np.sqrt(x[0]**2 + y[0]**2)/2
    params0 = np.array((x[0], y[0],r_guess,r_guess))  # use first data point as initial guess
    params  = leastsq(residual, params0, args=(x,y), Dfun=Jacobian)[0]
    xc, yc, a, b = params    
    print('xc = '+'%11.8f'%xc, 'zc = '+'%11.8f'%yc, 
          'a = '+'%11.8f'%a,   'b = '+'%11.8f'%b)
   
    if(plot_flag):
        # plotting
        t         = np.linspace(0,2*np.pi,num=10000)
        xx        = xc + a*np.cos(t)
        yy        = yc + b*np.sin(t)
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
    
    return xc, yc, a, b   
    

def n_std_threshold(img,n):
    """
    Thresholds image with threshold set to mean + n*standard_deviation
    inputs: img           image
            n             constant (threshold = mu + n*sigma)
            
    outputs: thresh_img   binary image    
    """
    mu = np.mean(img.flatten())
    sigma = np.std(img.flatten())
    
    thresh_low = mu + n*sigma

    return img > thresh_low
    

def distance_threshold(img,radius_low,radius_high,center=0):
    """ 
    Creates binary image by thresholding pixel distance from given center
    inputs
            img             image to be thresholded by pixel distance
            radius_low      distance above which pixels are kept 
            radius_high     distance below which pixels are kept
            center          [x,y] position from which distance is measured
    outputs
            bin_img         binary image where pixels equal to 1 satisfy
                            radius_low < distance < radius_high
    """
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
            img         an image
            bin_img     a binary image
            
    outputs
            x           x coordinate of pixels
            y           y coordinate of pixels
            f           value of pixels
    """
    y,x = np.nonzero(bin_img)
    f = img[y,x]
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
            if( ((x2-x1) == 0) & ((y2-y1) == 0) ):
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
    
def line_normal_to_curve(x0,y0,distance):
    """
    Finds two points along line perpendicular to ellipse a distance from the
    point on the ellipse at angle theta about the center. Returns slope and 
    intercept of the line as well
    inputs
        x0          x-coordinates of three points
        y0          y-coordinates of three points
        distance    distance from point to find two points along line
    outputs
        x           coordinates of two points
        y           coordinates of two points
        slope       slope of line normal to ellipse at x0,y0
        intercept   y-intercpet of the line normal to ellipse at x0,y0
    """
   
    if( np.abs(y0[1]) > np.abs(x0[1])):
        # Slope of line normal to ellipse at that point
        b_diff = (y0[1]-y0[0])/(x0[1]-x0[0])
        f_diff = (y0[2]-y0[1])/(x0[2]-x0[1])
        slope  = -1/(0.5*b_diff + 0.5*f_diff)
        intercept = -slope*x0[1] + y0[1]
        
        # Polynomial coefficients
        p = [(1 + slope**2), 
             (2*slope*(intercept - y0[1]) - 2*x0[1]), 
             ((intercept - y0[1])**2 + x0[1]**2 - distance**2) ]  
        
        x = np.real(np.roots(p))
        y = np.real(slope*x + intercept)
    else:
        # Slope of line normal to ellipse at that point
        b_diff = (x0[1]-x0[0])/(y0[1]-y0[0])
        f_diff = (x0[2]-x0[1])/(y0[2]-y0[1])
        slope  = -1/(0.5*b_diff + 0.5*f_diff)
        intercept = -slope*y0[1] + x0[1]
        
        # Polynomial coefficients
        p = [(1 + slope**2), 
             (2*slope*(intercept - x0[1]) - 2*y0[1]), 
             ((intercept - x0[1])**2 + y0[1]**2 - distance**2) ] 
       
        y = np.roots(p)
        x = slope*y + intercept
    return x, y
    

def radial_projection_ellipse(img,xc,yc,a,b,orient,num_r,theta,distance):
    """
    Interpolates along a line in the radial direction
    
    inputs
            img     image
            center  center of the rings in the image
            a,b     major/minor axis of ring of interest
            orient  angle at which ellipse is oriented
            num_r   number of points to sample along line
            theta   angle of line (give three angles to identify center)
            d_in    inside window containing ring
            d_out   outside window containing ring]
            
    outputs
            r_project  image values at points along line
            r_domain   domain over which image values are defined
    """
    # Points centered at angle theta[1]
 
    xi = a*np.cos(theta) + xc
    yi = b*np.sin(theta) + yc
    x0 =  xi*np.cos(orient) + yi*np.sin(orient)
    y0 = -xi*np.sin(orient) + yi*np.cos(orient)

    # Get endpoints of line perpendicular to ellipse at x0, y0
    x_l, y_l = line_normal_to_curve(x0,y0,distance)

    sort = (x_l-xc)**2 + (y_l-yc)**2
    sort_ind = [np.argmin(sort), np.argmax(sort)]
    x_line = x_l[sort_ind]
    y_line = y_l[sort_ind]                       

    if( x_line[1] == x_line[0]):
        x_domain = x_line[0]*np.ones(2*distance)
        y_domain = np.linspace(y_line[0],y_line[1],2*distance)
    elif( y_line[1] == y_line[0]):
        y_domain = y_line[0]*np.ones(2*distance)
        x_domain = np.linspace(x_line[0],x_line[1],2*distance)
    else:
        x_domain = np.linspace(x_line[0],x_line[1],2*distance)
        y_domain = np.linspace(y_line[0],y_line[1],2*distance)
        
    f = np.zeros(x_domain.shape)
     
    for idx in range(len(x_domain)):
        
        # identify surrounding four points
        x = x_domain[idx]
        y = y_domain[idx]    
        x1 = np.floor( x )
        x2 = np.ceil(  x )
        y1 = np.floor( y )
        y2 = np.ceil(  y )
  
        # make sure we are in image
        n,m = img.shape
        if( (x1 < n) & (x2 < n) & (x1 > 0) & (x2 > 0) &
            (y1 < m) & (y2 < m) & (y1 > 0) & (y2 > 0) ):
            if( ((x2-x1) == 0) & ((y2-y1) == 0) ):
                f[idx] = img[y1,x1]
            elif(x2-x1 == 0):
                f[idx] = img[y1,x1] + (img[y2,x2]-img[y1,x1])*(y-y1)/(y2-y1)
            elif(y2-y1 == 0):
                f[idx] = img[y1,x1] + (img[y2,x2]-img[y1,x1])*(x-x1)/(x2-x1)
            else: 
                # interpolate
                a = np.matrix([x2-x,x-x1])
                Q = np.matrix([[img[y1,x1],img[y1,x2]],
                              [img[y2,x1],img[y2,x2]]])      
                b = np.matrix([[y2-y],[y-y1]])
                f[idx] = np.dot(np.dot(a,Q),b)/((x2-x1)*(y2-y1))

    return f, x_domain, y_domain

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
        
def do_peak_fit(y,edgeCut,index=0,param=0,plot_flag=0):
    """
    Fits peak with two-sided Gaussian. Assumes peak is in the center and that
    edgeCut tells which portion of y to treat as background. interpolates
    between the points at the cutoff to subtract background before fitting
    a peak.
    
    inputs
            y           a 1D vector containing a peak
            edgeCut     fraction at edge of y to treat as background
            index       used to identify plot, appears in title
            param       used to identify plot, appears in title
            plot_flag   produces and saves plot to "plots" directory
    outputs
            mu          mean of the fitted two-sided Gaussian
            variance    total variance of the two-sided Gaussian
            amp         amplitude of the two-sided Gaussian
            err         ||fit - y||/||y||
    """
    amp_est                 = np.max(y)
    fwhm_est                = len(y)/2.0
    fit_domain              = np.arange(len(y))
    peakCtr, loCut, hiCut   = DA.get_peak_fit_indices(y,lo=edgeCut,hi=1-edgeCut)
    data_rm_back, back      = peak.RemoveBackground(fit_domain,y,loCut,hiCut)
    _, param_opt, err       = peak.fitPeak(fit_domain, data_rm_back, peakCtr, 
                                       fwhm=fwhm_est, amp=amp_est,
                                       FitType='Gaussian', n=1)
                                       
    mu, sigmaL, sigmaR, amp = param_opt
    
    if(plot_flag):
        plt.close(1)
        plt.figure(1)
        plt.plot(y,'o-b') 
        x_fit = np.linspace(fit_domain[0],fit_domain[-1],500)
        plt.plot(x_fit,peak.gaussian(x_fit,param_opt)+np.mean(back),'-r')   
        title = str(index) + '_err_' + str(err) + '_param_' + str(param)
        plt.title(title)
        plt.savefig(os.path.join('plots',title + '.png'))

    
    return mu, ((sigmaL+sigmaR)/2)**2, amp, err
    
def crop_around_max(x):
    """
    Finds max of vector and crops segment so max is at center
    inputs
            x           a 1D numpy array
    outputs
            x_cropped   cropped 1D numpy array
            indices     indices such that x_cropped = x[indices]
            center      index where center is located in x
    """
    out = np.argmax(x)
    center = round(out)
    if( center < (len(x)-1-center) ):
    x_crop = x[0:(2*center+2)]
    indices = range(0,int(2*center+2))
    elif( center > (len(x)-1-center) ):
    x_crop = x[(2*center-(len(x)-1)):len(x)]
    indices = range(int(2*center-(len(x)-1)),len(x))
    else:
    x_crop = np.copy(x)
    indices = range(0,len(x))
     
    return x_crop, indices, center
     
     
def ring_fit(img,r_est,dr,pf1=False):
    """  
    Fits an ellipse to a ring of an xray diffraction image 
    inputs
        img-    image of xray diffraction rings    
        r_est-  estimate of radius of ring of interest in pixels
        dr-     space around ring on either side in pixels
    outputs
        a-      x-axis radius parameter of fitted ellipse     
        b-      z-axis radius parameter of fitted ellipse
        xc-     x-coordinate of center of ellipse
        zc-     z-coordinate of center of ellipse
        rot-    orientation angle of ellipse
        
        
    EX:
    # Estimate radius of ring
    plt.close('all')
    plt.figure(1)
    plt.imshow(img,cmap='jet',vmin=0,vmax=200)
    xc = 1026
    zc = 1018
    radius_test = 606
    plt.plot([xc,xc+radius_test],[zc,zc],'o-w')
    
    # Fit ring
    radius = 370
    dr     = 30
    a,b,xc,zc,rot = RingIP.ring_fit(img,radius,dr)

    """  
    if(pf1):
        plt.close(pf1)
        plt.figure(pf1)
        plt.imshow(img,cmap='jet',vmin=0,vmax=200)
    
    r_in = r_est - dr
    r_out = r_est + dr

    # Keeps only pixels near ring
    bin_img = distance_threshold(img,r_in,r_out)    
    
    # Thresholds remainder of image with threshold 
    # 2 standard deviations above the mean
    thresh_img = n_std_threshold(img*bin_img,1)

    # Converts identified pixels to a set of x,z,f data points
    x,z,f = get_points(img,bin_img*thresh_img)

    # Fit ellipse to found points (ignores intensity)
    out = fit_ellipse_lstsq(x,z)
    a = out[0]
    b = out[1]
    rot = out[2]
    xc = out[3]
    zc = out[4]

    if(pf1):
        plt.figure(pf1)
        plt.plot(x,z,'wx')
        xx,zz = gen_ellipse(a,b,xc,zc,rot)
        plt.plot(xx,zz,'w-')
        plt.xlim([np.min(xx)-10, np.max(xx)+10])
        plt.ylim([np.min(zz)-10, np.max(zz)+10])

    return a, b, xc, zc, rot     
     
def two_stage_ring_fit(img,edgeCut,r_est,dr,pf1=False,pf2=False):
    """  
    Fits an ellipse to a ring of an xray diffraction image 
    inputs
        img-    image of xray diffraction rings    
        edgeCut-edge cutoff for peak fitting (fraction of vector length)
        r_est-  estimate of radius of ring of interest in pixels
        dr-     space around ring on either side in pixels
        pf1-    plots output of 2nd stage in figure(pf1)
        pf2-    plots output of 1st stage in figure(pf2)
    outputs
        a-      x-axis radius parameter of fitted ellipse     
        b-      z-axis radius parameter of fitted ellipse
        xc-     x-coordinate of center of ellipse
        zc-     z-coordinate of center of ellipse
        rot-    orientation angle of ellipse
        
        
    EX:
    # Estimate radius of ring
    plt.close('all')
    plt.figure(1)
    plt.imshow(img,cmap='jet',vmin=0,vmax=200)
    xc = 1026
    zc = 1018
    radius_test = 606
    plt.plot([xc,xc+radius_test],[zc,zc],'o-w')
    
    # Fit ring
    edgeCut = 0.2
    radius = 370
    dr     = 30
    a,b,xc,zc,rot = RingIP.two_stage_ring_fit(img,edgeCut,radius,dr)

    """  
    if(pf1):
        plt.close(pf1)
        plt.figure(pf1)
        plt.imshow(img,cmap='jet',vmin=0,vmax=200)
        
    if(pf2):
        plt.close(pf2)
        plt.figure(pf2)
        plt.imshow(img,cmap='jet',vmin=0,vmax=200)
    
    r_in = r_est - dr
    r_out = r_est + dr

    # Keeps only pixels near ring
    bin_img = distance_threshold(img,r_in,r_out)    
    
    # Thresholds remainder of image with threshold 
    # 2 standard deviations above the mean
    thresh_img = n_std_threshold(img*bin_img,1)

    # Converts identified pixels to a set of x,z,f data points
    x,z,f = get_points(img,bin_img*thresh_img)

    # Fit ellipse to found points (ignores intensity)
    out1 = fit_ellipse_lstsq(x,z)
    a1 = out1[0]
    b1 = out1[1]
    rot1 = out1[2]
    xc1 = out1[3]
    zc1 = out1[4]

    if(pf2):
        plt.figure(pf2)
        plt.plot(x,z,'wx')
        xx,zz = gen_ellipse(a1,b1,xc1,zc1,rot1)
        plt.plot(xx,zz,'w-')
        plt.xlim([np.min(xx)-10, np.max(xx)+10])
        plt.ylim([np.min(zz)-10, np.max(zz)+10])

    # Identify approximate number of significant samples to interpolate
    num_r = round(r_out-r_in)
    # Identify smallest possible angular interval dtheta
    r1 = np.mean(np.array([a1,b1]))
    dtheta = np.abs(np.pi/4 - np.arctan((r1+1)/(r1-1)))
    # Might include integer number of steps as an input parameter
    step = 20*dtheta
    theta = np.arange(dtheta,2*np.pi + step,step)
    
    mu = np.zeros(theta.shape)
    err = np.zeros(theta.shape)
    x2 = np.array([])
    z2 = np.array([])
    for i in range(len(theta)-1):
        # Get points along line normal to ellipse at angle theta[i]
        f, x_dom, y_dom = radial_projection_ellipse(img,xc1,zc1, 
                                                    a1,b1,rot1,num_r,
                                                    [theta[i-1],theta[i],
                                                    theta[i+1]],dr)
        # Crop segment so peak is at the center                                                    
        f_crop, crop_ind, _ = crop_around_max(f)
        
        x_dom_crop = x_dom[crop_ind]
        y_dom_crop = y_dom[crop_ind]
        
        # Fit peaks in cropped segments if cropped segment is long enough
        if (len(f_crop) > 0.2*len(f)):
            mu[i],_,_,err[i] = do_peak_fit(f_crop,edgeCut)
        else:
            mu[i] = 0
            err[i] = 1
        
        # Keep only peak fits within error threshold
        if(err[i]<0.2):
            xmid = x_dom_crop[int(round(mu[i]))]
            ymid = y_dom_crop[int(round(mu[i]))]
            x2 = np.append(x2,xmid)
            z2 = np.append(z2,ymid)
            
            if(pf1):
                x1p  = x_dom_crop[0]
                x2p  = x_dom_crop[-1]
                y1p  = y_dom_crop[0]
                y2p  = y_dom_crop[-1]
                plt.figure(pf1)
                plt.plot((x1p,xmid,x2p),(y1p,ymid,y2p),'o-w')

    
    # Fit ellipse to the top of the peaks
    out2 = fit_ellipse_lstsq(x2,z2)
    a2 = out2[0]
    b2 = out2[1]
    rot2 = out2[2]
    xc2 = out2[3]
    zc2 = out2[4]

    if(pf1):
        plt.figure(pf1)
        xx,zz = gen_ellipse(a1,b1,xc1,zc1,rot1)
        plt.plot(xx,zz,'w-')
        plt.xlim([np.min(xx)-10, np.max(xx)+10])
        plt.ylim([np.min(zz)-10, np.max(zz)+10])
    return a2, b2, xc2, zc2, rot2

def gen_ellipse(a,b,xc,yc,rot):
    """
    Generates x,y data points along the perimeter of ellipse defined by inputs
    inputs            
            a-      x-axis radius parameter of fitted ellipse     
            b-      z-axis radius parameter of fitted ellipse
            xc-     x-coordinate of center of ellipse
            zc-     z-coordinate of center of ellipse
            rot-    orientation angle of ellipse
    
    outputs
            x       x-coordinates of points on ellipse
            y       y-coordinates of points on ellipse
    """
    t         = np.linspace(0,2*np.pi,num=10000)
    xa        = xc + a*np.cos(t)
    ya        = yc + b*np.sin(t)
    x  =  (xa)*np.cos(rot) + (ya)*np.sin(rot)
    y  = -(xa)*np.sin(rot) + (ya)*np.cos(rot) 
    return x,y