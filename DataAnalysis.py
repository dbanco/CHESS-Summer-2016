"""
Data Analysis Functions

@author: Kenny Swartz
06/07/2016
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.optimize
import DataReader
import PeakFitting

def total_variation(y, lamb=0):
    
    # second order accurate finite difference approximation
    n            = y.shape[0]            # number of data points
    D2           = np.diag(np.ones(n-1), k=1) + np.diag(-2*np.ones(n), k=0) + np.diag(np.ones(n-1), k=-1)
    D2[ 0,  :3]  = np.array([1,-2,1])    # apply forward differencing on left side
    D2[-1, -3:]  = np.array([1,-2,1])    # apply backward differencing on right side
    
    # optimization 
    def obj(x, y, lamb, D2): 
        return la.norm(x-y, 2.0)**2 + lamb*la.norm(np.dot(D2, x), 1.0)
     
    x0           = np.mean(y)*np.ones(n)  # initial guess
    opt_results  = scipy.optimize.minimize(obj, x0, args=(y, lamb, D2))
    
    return opt_results.x    



def get_indices(peak, ctr=0.5, lo=0.25, hi=0.75):
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


    
def analyze_strip(image, strip_orient, strip_width, pix_rng, peak_path, strip_loc = 0.5, FWHM=10, Am=2000):
    """ function fits a peak to a summed strip of a ge2 detector image
    
    inputs:
    image             : ge2 detector image
    strip_orient      : 'v' for vertical strip, 'h' for horizontal strip
    strip_width       : width of strip that is analyzed to fit peak
    pix_rnge          : range of pixels where the desired peak is located
    peak_directory    : directory where peak fits are saved
    FWHM              : initial guess for full width half max of peak
    Am                : initial guess for peak amplitude
    
    outputs:
    true_center       : location (index) of peak center   """

    if strip_orient == 'v':
        vertical_strip    = image[ :, image.shape[1]//2-strip_width//2 : image.shape[1]//2+strip_width//2 ]
        line              = np.sum(vertical_strip, axis=1)
        
    if strip_orient == 'h':
        center_index      = int(round(image.shape[0]*strip_loc))
        lower_bound       = np.max([center_index-strip_width//2, 0])
        upper_bound       = np.min([image.shape[0], center_index+strip_width//2])
        horizontal_strip  = image[ lower_bound : upper_bound , :]
        line              = np.sum(horizontal_strip, axis=0)
    
    peak                  = line[pix_rng[0]:pix_rng[1]]
    x_index               = np.linspace(1,len(peak),num=len(peak))
    peakCtr, loCut, hiCut = get_indices(peak)
    y_bg_corr, background = PeakFitting.RemoveBackground(x_index, peak, peakCtr, loCut, hiCut)
    fit, best_parameter   = PeakFitting.fitPeak(x_index, y_bg_corr, FWHM, peakCtr, Am, FitType = 'Gaussian')
    fit_center            = best_parameter[1]
    true_center           = fit_center + pix_rng[0]
    std_dev_left          = best_parameter[0]
    std_dev_right         = best_parameter[2]
    """
    plt.close('all')
    plt.plot(x_index, y_bg_corr, 'ok')
    plt.plot(x_index, fit,       '-r')
    plt.savefig(peak_path)
    plt.close('all')
    """
    return true_center, std_dev_left, std_dev_right



def write_scan_diameters(xray_dir, out_dir, peak_dir, first_folder, first_file_num, descrip, num_data_pts, strip_width, orient, lt_peak_rng, rb_peak_rng, FWHM=10, Am=2000):
            
    dark_path             = DataReader.get_ge2_path(xray_dir, first_folder, first_file_num)
    dark_image            = DataReader.ge2_reader(dark_path)[0]
    
    diams                 = np.zeros(num_data_pts)  # ring diameters
     
    for data_pt in range(num_data_pts):
        
        dir_num               = first_folder   + data_pt
        file_num              = first_file_num + data_pt
        
        print('reading image ' + str(dir_num))
        path                  = DataReader.get_ge2_path(xray_dir, dir_num, file_num)
        image                 = DataReader.ge2_reader(path)[0]  # only using first image because of shutter timing error
        image                -= dark_image                       # subtract dark image
        
        peak_path             = peak_dir+'peak_'+str(dir_num)+orient+'.tiff'
        lt_center             = analyze_strip(image, orient, strip_width, lt_peak_rng, peak_path, FWHM, Am)
        rb_center             = analyze_strip(image, orient, strip_width, rb_peak_rng, peak_path, FWHM, Am)
        diams[data_pt]        = rb_center - lt_center

    # write data to a text file
    out_file = open(out_dir+descrip+orient+'.txt', 'w')
    for data_pt in range(num_data_pts):
        out_file.write('%18.14f'%diams[data_pt] + '\n')
    out_file.close()     


    
def find_closest_vic2d(vic2d_data, x, y, out_index, x_ind=5, y_ind=6):
    """ function finds the closest vic2d data point to desired data points described by x and y locations
    
    inputs:
    vic2d_data        : numpy array of vic2d data read in by DataReader.vic2d_reader (number of data points x columns of output data)
    x                 : 1d array of x coordinates of desired data points
    y                 : 1d array of y coordinates of desired data points
    out_index         : index corresponding to column of desired data to grab
    x_ind, y_ind      : indices corresponding to x and y coordinates in mm (will usually be 5 and 6 if calibration is performed)
    
    outputs:
    vic2d_out         : 1d array of desired vic2d data at closest points to input x,y coordinates """
    
    vic2d_out             = np.zeros(x.shape[0])

    for i_loc in range(x.shape[0]):
        dist_2                = (vic2d_data[:, x_ind] - x[i_loc])**2 + (vic2d_data[:, y_ind] - y[i_loc])**2
        index                 = np.argmin(dist_2)
        vic2d_out[i_loc]      = vic2d_data[index, out_index]    
        
    return vic2d_out


    
def shift_vic2d_origin(vic2d_data, xc, yc):
    """ function shifts the coordinate system of vic2d data to a desired location at (xc, yc)
    
    inputs:
    vic2d_data        : numpy array of vic2d data read in by DataReader.vic2d_reader (number of data points x columns of output data)
    xc, yc            : x and y coordinates of desired new origin
    
    outputs:
    vic2d_data        : numpy array of vic2d data with shifted coordinate system """

    vic2d_data[:,5] -= xc
    vic2d_data[:,6] -= yc
    
    return vic2d_data