"""
Data Analysis Functions

@author: Kenny Swartz
06/07/2016
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.optimize
import importlib
import DataReader, PeakFitting, Toolbox
DataReader  = importlib.reload(DataReader)
PeakFitting = importlib.reload(PeakFitting)
Toolbox     = importlib.reload(Toolbox)


def total_variation(y, lamb=0):
    
    # second order central difference approximation to second derivative
    D2          = Toolbox.D2_SOCD(y.shape[0])
    
    # optimization 
    def obj(x, y, lamb, D2): 
        return la.norm(x-y, 2.0)**2 + lamb*la.norm(np.dot(D2, x), 1.0)
     
    x0          = np.mean(y)*np.ones(y.shape[0])   # initial guess
    opt_results = scipy.optimize.minimize(obj, x0, args=(y, lamb, D2))
    
    return opt_results.x    



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



def get_average_strip(xray_dir, first_folder, first_file_num, num_data_pts, strip_width, strip_orient, image_size=2048):
       
    dark_path   = DataReader.get_ge2_path(xray_dir, first_folder, first_file_num)
    dark_image  = DataReader.ge2_reader(dark_path)[0]
    
    strips      = np.zeros((num_data_pts, strip_width, image_size))
    
    for i_data_pt in range(num_data_pts):
        dir_num               = first_folder    + i_data_pt
        file_num              = first_file_num  + i_data_pt
        print('reading image ' + str(dir_num))
        path                  = DataReader.get_ge2_path(xray_dir, dir_num, file_num)
        image                 = DataReader.ge2_reader(path)[0]  # only using first image because of shutter timing error
        image                -= dark_image    
        
        if strip_orient == 'v':
            strips[i_data_pt]  =  image[ :, image.shape[1]//2-strip_width//2 : image.shape[1]//2+strip_width//2 ]
    
<<<<<<< HEAD
def analyze_strip(image, strip_orient, strip_width, pix_rng, peak_path, strip_loc = 0.5, FWHM=10, Am=2000):
=======
        if strip_orient == 'h':
            strips[i_data_pt]  = image[ image.shape[0]//2-strip_width//2 : image.shape[0]//2+strip_width//2 , :]
            
    if strip_orient == 'v': 
        return np.mean(strips, axis=1)
    
    if strip_orient == 'h':
        return np.mean(strips, axis=0)
 

   
def analyze_strip(image, strip_orient, strip_width, pix_rng, peak_path, fwhm0, amp0):
>>>>>>> 8776a81ec32c4f7930dac524ee228f1edd1da7a9
    """ function fits a peak to a summed strip of a ge2 detector image
    
    inputs:
    image             : ge2 detector image
    strip_orient      : 'v' for vertical strip, 'h' for horizontal strip
    strip_width       : width of strip that is analyzed to fit peak
    pix_rnge          : range of pixels where the desired peak is located
    peak_path         : path where peak plot should be saved
    
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
<<<<<<< HEAD
    x_index               = np.linspace(1,len(peak),num=len(peak))
    peakCtr, loCut, hiCut = get_indices(peak)
    y_bg_corr, background = PeakFitting.RemoveBackground(x_index, peak, peakCtr, loCut, hiCut)
    fit, best_parameter   = PeakFitting.fitPeak(x_index, y_bg_corr, FWHM, peakCtr, Am, FitType = 'Gaussian')
    fit_center            = best_parameter[1]
    true_center           = fit_center + pix_rng[0]
    std_dev_left          = best_parameter[0]
    std_dev_right         = best_parameter[2]
    """
=======
    x                     = np.arange(len(peak))
    peakCtr, loCut, hiCut = get_peak_fit_indices(peak)
    peak_bg_rm, _         = PeakFitting.RemoveBackground(x, peak, loCut, hiCut)
    peak_fit, p_opt, err  = PeakFitting.fitPeak(x, peak_bg_rm, peakCtr, fwhm0, amp0)
    peak_ctr              = p_opt[0]
    true_center           = peak_ctr + pix_rng[0]

>>>>>>> 8776a81ec32c4f7930dac524ee228f1edd1da7a9
    plt.close('all')
    plt.plot([peak_ctr,peak_ctr],[0,np.max(peak)],'--r')
    plt.plot(x, peak,       'ok')
    plt.plot(x, peak_bg_rm, 'or')
    plt.plot(x, peak_fit,   '-r', lw=3)
    plt.savefig(peak_path)
    plt.close('all')
<<<<<<< HEAD
    """
    return true_center, std_dev_left, std_dev_right
=======

    return true_center, err 
>>>>>>> 8776a81ec32c4f7930dac524ee228f1edd1da7a9



def write_scan_diameters(xray_dir, out_dir, peak_dir, first_folder, first_file_num, descrip, num_data_pts, strip_width, orient, lt_peak_rng, rb_peak_rng, fwhm0=10, amp0=3000):
            
    dark_path             = DataReader.get_ge2_path(xray_dir, first_folder, first_file_num)
    dark_image            = DataReader.ge2_reader(dark_path)[0]
    
    lt_centers, lt_errs   = np.zeros(num_data_pts), np.zeros(num_data_pts)
    rb_centers, rb_errs   = np.zeros(num_data_pts), np.zeros(num_data_pts)
    diams                 = np.zeros(num_data_pts)  # ring diameters
     
    for data_pt in range(num_data_pts):
        
        dir_num               = first_folder   + data_pt
        file_num              = first_file_num + data_pt
        
        print('reading image ' + str(dir_num))
        path                  = DataReader.get_ge2_path(xray_dir, dir_num, file_num)
        image                 = DataReader.ge2_reader(path)[0]  # only using first image because of shutter timing error
        image                -= dark_image                       # subtract dark image
        
        peak_path             = peak_dir+'peak_'+str(dir_num)+orient+'lt'+'.tiff'
        center, err           = analyze_strip(image, orient, strip_width, lt_peak_rng, peak_path, fwhm0, amp0)
        lt_centers[data_pt]   = center
        lt_errs[data_pt]      = err
                
        peak_path             = peak_dir+'peak_'+str(dir_num)+orient+'rb'+'.tiff'
        center, err           = analyze_strip(image, orient, strip_width, rb_peak_rng, peak_path, fwhm0, amp0)
        rb_centers[data_pt]   = center
        rb_errs[data_pt]      = err
        
        diams[data_pt]        = rb_centers[data_pt] - lt_centers[data_pt]

    # write data to a text file
    out_file = open(out_dir+descrip+orient+'.txt', 'w')
    for data_pt in range(num_data_pts):
        out_file.write('%18.14f'%diams[data_pt] + '\n')
    out_file.close()     
    
    return lt_centers, lt_errs, rb_centers, rb_errs, diams

    
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