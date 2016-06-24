"""
Data Analysis Functions

@author: Kenny Swartz
06/07/2016
"""
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os, importlib, scipy.optimize
import DataReader, PeakFitting, Toolbox
DataReader  = reload(DataReader)
PeakFitting = reload(PeakFitting)
Toolbox     = reload(Toolbox)


class Specimen:
    def __init__(self, name, data_dir, out_dir, x_range, x_num, z_range, z_num, step_names, dark_dirs, init_dirs):

        self.name             = name
        self.data_dir         = data_dir+name+'/'
        self.out_dir          = out_dir +name+'/'        
        
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)        
        if not os.path.exists(self.out_dir+'peaks/'):
            os.mkdir(self.out_dir+'peaks/')
        if not os.path.exists(self.out_dir+'results/'):
            os.mkdir(self.out_dir+'results/')       
        
        self.x_range          = x_range
        self.x_num            = x_num
        self.z_range          = z_range
        self.z_num            = z_num
        self.step_names       = step_names
        self.n_load_step      = len(step_names)
        self.dark_dirs        = dark_dirs
        self.init_dirs        = init_dirs
        self.n_data_pt        = x_num*z_num
        

class Ring:
    def __init__(self, name, sample, left_range, right_range, top_range, bottom_range, lambda_sel, strip_width=50):
        
        self.name             = name
        self.peak_dir         = sample.out_dir+'peaks/'+name+'/'
        self.out_dir          = sample.out_dir+'results/'+name+'/'
        
        if not os.path.exists(self.peak_dir):
            os.mkdir(self.peak_dir)
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
            
        self.left             = left_range
        self.right            = right_range
        self.top              = top_range
        self.bottom           = bottom_range
        self.strip_width      = strip_width
        self.lambda_sel       = lambda_sel


def total_variation(y, lamb=0):
    
    # second order central difference approximation to second derivative
    D2          = Toolbox.D2_SOCD(y.shape[0])
    
    # optimization 
    def obj(x, y, lamb, D2): 
        return la.norm(x-y, 2.0)**2 + lamb*la.norm(np.dot(D2, x), 1.0)
     
    x0          = np.mean(y)*np.ones(y.shape[0])   # initial guess
    opt_results = scipy.optimize.minimize(obj, x0, args=(y, lamb, D2))
    
    return opt_results.x    


def plot_data_fit(path, data, fit, descrip):
    plt.close('all'), plt.ylabel('data'), plt.title(descrip)
    plt.plot(data,  'or', ms=6, label='data points')
    plt.plot(fit , '-ok', ms=6, label='optimized fit')
    plt.grid(True), plt.legend(numpoints=1, bbox_to_anchor=(1.4, 1.02))
    plt.savefig(path), plt.close('all')
    
    
def test_lambdas(data, l_rng, nl=20):
    
    lambdas = np.linspace(l_rng[0], l_rng[1], num=nl)
    error2  = np.zeros(nl)
    
    for i in range(nl):
        data_opt  = total_variation(data, lambdas[i])
        error2[i] = la.norm(data_opt-data, 2.0)**2
        
    plt.close('all')
    plt.plot(lambdas, error2)
    plt.xlabel('lambda'), plt.ylabel('2-norm of error squared')
    plt.grid(True), plt.show()


def find_ref_diam(diams, ref0):
    
    def strain_integral(ref, diams):
        strain = (ref - diams) / ref
        return np.sum(strain**2)

    result = scipy.optimize.minimize(strain_integral, ref0, args=(diams))
    assert result.success == True
        
    return result.x


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


    
def analyze_strip(image, strip_orient, strip_width, pix_rng, peak_path, fwhm0, amp0, strip_loc = 0.5):
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
    x                     = np.arange(len(peak))
    peakCtr, loCut, hiCut = get_peak_fit_indices(peak)
    peak_bg_rm, _         = PeakFitting.RemoveBackground(x, peak, loCut, hiCut)
    peak_fit, p_opt, err  = PeakFitting.fitPeak(x, peak_bg_rm, peakCtr, fwhm0, amp0)
    peak_ctr              = p_opt[0]
    true_center           = peak_ctr + pix_rng[0]

    plt.close('all')
    plt.plot([peak_ctr,peak_ctr],[0,np.max(peak)],'--r')
    plt.plot(x, peak,       'ok')
    plt.plot(x, peak_bg_rm, 'or')
    plt.plot(x, peak_fit,   '-r', lw=3)
    plt.savefig(peak_path)
    plt.close('all')

    del line, peak, x, peak_bg_rm, peak_fit

    return true_center, err 



def write_scan_diameters(sample, ring, x, z, step_num, orient, err_thres = 0.2, fwhm0=10, amp0=3000):

    if orient == 'h':    
        l_peak_rng = ring.left
        u_peak_rng = ring.right
    if orient == 'v':
        l_peak_rng = ring.top
        u_peak_rng = ring.bottom
            
    dark_path             = sample.data_dir+str(sample.dark_dirs[step_num])+'/ff/'
    dark_file             = os.listdir(dark_path)
    assert len(dark_file) == 1
    dark_image            = DataReader.ge2_reader(dark_path+dark_file[0])
    if len(dark_image.shape) > 1:
        dark_image        = np.mean(dark_image, axis=0)
    
    l_centers, l_errs     = np.zeros(sample.n_data_pt), np.zeros(sample.n_data_pt)
    u_centers, u_errs     = np.zeros(sample.n_data_pt), np.zeros(sample.n_data_pt)
    diams                 = np.zeros(sample.n_data_pt)  
     
    for i_data_pt in range(sample.n_data_pt):
        
        dir_num               = sample.init_dirs[step_num] + i_data_pt
        path                  = sample.data_dir+str(dir_num)+'/ff/'
        file                  = os.listdir(path)
        assert len(file) == 1
        
        print('reading image ' + str(dir_num))
        image                 = DataReader.ge2_reader(path+file[0])[0]  # only using first image because of shutter timing error
        image                -= dark_image                              # subtract dark image
        
        peak_path                                = ring.peak_dir+orient+'_lower_'+str(dir_num)+'.tiff'
        l_centers[i_data_pt], l_errs[i_data_pt]  = analyze_strip(image, orient, ring.strip_width, l_peak_rng, peak_path, fwhm0, amp0)
        
        peak_path                                = ring.peak_dir+orient+'_upper_'+str(dir_num)+'.tiff'
        u_centers[i_data_pt], u_errs[i_data_pt]  = analyze_strip(image, orient, ring.strip_width, u_peak_rng, peak_path, fwhm0, amp0)
        
        diams[i_data_pt]                         = u_centers[i_data_pt] - l_centers[i_data_pt]
                
        del image

    # write data to a text file
    out_file = open(ring.out_dir+sample.step_names[step_num]+'_diams_'+orient+'.txt', 'w')
    for i_data_pt in range(sample.n_data_pt):
        if l_errs[i_data_pt] <= err_thres and u_errs[i_data_pt] <= err_thres:
            out_file.write('%18.12f'%x[i_data_pt]     + '\t')
            out_file.write('%18.12f'%z[i_data_pt]     + '\t')
            out_file.write('%18.12f'%diams[i_data_pt] + '\n')
    out_file.close()     
    
    return l_centers, l_errs, u_centers, u_errs, diams

  
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