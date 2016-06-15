"""
analysis for Chess beam run

@author: Kenny Swartz
06/07/2016
"""
#%%
import os
import numpy as np
import numpy.linalg as la
import scipy.optimize
import matplotlib.pyplot as plt
import importlib
os.chdir('/home/kswartz92/Documents/python_scripts/CHESS-Summer-2016')
import DataReader
import DataAnalysis

xray_dir              = '/media/kswartz92/Swartz/Chess/Al7075insitu/'
out_dir               = '/home/kswartz92/Documents/chess/al7075_insitu/results'
peak_dir              = '/home/kswartz92/Documents/chess/al7075_insitu/peaks'

row_skip              = 1        # rows on top and bottom
x_range               = [-6,6]   # mm
x_num                 = 5
z_range               = [-5,5]   # mm
z_num                 = 41
num_data_pts          = x_num*z_num
xa                    = np.linspace(x_range[0], x_range[1], num=x_num, endpoint=True)
za                    = np.linspace(z_range[0], z_range[1], num=z_num, endpoint=True)
x2d, z2d              = np.meshgrid(xa,za)
x2d                   = x2d[row_skip:-row_skip, :]
z2d                   = z2d[row_skip:-row_skip, :]

n_step                = 5
calibrant_dirs        = [11  , 276 , 484 , 691 , 898 ]
calibrant_files       = [1240, 1502, 1708, 1914, 2120]

first_folders         = [68  , 277 , 485 , 692 , 899 ]
first_file_nums       = [1297, 1503, 1709, 1915, 2121]

descrips              = ['initial', 'oneturn', 'twoturn', 'threeturn', 'unloaded']
#%%    read detector data
DataAnalysis          = importlib.reload(DataAnalysis)

strip_width           = 20             # vertical pixel width of analyzed strip
left_311              = [307,320]      # pixel range of left 311 peak
right_311             = [1730,1760]    # pixel range of right 311 peak 
 
# find centers
for i_step in range(n_step):
    DataAnalysis.write_scan_diameters(xray_dir, out_dir, peak_dir, first_folders[i_step], first_file_nums[i_step], descrips[i_step], num_data_pts, strip_width, 'h', left_311, right_311)     
    DataAnalysis.write_scan_diameters(xray_dir, out_dir, peak_dir, first_folders[i_step], first_file_nums[i_step], descrips[i_step], num_data_pts, strip_width, 'v', left_311, right_311)                          
#%%    read diameters
DataReader            = importlib.reload(DataReader)
 
diams_h               = np.zeros((n_step, num_data_pts), dtype=float)
diams_v               = np.zeros((n_step, num_data_pts), dtype=float)
for i_step in range(n_step):
    diams_h[i_step,:]   = DataReader.read_data_from_text(out_dir+descrips[i_step]+'h'+'.txt')
    diams_v[i_step,:]   = DataReader.read_data_from_text(out_dir+descrips[i_step]+'v'+'.txt')

# cut off unwanted rows of data
diams2d_h             = diams_h.reshape((n_step,z_num,x_num))
diams2d_h             = diams2d_h[:, row_skip:-row_skip, :]
mean_diams_h          = np.mean(diams2d_h, axis=2)

diams2d_v             = diams_v.reshape((n_step,z_num,x_num))
diams2d_v             = diams2d_v[:, row_skip:-row_skip, :]
mean_diams_v          = np.mean(diams2d_v, axis=2)
#%%
DataAnalysis          = importlib.reload(DataAnalysis)

def pluck_point(array, index):
        return np.hstack([array[:index], array[index+1:]])  


def test_lambdas(diams, pls, l_rng, nl=20):
    
    lambdas = np.linspace(l_rng[0], l_rng[1], num=nl)
    error2  = np.zeros(nl)
    
    for i in range(nl):
        
        data = diams.copy()
        x    = np.arange(data.shape[0])
        
        for pl in pls:
            data = pluck_point(data, pl)
            x    = pluck_point(x, pl)
            
        data_opt  = DataAnalysis.total_variation(data, lambdas[i])
        error2[i] = la.norm(data_opt-data, 2.0)**2
        
    plt.close('all')
    plt.plot(lambdas, error2)
    plt.xlabel('lambda')
    plt.ylabel('2-norm of error squared')
    plt.grid(True)
    plt.show()
    
test_lambdas(mean_diams_v[:,4], [], [0,10])    
#%%
DataAnalysis          = importlib.reload(DataAnalysis)


def filter_data(data_array, pls, lambs):
    
    pos, diam_fits = [], [] 
    
    for i_step in range(n_step):
        
        data = data_array[i_step, :].copy()
        z    = z2d[:,0].copy()
        
        for pl in pls[i_step]:
            data = pluck_point(data, pl)
            z    = pluck_point(z, pl)
            
        data_opt  = DataAnalysis.total_variation(data, lambs[i_step])
        
        pos.append(z)
        diam_fits.append(data_opt)
        
        plt.close('all')
        plt.xlabel('z')
        plt.ylabel('311 diameter (pixels)')
        plt.title(descrips[i_step])
        plt.plot(z2d[:,0], data_array[i_step, :] , 'or', ms=6, label='unused points')
        plt.plot(z,        data                  , 'og', ms=6, label='used points')
        plt.plot(z,        data_opt              , '-k', lw=2, label='optimized fit')
        plt.xlim([-5,5])
        plt.ylim([1410, 1440])
        plt.grid(True)
        plt.legend(loc=4, numpoints=1)
        plt.show()
        
    return pos, diam_fits


pls_h              = [ [1,20],   [],    [],  [4,4,32],  [0,0,0,9] ]
lambs_h            = [  4.0  ,  4.4,   6.9,       5.9,       2.8  ]

pos_h, diam_fits_h = filter_data(mean_diams_h, pls_h, lambs_h)

pls_v              = [ [1,8,9,11,12,13],   [8,31],    [32],  [26,27],  [2,20,24] ]
lambs_v            = [          5      ,    1 ,        2,       1,      1  ]

pos_v, diam_fits_v = filter_data(mean_diams_v, pls_v, lambs_v)
   
    
#%%    optimization

def find_reference_diam(diams, ref0):
    
    def obj(ref, diams):
        strain = (ref-diams) / ref
        return np.sum(strain**2)
        
    optimize_results = scipy.optimize.minimize(obj, ref0, args=(diams))

    return optimize_results.x

refs = np.zeros(n_step)
for i_step in range(n_step):
    refs[i_step] =  find_reference_diam(diam_fits_h[i_step], np.mean(diam_fits_h[0]))
ref = np.mean(refs)

exx_strain = (ref-diam_fits_h) / ref
eyy_strain = (ref-diam_fits_v) / ref
#%%
def plot_data(path, z, strain, descrip, c):
     
    plt.close('all')
    plt.xlabel('z')
    plt.ylabel('strain')
    plt.xlim([-5,5])
    plt.ylim([-0.006, 0.006])
    plt.plot(z, strain, '-', color=c, label=descrip, lw=2)
    plt.grid(True)
    plt.legend(loc=1, numpoints=1)
    plt.show()
    plt.savefig(path)

def write_data(path, z, strain):
    out = open(path, 'w')
    for i_data_pt in range(strain.shape[0]):
        out.write('%16.12f'%z[i_data_pt]+'\t'+'%16.12f'%strain[i_data_pt]+'\n')
    out.close()

colors = ['r','g','b','k','m']
for i_step in range(n_step):
    plot_data( out_dir+'exx_'+descrips[i_step]+'.png', pos_h[i_step], exx_strain[i_step], descrips[i_step], colors[i_step])
    plot_data( out_dir+'eyy_'+descrips[i_step]+'.png', pos_v[i_step], eyy_strain[i_step], descrips[i_step], colors[i_step])
    write_data(out_dir+'exx_'+descrips[i_step]+'.txt', pos_h[i_step], exx_strain[i_step])
    write_data(out_dir+'eyy_'+descrips[i_step]+'.txt', pos_v[i_step], eyy_strain[i_step])
    
plt.close('all')
plt.xlabel('z')
plt.ylabel('exx')
plt.xlim([-5,5])
plt.ylim([-0.006, 0.006])
for i_step in range(n_step):
    plt.plot(pos_h[i_step], exx_strain[i_step], '-', color=colors[i_step], label=descrips[i_step], lw=2)
plt.grid(True)
plt.legend(numpoints=1, bbox_to_anchor=(1.38, 1.02) )
plt.show()