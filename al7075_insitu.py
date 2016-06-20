"""
analysis for Chess beam run

@author: Kenny Swartz
06/07/2016
"""
import os
import numpy as np
import numpy.linalg as la
import scipy.optimize
import matplotlib.pyplot as plt
import importlib
os.chdir('/home/kswartz92/Documents/python_scripts/CHESS-Summer-2016')
import DataReader, DataAnalysis, PeakFitting
DataReader            = importlib.reload(DataReader)
DataAnalysis          = importlib.reload(DataAnalysis)
PeakFitting           = importlib.reload(PeakFitting)

xray_dir              = '/media/kswartz92/Swartz/Chess/Al7075insitu/'
out_dir               = '/home/kswartz92/Documents/chess/al7075_insitu/results/'
peak_dir              = '/home/kswartz92/Documents/chess/al7075_insitu/peaks/'

row_skip              = 3        # rows on top and bottom
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

strip_width           = 50             # vertical pixel width of analyzed strip
left_311              = [277,337]      # pixel range of left 311 peak
right_311             = [1713,1773]    # pixel range of right 311 peak 
top_311               = [274,334]
bot_311               = [1709,1769]

descrips              = ['initial', 'oneturn', 'twoturn', 'threeturn', 'unloaded']
#%% 
l_centers, l_errs     = np.zeros((n_step, num_data_pts)), np.zeros((n_step, num_data_pts))
t_centers, t_errs     = np.zeros((n_step, num_data_pts)), np.zeros((n_step, num_data_pts))
r_centers, r_errs     = np.zeros((n_step, num_data_pts)), np.zeros((n_step, num_data_pts))
b_centers, b_errs     = np.zeros((n_step, num_data_pts)), np.zeros((n_step, num_data_pts))
diams_h, diams_v      = np.zeros((n_step, num_data_pts)), np.zeros((n_step, num_data_pts))
for i_step in range(n_step):
    l_centers[i_step,:], l_errs[i_step,:], r_centers[i_step,:], r_errs[i_step,:], diams_h[i_step,:] = DataAnalysis.write_scan_diameters(xray_dir, out_dir, peak_dir, first_folders[i_step], first_file_nums[i_step], descrips[i_step], num_data_pts, strip_width, 'h', left_311, right_311)  
    #t_centers[i_step,:], t_errs[i_step,:], b_centers[i_step,:], b_errs[i_step,:], diams_v[i_step,:] = DataAnalysis.write_scan_diameters(xray_dir, out_dir, peak_dir, first_folders[i_step], first_file_nums[i_step], descrips[i_step], num_data_pts, strip_width, 'v', top_311,  bot_311)                          
#%%    read diameters
DataReader            = importlib.reload(DataReader)
 
diams_h               = np.zeros((n_step, num_data_pts), dtype=float)
diams_v               = np.zeros((n_step, num_data_pts), dtype=float)
for i_step in range(n_step):
    diams_h[i_step, :]   = DataReader.read_data_from_text(out_dir+descrips[i_step]+'h'+'.txt')
    diams_v[i_step, :]   = DataReader.read_data_from_text(out_dir+descrips[i_step]+'v'+'.txt')
    
# cut off unwanted rows of data
diams_array_h         = diams_h.reshape((n_step,z_num,x_num))
diams_array_h         = diams_array_h[:, row_skip:-row_skip, :]
mean_diams_h          = np.mean(diams_array_h, axis=2)

diams_array_v         = diams_v.reshape((n_step,z_num,x_num))
diams_array_v         = diams_array_v[:, row_skip:-row_skip, :]
mean_diams_v          = np.mean(diams_array_v, axis=2)
#%%  test lambda values
DataAnalysis          = importlib.reload(DataAnalysis)

def test_lambdas(data, l_rng, nl=20):
    
    lambdas = np.linspace(l_rng[0], l_rng[1], num=nl)
    error2  = np.zeros(nl)
    
    for i in range(nl):
        data_opt  = DataAnalysis.total_variation(data.copy(), lambdas[i])
        error2[i] = la.norm(data_opt-data, 2.0)**2
        
    plt.close('all')
    plt.plot(lambdas, error2)
    plt.xlabel('lambda'), plt.ylabel('2-norm of error squared')
    plt.grid(True), plt.show()
    
test_lambdas(mean_diams_h[:,3], [0,10])    
#%%   filter data
DataAnalysis          = importlib.reload(DataAnalysis)

def plot_fit(data, fit, descrip):
    plt.close('all')
    plt.xlabel('z'), plt.ylabel('ring diameter (pixels)'), plt.title(descrip)
    plt.plot(data,  'or', ms=6, label='data points')
    plt.plot(fit , '-ok', ms=6, label='optimized fit')
    plt.grid(True), plt.legend(loc=4, numpoints=1), plt.show()
    
lambda_sel_h          = 4.5
lambda_sel_v          = 2.0
diam_fits_h           = np.zeros(diams_array_h.shape)
diam_fits_v           = np.zeros(diams_array_v.shape)

for i_step in range(n_step):
    for i_col in range(x_num):
        diam_fits_h[i_step, :, i_col]  = DataAnalysis.total_variation(diams_array_h[i_step, :, i_col].copy(), lambda_sel_h)
        plot_fit(diams_array_h[i_step, :, i_col], diam_fits_h[i_step, :, i_col], 'step='+str(i_step)+' column='+str(i_col)+' h')
        diam_fits_v[i_step, :, i_col]  = DataAnalysis.total_variation(diams_array_v[i_step, :, i_col].copy(), lambda_sel_v)
        plot_fit(diams_array_v[i_step, :, i_col], diam_fits_v[i_step, :, i_col], 'step='+str(i_step)+' column='+str(i_col)+' v')
#%%    find reference diameter

def strain_integral(diam_ref, diams_meas):
    strain = (diam_ref - diams_meas) / diam_ref
    return np.sum(strain**2)

refs = np.zeros(n_step)
for i_step in range(n_step):
    refs[i_step] = scipy.optimize.minimize(strain_integral, np.mean(diam_fits_h[0,:,2]), args=(diam_fits_h[i_step,:,2])).x
ref  = np.mean(refs)

exx = (ref-diam_fits_h) / ref
eyy = (ref-diam_fits_v) / ref
#%%
def plot_data(path, z, strain, descrip, c):
     
    plt.close('all')
    plt.xlabel('z'),  plt.ylabel('strain'), plt.title(descrip)
    plt.plot(z, strain[:, 0], '-or', label='x=-6', lw=2)
    plt.plot(z, strain[:, 1], '-og', label='x=-3', lw=2)
    plt.plot(z, strain[:, 2], '-ok', label='x= 0', lw=2)
    plt.plot(z, strain[:, 3], '-ob', label='x= 3', lw=2)
    plt.plot(z, strain[:, 4], '-om', label='x= 6', lw=2)
    plt.legend(loc=1, numpoints=1)
    plt.grid(True), plt.savefig(path), plt.show()
    
def write_data(path, x, z, strain):
    out = open(path, 'w')
    for i_data_pt in range(strain.shape[0]):
        out.write('%16.12f'%x[i_data_pt]+'\t'+'%16.12f'%z[i_data_pt]+'\t'+'%16.12f'%strain[i_data_pt]+'\n')
    out.close()

colors = ['r','g','b','k','m']
for i_step in range(n_step):
    plot_data( out_dir+'exx_'+descrips[i_step]+'.tif', np.unique(z2d), exx[i_step], descrips[i_step], colors[i_step])
    write_data(out_dir+'exx_'+descrips[i_step]+'.txt', x2d.flatten(), z2d.flatten(), exx[i_step].flatten())
    
    plot_data( out_dir+'eyy_'+descrips[i_step]+'.tif', np.unique(z2d), eyy[i_step], descrips[i_step], colors[i_step])
    write_data(out_dir+'eyy_'+descrips[i_step]+'.txt', x2d.flatten(), z2d.flatten(), eyy[i_step].flatten())    
    
plt.close('all')
plt.xlabel('z'),   plt.ylabel('exx')
plt.ylim([-0.01, 0.01])
for i_step in range(n_step):
    plt.plot(np.unique(z2d), np.mean(exx[i_step], axis=1), '-', color=colors[i_step], label=descrips[i_step], lw=2)
plt.legend(numpoints=1, bbox_to_anchor=(1.38, 1.02) )
plt.grid(True), plt.savefig(out_dir+'exx_means.png'), plt.show()

plt.close('all')
plt.xlabel('z'),   plt.ylabel('eyy')
plt.ylim([-0.01, 0.01])
for i_step in range(n_step):
    plt.plot(np.unique(z2d), np.mean(eyy[i_step], axis=1), '-', color=colors[i_step], label=descrips[i_step], lw=2)
plt.legend(numpoints=1, bbox_to_anchor=(1.38, 1.02) )
plt.grid(True), plt.savefig(out_dir+'eyy_means.png'), plt.show()