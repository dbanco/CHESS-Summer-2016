# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 08:18:42 2016

@author: kswartz92
"""

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
import DataReader, DataAnalysis
DataReader            = importlib.reload(DataReader)
DataAnalysis          = importlib.reload(DataAnalysis)

xray_dir              = '/media/kswartz92/Swartz/Chess/Al7075_CLF_1/'
out_dir               = '/home/kswartz92/Documents/chess/al7075_clf_1/results/'
peak_dir              = '/home/kswartz92/Documents/chess/al7075_clf_1/peaks/'

row_skip              = 3        # rows on top and bottom
x_range               = [-2,2]   # mm
x_num                 = 3
z_range               = [-2,2]   # mm
z_num                 = 21
num_data_pts          = x_num*z_num
xa                    = np.linspace(x_range[1], x_range[0], num=x_num, endpoint=True)
za                    = np.linspace(z_range[1], z_range[0], num=z_num, endpoint=True)
x2d, z2d              = np.meshgrid(xa,za)
x2d                   = x2d[row_skip:-row_skip, :]
z2d                   = z2d[row_skip:-row_skip, :]

n_step                = 6
first_folders         = [170, 236, 302, 367, 435, 500]
first_file_nums       = [162, 225, 288, 351, 414, 477]

strip_width           = 50             # vertical pixel width of analyzed strip
left_111              = [372,432]      # pixel range of left 111 peak
right_111             = [1623,1683]    # pixel range of right 111 peak 

descrips              = ['load1', 'unload1', 'load2', 'unload2', 'load3', 'unload3']
#%%
l_centers             = np.zeros((n_step, num_data_pts))
l_errs                = np.zeros((n_step, num_data_pts))
r_centers             = np.zeros((n_step, num_data_pts))
r_errs                = np.zeros((n_step, num_data_pts))
diams                 = np.zeros((n_step, num_data_pts))
for i_step in range(n_step):         
    l_centers[i_step,:], l_errs[i_step,:], r_centers[i_step,:], r_errs[i_step,:], diams[i_step,:] = DataAnalysis.write_scan_diameters(xray_dir, out_dir, peak_dir, first_folders[i_step], first_file_nums[i_step], descrips[i_step], num_data_pts, strip_width, 'h', left_111, right_111)                            
#%%    read diameters
DataReader            = importlib.reload(DataReader)
 
diams_h               = np.zeros((n_step, num_data_pts), dtype=float)
for i_step in range(n_step):
    diams_h[i_step,:]   = DataReader.read_data_from_text(out_dir+descrips[i_step]+'h'+'.txt')

# cut off unwanted rows of data
diams_array_h         = diams_h.reshape((n_step,z_num,x_num))
diams_array_h         = diams_array_h[:, row_skip:-row_skip, :]
mean_diams_h          = np.mean(diams_array_h, axis=2)
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
    plt.grid(True), plt.legend(numpoints=1, bbox_to_anchor=(1.4, 1.02) ), plt.show()
    
lambda_sel            = 3.5
diam_fits_h           = np.zeros(diams_array_h.shape)

for i_step in range(n_step):
    for i_col in range(x_num):
        diam_fits_h[i_step, :, i_col]  = DataAnalysis.total_variation(diams_array_h[i_step, :, i_col].copy(), lambda_sel)
        plot_fit(diams_array_h[i_step, :, i_col], diam_fits_h[i_step, :, i_col], 'step='+str(i_step)+' column='+str(i_col)+' h')
#%%    find reference diameter

def strain_integral(diam_ref, diams_meas):
    strain = (diam_ref - diams_meas) / diam_ref
    return np.sum(strain**2)

refs = np.zeros(n_step)
for i_step in range(n_step):
    refs[i_step] = scipy.optimize.minimize(strain_integral, np.mean(diam_fits_h[0,:,2]), args=(diam_fits_h[i_step,:,2])).x
ref  = np.mean(refs)

exx_strain = (ref-diam_fits_h) / ref
#%%
def plot_data(path, z, strain, descrip, c):
     
    plt.close('all')
    plt.xlabel('z'),  plt.ylabel('strain'), plt.title(descrip)
    plt.ylim([-0.015, 0.015])
    plt.plot(z, strain[:, 0], '-or', label='x=-2', lw=2)
    plt.plot(z, strain[:, 1], '-og', label='x= 0', lw=2)
    plt.plot(z, strain[:, 2], '-ok', label='x= 2', lw=2)
    plt.legend(numpoints=1, bbox_to_anchor=(1.3, 1.02) )
    plt.grid(True), plt.savefig(path), plt.show()
    
def write_data(path, x, z, strain):
    out = open(path, 'w')
    for i_data_pt in range(strain.shape[0]):
        out.write('%16.12f'%x[i_data_pt]+'\t'+'%16.12f'%z[i_data_pt]+'\t'+'%16.12f'%strain[i_data_pt]+'\n')
    out.close()

colors = ['r','g','b','k','m','c']
for i_step in range(n_step):
    plot_data( out_dir+'exx_'+descrips[i_step]+'.png', z2d[:,0], exx_strain[i_step], descrips[i_step], colors[i_step])
    write_data(out_dir+'exx_'+descrips[i_step]+'.txt', x2d.flatten(), z2d.flatten(), exx_strain[i_step].flatten())
    
plt.close('all')
plt.xlabel('z'),   plt.ylabel('exx')
plt.ylim([-0.015, 0.015])
for i_step in range(n_step):
    plt.plot(z2d[:,0], np.mean(exx_strain[i_step], axis=1), '-', color=colors[i_step], label=descrips[i_step], lw=2)
plt.legend(numpoints=1, bbox_to_anchor=(1.38, 1.02) )
plt.grid(True), plt.savefig(out_dir+'exx_means.png'), plt.show()