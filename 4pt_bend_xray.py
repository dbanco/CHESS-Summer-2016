"""
analysis for Chess beam run

@author: Kenny Swartz
06/20/2016
"""
import numpy as np
import matplotlib.pyplot as plt
import os, importlib
os.chdir('/home/kswartz92/Documents/python_scripts/CHESS-Summer-2016')
import DataReader, DataAnalysis
DataReader            = importlib.reload(DataReader)
DataAnalysis          = importlib.reload(DataAnalysis)
data_dir              = '/media/kswartz92/Swartz/chess/'
out_dir               = '/home/kswartz92/Documents/chess/'
#%%
specimen_name         = 'ti64_plain'
x_range               = [-6,6]   # mm
x_num                 = 5
z_range               = [-4,4]   # mm
z_num                 = 33
step_names            = ['initial', '1000N', '2500N', 'unload']
dark_dirs             = [96, 265, 434, 602]
init_dirs             = [97, 266, 435, 603]
#%%
specimen_name         = 'ti64_notched'
x_range               = [-6,6]   # mm
x_num                 = 13
z_range               = [-4,4]   # mm
z_num                 = 41
step_names            = ['initial', '1000N', '2000N', 'unload']
dark_dirs             = [5, 544, 1081, 1618]
init_dirs             = [6, 545, 1082, 1619]
#%%
specimen_name         = 'al7075_plain'
x_range               = [-2,2]   # mm
x_num                 = 3
z_range               = [-2,2]   # mm
z_num                 = 21
step_names            = ['initial', 'load1', 'unload1', 'load2', 'unload2', 'load3', 'unload3']
dark_dirs             = [25, 170, 236, 302, 367, 435, 500]
init_dirs             = [25, 170, 236, 302, 367, 435, 500]
#%%
specimen_name         = 'al7075_mlf'
x_range               = [-6,6]   # mm
x_num                 = 5
z_range               = [-5,5]   # mm
z_num                 = 41
step_names            = ['initial', 'oneturn', 'twoturn', 'threeturn', 'unloaded']
dark_dirs             = [68, 277, 485, 692, 899]
init_dirs             = [68, 277, 485, 692, 899]
#%%
ring_name             = 'ti_100'
left                  = [508, 548]
right                 = [1506, 1546]
top                   = [505, 545]
bottom                = [1505, 1545]
lambda_sel            = 1.0
#%%
ring_name             = 'al_111'
left                  = [372,432]      
right                 = [1623,1683]    
top                   = []
bottom                = []
lambda_sel            = 2.0
#%%
ring_name             = 'al_311'
left                  = [277,337]
right                 = [1713,1773]
top                   = [274,334]
bottom                = [1709,1769]
lambda_sel            = 2.0 
#%%
sample                = DataAnalysis.Specimen(specimen_name, data_dir, out_dir, x_range, x_num, z_range, z_num, step_names, dark_dirs, init_dirs)
ring                  = DataAnalysis.Ring(ring_name, sample, left, right, top, bottom, lambda_sel)

if specimen_name == 'ti64_notched' or specimen_name=='ti64_plain':
    xa                    = np.linspace(sample.x_range[1], sample.x_range[0], num=x_num, endpoint=True)
    za                    = np.linspace(sample.z_range[1], sample.z_range[0], num=z_num, endpoint=True)
    z2d, x2d              = np.meshgrid(za, xa)
if specimen_name == 'al7075_plain':
    xa                    = np.linspace(x_range[1], x_range[0], num=x_num, endpoint=True)
    za                    = np.linspace(z_range[1], z_range[0], num=z_num, endpoint=True)
    x2d, z2d              = np.meshgrid(xa, za)
if specimen_name == 'al7075_mlf':
    xa                    = np.linspace(x_range[0], x_range[1], num=x_num, endpoint=True)
    za                    = np.linspace(z_range[0], z_range[1], num=z_num, endpoint=True)
    x2d, z2d              = np.meshgrid(xa, za)

x1d, z1d              = x2d.flatten(), z2d.flatten()
#%%    read peak diameters if they have been fit, if not fit here

orient                = 'h'
try:
    x, z, diams = [], [], []
    for i_step in range(sample.n_load_step):
        txt_data          = DataReader.read_data_from_text(ring.out_dir+sample.step_names[i_step]+'_diams_'+orient+'.txt')
        x.append(txt_data[:, 0]), z.append(txt_data[:, 1]), diams.append(txt_data[:, 2])
except:
    l_centers, l_errs     = np.zeros((sample.n_load_step, sample.n_data_pt)), np.zeros((sample.n_load_step, sample.n_data_pt))
    u_centers, u_errs     = np.zeros((sample.n_load_step, sample.n_data_pt)), np.zeros((sample.n_load_step, sample.n_data_pt))
    diams                 = np.zeros((sample.n_load_step, sample.n_data_pt))
    for i_step in range(sample.n_load_step):         
        l_centers[i_step,:], l_errs[i_step,:], u_centers[i_step,:], u_errs[i_step,:], diams[i_step,:] = DataAnalysis.write_scan_diameters(sample, ring, x1d, z1d, i_step, orient)

#%% 

#     total variation filtering 

fits, coords          = [], []
for i_step in range(sample.n_load_step):
    s_fits, s_coords      = [], []
    xvals                 = np.unique(x[i_step])
    for xval in xvals:
        x_col                 = x[i_step][x[i_step]==xval]
        z_col                 = z[i_step][x[i_step]==xval]
        diam_col              = diams[i_step][x[i_step]==xval]
        fit                   = DataAnalysis.total_variation(diam_col, ring.lambda_sel) 
        path                  = ring.out_dir+sample.step_names[i_step]+'_x_'+str(xval)+orient+'.tiff'
        DataAnalysis.plot_data_fit(path, diam_col, fit, sample.step_names[i_step]+', x='+str(xval)+', '+orient)
        s_fits.append(fit)
        s_coords.append([x_col, z_col])
    fits.append(s_fits)
    coords.append(s_coords)
    
#    strain calculation

ref_diam              = DataAnalysis.find_ref_diam(fits[1][(sample.x_num-1)//2], ring.right[0]-ring.left[0])
strain                = []
for i_step in range(sample.n_load_step):
    s_strain = []
    for i_col in range(len(fits[i_step])):
        s_strain.append((ref_diam - fits[i_step][i_col]) / ref_diam)
    strain.append(s_strain)
    
    
#    average strains

    
    

#     plot data 
    
for i_step in range(sample.n_load_step):
    path = ring.out_dir+sample.step_names[i_step]+'_strain_'+orient+'.tiff'   
    plt.close('all')
    plt.xlabel('z'),  plt.ylabel('strain')
    for i_col in range(len(strain[i_step])):
        plt.plot(coords[i_step][i_col][1], strain[i_step][i_col], '-', label='x='+str(coords[0][i_col][0][0]), lw=2)
    plt.legend(numpoints=1, bbox_to_anchor=(1.3, 1.02) )
    plt.grid(True), plt.savefig(path)

"""
plt.close('all')
plt.xlabel('z'),   plt.ylabel('strain')
for i_step in range(sample.n_load_step):
    plt.plot(coords[i_step][0][1], strain[i_step][i_col], '-', label=sample.step_names[i_step], lw=2)
plt.grid(True), plt.legend(numpoints=1, bbox_to_anchor=(1.38, 1.02) ), plt.savefig(ring.out_dir+'mean_strains_'+orient+'.png')
plt.close('all')
"""

#     write data

for i_step in range(sample.n_load_step):
    path = ring.out_dir+sample.step_names[i_step]+'_strain_'+orient+'.txt'
    out = open(path, 'w')
    for i_col in range(len(strain[i_step])):
        for i_data_pt in range(len(strain[i_step][i_col])):
            out.write('%16.12f'%coords[i_step][i_col][0][i_data_pt]+'\t'+'%16.12f'%coords[i_step][i_col][1][i_data_pt]+'\t'+'%16.12f'%strain[i_step][i_col][i_data_pt]+'\n')
    out.close()
    