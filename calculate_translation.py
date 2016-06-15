"""
Created on Thu Jun  9 09:50:31 2016

@author: Kenny Swartz
"""

import numpy as np

def get_samp_shifts(w, shift_wrt_beam):
    """ function calculates how to shift sampX and sampY when load frame is rotated relative to sample 
        assumes sample is properly rotated for scanning
     
    inputs:
    
    w                     : load frame angle in degrees
    shift_wrt_beam        : 1d numpy array containing desired shift perpendicular and parallell to xray beam
                            (1st entry positive cooresponds to moving away from workstation
                             2nd entry positive cooresponds to moving downstream)
    
    outputs:
    
    shift_wrt_samp        : 1d numpy array containing sampX and sampY shifts that will result in desired shift w.r.t. beam """
    
    def rot_mat(w):
        R                     = np.array([[np.cos(np.radians(w)), -np.sin(np.radians(w))],
                                          [np.sin(np.radians(w)),  np.cos(np.radians(w))] ])
        return R
        
    shift_wrt_samp      = np.dot(rot_mat(-w), shift_wrt_beam)  # rotate shift vector by -w to place in sampX and sampY coordinate system

    print('sampX = ' + '%8.6f'%shift_wrt_samp[0])
    print('sampY = ' + '%8.6f'%shift_wrt_samp[1])

    return shift_wrt_samp     
    

# create coordinate arrays 
x_range               = [6,-6]
x_num                 = 5
z_range               = [5,-5]
z_num                 = 41
num_data_pts          = x_num*z_num
xa                    = np.linspace(x_range[0], x_range[1], num=x_num, endpoint=True)
za                    = np.linspace(z_range[0], z_range[1], num=z_num, endpoint=True)
x_array, z_array      = np.meshgrid(xa,za)
x1d                   = x_array.flatten()
z1d                   = z_array.flatten()  

w                     = 20

sampX_1d              = np.zeros(x1d.shape[0])
sampY_1d              = np.zeros(x1d.shape[0])

for i_pt in range(x1d.shape[0]):
    shift_wrt_samp = get_samp_shifts(w, np.array([x1d[i_pt], 0]))
    sampX_1d[i_pt] = shift_wrt_samp[0]
    sampY_1d[i_pt] = shift_wrt_samp[1]