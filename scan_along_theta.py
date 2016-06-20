# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:18:44 2016

@author: kswartz92
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
#%%%
import time
dark_path             = DataReader.get_ge2_path(xray_dir, first_folders[4], first_file_nums[4])
dark_image            = DataReader.ge2_reader(dark_path)[0]
    
dir_num               = first_folders[4] + 15
file_num              = first_file_nums[4] + 15
    
path                  = DataReader.get_ge2_path(xray_dir, dir_num, file_num)
image                 = DataReader.ge2_reader(path)[0]  # only using first image because of shutter timing error
image                -= dark_image                      # subtract dark image

xda                   = np.linspace(-1024, 1024, num=2048, endpoint=True)
yda                   = np.linspace(-1024, 1024, num=2048, endpoint=True)

xd, yd                = np.meshgrid(xda, yda)

rd                    = np.sqrt(xd**2 + yd**2)
td                    = np.arctan2(yd, xd)

theta_tol             = np.radians(0.2)
theta_val             = np.pi/4
grab                  = np.abs(td-theta_val) < theta_tol
radius                = rd[grab]
line                  = image[grab]

plt.close('all')
plt.plot(radius, line, 'ok')
plt.xlim([710,740])
plt.show()
time.sleep(3)

n = 1000
keep_r                = np.linspace(np.min(radius), np.max(radius), num=n)
dr                    = keep_r[1]-keep_r[0]
keep_i                = np.zeros(n)
for i in range(n):
    keep_i[i] = np.sum(line[np.abs(radius-keep_r[i])<2*dr])
    
plt.close('all')
plt.plot(keep_r, keep_i, 'ok')
plt.xlim([710,740])
plt.show()


