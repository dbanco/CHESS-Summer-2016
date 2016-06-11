"""
python script used to track scan points with vic2d results

@author: Kenny Swartz
06/07/2015
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib
import os
os.chdir('/home/kswartz92/Documents/python_scripts/')
import DataReader
import DataAnalysis
DataReader            = importlib.reload(DataReader)
DataAnalysis          = importlib.reload(DataAnalysis)

# User Inputs

dic_directory         = '/media/kswartz92/Swartz/Chess/Al7075insitu/DIC/'
initial_file          = 'dic_4536.csv'
deformed_file         = 'dic_4539.csv'

# vic2d coordinate system
x_center              = 0.16           # mm
y_center              = 2.11           # mm

# hutch coordinate system
x_range               = [-6,6]         # mm
x_num                 = 5
z_range               = [-5,5]         # mm
z_num                 = 41

# read vic2d data
initial_data          = DataReader.vic2d_reader(dic_directory+initial_file)
deformed_data         = DataReader.vic2d_reader(dic_directory+deformed_file)

# shift coordinates so origin is at sample center
initial_data          = DataAnalysis.shift_vic2d_origin(initial_data,  x_center, y_center)
deformed_data         = DataAnalysis.shift_vic2d_origin(deformed_data, x_center, y_center)

# create 1d coordinate arrays
xa                    = np.linspace(x_range[0], x_range[1], num=x_num, endpoint=True)
za                    = np.linspace(z_range[0], z_range[1], num=z_num, endpoint=True)
x_array, z_array      = np.meshgrid(xa,za)
x1d                   = x_array.flatten()
z1d                   = z_array.flatten()    

# find x,y,u,v at closest vic2d points 
initial_x             = DataAnalysis.find_closest_vic2d(initial_data,  x1d, z1d, 5) 
initial_y             = DataAnalysis.find_closest_vic2d(initial_data,  x1d, z1d, 6) 
deformed_u            = DataAnalysis.find_closest_vic2d(deformed_data, x1d, z1d, 7) 
deformed_v            = DataAnalysis.find_closest_vic2d(deformed_data, x1d, z1d, 8)

# calculate new coordinates
new_x                 = initial_x + deformed_u
new_z                 = initial_y + deformed_v

# error calculation
z_3turn_from_exp      = z1d - 2.95
x_diff                = x1d              - new_x
z_diff                = z_3turn_from_exp - new_z
dist                  = np.sqrt( x_diff**2 + z_diff**2 )
print( 'max x error        = ' + str(np.max(x_diff)) + ' mm')
print( 'max z error        = ' + str(np.max(z_diff)) + ' mm')
print( 'max distance error = ' + str(np.max(dist)) + ' mm')

# plots for explanation
plt.close('all')
xlimits=[-10,10]
ylimits=[-10,10]

plt.figure(1)
plt.xlabel('x')
plt.ylabel('z')
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.plot(x1d,               z1d, 'xk', label='macro coordinates')
plt.legend(numpoints=1)

plt.figure(2)
plt.xlabel('x')
plt.ylabel('z')
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.plot(x1d,               z1d, 'xk', label='initial macro coordinates')
plt.plot(x1d,  z_3turn_from_exp, 'or', label='updated macro coordinates')
plt.legend(numpoints=1)

plt.figure(3)
plt.xlabel('x')
plt.ylabel('z')
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.plot(x1d,               z1d, 'xk', label='initial macro coordinates')
plt.plot(new_x,           new_z, 'ob', label='vic2d updated coordinates')
plt.legend(numpoints=1)

plt.figure(4)
plt.xlabel('x')
plt.ylabel('z')
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.plot(x1d,  z_3turn_from_exp, 'or', label='updated macro coordinates')
plt.plot(new_x,           new_z, 'ob', label='vic2d updated coordinates')
plt.legend(numpoints=1)