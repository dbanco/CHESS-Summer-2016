# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:39:19 2016

@author: kswartz92
"""
# DIC
DataReader            = importlib.reload(DataReader)
DataAnalysis          = importlib.reload(DataAnalysis)

dic_dir               = '/media/kswartz92/Swartz/Chess/Al7075insitu/DIC/'
file_names            = ['dic_4536.csv', 'dic_4537.csv', 'dic_4538.csv', 'dic_4539.csv', 'dic_4540.csv']

dic_data              = []
for i_file in range(len(file_names)):
    step_data             = DataReader.vic2d_reader(dic_dir+file_names[i_file])
    dic_data.append(step_data)
dic_data              = np.array(dic_data, dtype=float)

# shift coordinates so origin is at sample center
x_center              = 0.16           # mm
y_center              = 2.11           # mm
dic_data[:,:,5]      -= x_center  
dic_data[:,:,6]      -= y_center

# for each detector image data point, find closest vic2d data points
dic_exx               = []
for i_step in range(1, len(file_names)):
    step_exx              = DataAnalysis.find_closest_vic2d(dic_data[i_step], x1d, z1d, 9)  # index 9 is exx
    dic_exx.append(step_exx)
dic_exx               = np.array(dic_exx, dtype=float)

#%%    dic strain scatter plot
def dic_scatter_plot(x, y, plot_var, skip): 
    plt.close('all')
    plt.scatter(x[skip:-skip], y[skip:-skip], c=plot_var[skip:-skip], s=20, vmin=np.min(plot_var[skip:-skip]), vmax=np.max(plot_var[skip:-skip]))
    plt.colorbar()