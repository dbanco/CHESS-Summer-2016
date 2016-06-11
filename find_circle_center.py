"""
Created on Sat Jun 11 16:08:40 2016

@author: kswartz92
"""
import numpy as np
import numpy.linalg as la
#############################################################################    
def Newton(x0, input_coords, tol=1e-12):
    ####################################################
    def print_x(x):
        print('xc = '+'%11.8f'%x[0], 'yc = '+'%11.8f'%x[1], 'r = '+'%11.8f'%x[2]) 
    ####################################################
    def f(x):    # function evaluation
        xc, yc, r = x                                # x-center, y-center, radius (x1,x2,x3)
        f         = np.zeros(3)
        f[0]      = (x1-xc)**2 + (y1-yc)**2 - r**2   # f1
        f[1]      = (x2-xc)**2 + (y2-yc)**2 - r**2   # f2
        f[2]      = (x3-xc)**2 + (y3-yc)**2 - r**2   # f3
        return f 
    ####################################################
    def J(x):    # Jacobian evaluation
        xc, yc, r = x                                # x-center, y-center, radius (x1,x2,x3) 
        J         = np.zeros((3,3))
        J[0,0]    = -2*x1 + 2*xc                     # df1/dx1
        J[0,1]    = -2*y1 + 2*yc                     # df1/dx2
        J[0,2]    = -2*r                             # df1/dx3
        J[1,0]    = -2*x2 + 2*xc                     # df2/dx1
        J[1,1]    = -2*y2 + 2*yc                     # df2/dx2
        J[1,2]    = -2*r                             # df2/dx3
        J[2,0]    = -2*x3 + 2*xc                     # df3/dx1
        J[2,1]    = -2*y3 + 2*yc                     # df3/dx2
        J[2,2]    = -2*r                             # df3/dx3
        return J 
    ####################################################
        
    x = x0.copy()                        # initial guess
    print_x(x)
    
    while la.norm(f(x)) > tol:       # check for convergence
        h  = la.solve(J(x), -f(x))       # calculate Newton step
        x += h                           # update unknown vector
        print_x(x)
    
    return x
#############################################################################  
# User Inputs (edge of circle coordinates)
x1 = 5.145
y1 = 0.189

x2 = 0.234
y2 = 5.675

x3 = -5.125
y3 = -0.455
#############################################################################
input_coords = np.array([x1, y1, x2, y2, x3, y3])   
x0           = np.array([x1, y1, 5.0])
x            = Newton(x0, input_coords)