# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 23:30:16 2014

@author: luis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
#from scipy.stats import multivariate_normal

# Function returns a scalar evaluated at x with gaussian function.
def gauss_1d(x,mean,variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-(x-mean)**2/(2*variance))

# Return the sum of two gaussians that vary a long differing dimensions.
def sum_gauss(x,mean1,var1,y,mean2,var2):
    return gauss_1d(x,mean1,var2) + gauss_1d(y,mean2,var2)

# Return the two-dimensional gaussian with mean array and cov matrix values.
def mult_gauss(x,y,mean,cov):
    assert cov[0][1] is cov[1][0]
    assert cov[0][1] <= cov[0][0]*cov[1][1]
    X,Y = np.meshgrid(x,y)
    pos = np.emptry(X.shape + (2,))
    pos[:,:,0] = X; pos[:,:,1] = Y
    rv = multivariate_normal(mean,cov)
    return rv, X, Y

# Plot 3d surface and contours given any set of values, X, Y, Z    
def plot_3d(X,Y,Z):
    fig1 = plt.figure(1)
    ax1 = Axes3D(fig1)
    ax1.plot_surface(X,Y,Z)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    plt.show()
    
    plt.figure(2)
    im = pl.imshow(Z)
    cset = pl.contour(Z,[0.1,0.2,0.5,0.6,0.7])
    pl.clabel(cset,inline=True)
    pl.colorbar(im)
    plt.title('Contour')
    plt.show()

#def 

    