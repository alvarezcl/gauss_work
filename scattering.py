# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:21:56 2014

@author: luis
"""
# This file produces scatter plots from two gaussian distributions
# in order to simulate photon densities from different distributions.

import numpy as np
import matplotlib.pyplot as plt
import gauss
import pylab as pl

## Data set 1

# Produce a number of points in x-y from 1 distribution. 
mean = [0,0]
cov = [[3,0],[0,1]] 
N = 3000
x,y = np.random.multivariate_normal(mean,cov,N).T
plt.plot(x,y,'x')

# Produce a histogram of data set 1.
bins = 30
plt.figure(2)
X,Y,H,hist = gauss.hist_2dPlane(x,y,bins)

# Produce a bar graph of data set 1
bar1 = gauss.hist_3dBar(x,y,bins)

## Data set 2

# Produce a number of points in x-y from another distribution.
mean = [-5,5]
cov = [[2,1],[1,2]] 
xp,yp = np.random.multivariate_normal(mean,cov,N).T
plt.plot(xp,yp,'rx') 

# Produce a histogram of data set 2.
plt.figure(3)
Xp,Yp,Hp,histp = gauss.hist_2dPlane(xp,yp,bins)

# Produce a bar graph of data set 2
plt.figure(4)
bar2 = gauss.hist_3dBar(xp,yp,bins)


## Combined Data set
x_sum = np.array(x.tolist()+xp.tolist())
y_sum = np.array(y.tolist()+yp.tolist())
plt.figure(5)
X_tot,Y_tot,H_tot,hist_tot = gauss.hist_2dPlane(x_sum,y_sum,bins)
plt.figure(6)
bar_tot = gauss.hist_3dBar(x_sum,y_sum,bins)
