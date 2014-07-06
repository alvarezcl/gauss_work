# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:21:56 2014

@author: luis
"""
# This file produces multi-variate distributions and plots them in terms
# of points, histograms, and 3D bar graphs.

import numpy as np
import matplotlib.pyplot as plt
import gauss
import format

## Data set 1

# Produce a number of points in x-y from 1 distribution. 
mean = [0,0]
cov = [[3,0],[0,1]] 
N = 3000
x,y = np.random.multivariate_normal(mean,cov,N).T
plt.figure(1)
plt.plot(x,y,'x')

# Produce a histogram of data set 1.
bins = 30
X,Y,H,hist = format.hist_2dPlane(x,y,bins)

# Produce a bar graph of data set 1
bar1 = format.hist_3dBar(x,y,bins)

## Data set 2

# Produce a number of points in x-y from another distribution.
mean = [-5,5]
cov = [[2,1],[1,2]] 
xp,yp = np.random.multivariate_normal(mean,cov,N).T
plt.figure(4)
plt.plot(xp,yp,'rx') 

# Produce a histogram of data set 2.
Xp,Yp,Hp,histp = format.hist_2dPlane(xp,yp,bins)

# Produce a bar graph of data set 2
bar2 = format.hist_3dBar(xp,yp,bins)


## Combined Data set
x_sum = np.array(x.tolist()+xp.tolist())
y_sum = np.array(y.tolist()+yp.tolist())
X_tot,Y_tot,H_tot,hist_tot = gauss.hist_2dPlane(x_sum,y_sum,bins)
bar_tot = format.hist_3dBar(x_sum,y_sum,bins)