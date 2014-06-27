# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:21:56 2014

@author: luis
"""
# This file produces scatter plots from two gaussian distributions
# in order to simulate photon densities.

import numpy as np
import matplotlib.pyplot as plt

mean = 0
stdDev = 1
scatter = np.random.normal(mean,stdDev,1000)
hist = np.histogram(scatter,bins=10)
plt.hist(scatter,bins=10)
plt.show()

# Produce a number of points in x-y from 1 distribution. 
mean = [3,4]
cov = [[3,1],[1,3]] 
x,y = np.random.multivariate_normal(mean,cov,1000).T
plt.plot(x,y,'x'); plt.axis('equal'); plt.show()
Z = np.array([x,y])
# Produce a number of points in x-y from another distribution.
mean = [0,0]
cov = [[3,2],[2,3]] 
xp,yp = np.random.multivariate_normal(mean,cov,1000).T
plt.plot(xp,yp,'o'); plt.axis('equal'); plt.show()
Zp = np.array([xp,yp])

