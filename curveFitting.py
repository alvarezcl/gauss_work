# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:58:28 2014

@author: luis
"""
# This script will produce curve fitting in one dimension.

import numpy as np
import scipy as sc
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# Beginning in one dimension:
mean = 0; stdDev = 1; N = 100000
scatter = np.random.normal(mean,stdDev,N)
mu,sigma = norm.fit(scatter)

n, bins, patches = plt.hist(scatter,50,facecolor='green')
y = mlab.normpdf(bins,mu,sigma)
l = plt.plot(bins,y,'r--')

plt.xlabel('x-coord')
plt.ylabel('Occurrences')
plt.grid(True)
plt.show()
