# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:58:28 2014

@author: luis
"""
# This script will produce curve fitting in one dimension.

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import gauss

# Beginning in one dimension:
mean = 0; Var = 1; N = 1000
scatter = gauss.draw_1dGauss(mean,Var,N)
mu,sigma = norm.fit(scatter)
## Unbinned Fit
#xdata = np.linspace(-5,5,N)
#pop, pcov = curve_fit(gauss.gauss_1d,xdata,scatter)

binNum = 50
n, binEdge, patches = plt.hist(scatter,binNum,facecolor='green')
y = 2*max(n)*mlab.normpdf(binEdge,mu,sigma)
l = plt.plot(binEdge,y,'r',lw=2)

plt.xlabel('Value')
plt.ylabel('Occurrences')
plt.grid(True)
plt.title('Histogram of Samples Drawn From Gaussians and Curve Fits')

# Add another Gaussian
meanp = 4; varp = 1; Np = 1000
scatterp = gauss.draw_1dGauss(meanp,varp,Np)
mup,sigmap = norm.fit(scatterp)
n_p,binEdgep,patches = plt.hist(scatterp,binNum,facecolor='blue')
yp = 2*max(n_p)*mlab.normpdf(binEdgep,mup,sigmap)
m = plt.plot(binEdgep,yp,'k',lw=2)

plt.show()
plt.figure(2)

scatterTot = np.array(scatter.tolist()+scatterp.tolist())
n_sum, binEdge_sum, patches = plt.hist(scatterTot,2*binNum,facecolor='red')
y_tot = np.array(yp.tolist() + y.tolist())

#plt.plot(binEdge_sum,scatterTot,'k')
plt.title('Histogram of Sum of Gaussians')
plt.show()
