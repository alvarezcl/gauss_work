# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:23:27 2014

@author: luis
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import gauss

# Beginning from an analytic expression, plot histogram
mean = 0; var = 1; sigma = np.sqrt(var); N = 10000
A = 1/np.sqrt((2*np.pi*var))
points = gauss.draw_1dGauss(mean,var,N)
bins = N/100
hist, bin_edges = np.histogram(points,bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
p0 = [A,mean,sigma]
# Curve Fit Estimators
parameters, var_matrix = curve_fit(gauss.gaussFun, bin_centers, hist, p0=p0)
plt.hist(points,bins,facecolor='green')
x = np.linspace(mean-5*sigma,mean+5*sigma,1000)
p1, = plt.plot(x,gauss.gaussFun(x,*(parameters)),'r',lw=3)
plt.show()
