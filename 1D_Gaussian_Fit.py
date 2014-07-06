# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:23:27 2014

@author: luis
"""

## This file produces histogram of samples drawn from gaussian and 
## plots gaussian curve fit corresponding to the samples.


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import format

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
true_gauss = format.info_gaussian('Primary',max(hist),mean,sigma)
p2 = plt.hist(points,bins,facecolor='green',label= true_gauss); plt.legend()
lab_gauss = format.info_gaussian('Fit',parameters[0],parameters[1],np.abs(parameters[2]))
x = np.linspace(mean-5*sigma,mean+5*sigma,1000)
p1, = plt.plot(x,gauss.gaussFun(x,*(parameters)),'r',lw=3,label=lab_gauss)
plt.legend(prop={'size':12})
plt.title('Histogrammed Data with Curve Fit'); plt.ylabel('Frequency'); plt.xlabel('Value')
plt.show()
