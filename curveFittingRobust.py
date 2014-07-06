# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:46:43 2014

@author: luis
"""

## This file draws from a 2D Gaussian and obtains parameters 
## from the scatter and convert into variances in x and y 
## (including cov).

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import format

# Beginning from an analytic expression, plot histogram
mean = 0; var = 1; sigma = np.sqrt(var); N = 10000
A = 1/np.sqrt((2*np.pi*var))
points = gauss.draw_1dGauss(mean,var,N)

# Histogram Setup
bin_size = 0.1; min_edge = mean-6*sigma; max_edge = mean+9*sigma
Nn = (max_edge-min_edge)/bin_size; Nplus1 = Nn + 1
bins = np.linspace(min_edge, max_edge, Nplus1)

# Return info from primary gaussian
hist, bin_edges = np.histogram(points,bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    
# Now draw from a secondary gaussian
meanp = 2; varp = 1; sigmap = np.sqrt(varp); Np = N
pointsp = gauss.draw_1dGauss(meanp,varp,Np)    
histp, bin_edgesp = np.histogram(pointsp,bins)
bin_centresp = (bin_edgesp[:-1] + bin_edgesp[1:])/2
        
# Now the sum
points_tot = np.array(points.tolist()+pointsp.tolist())
hist_tot, bin_edges_tot = np.histogram(points_tot,bins)
bin_centres_tot = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2

# Initial seed
p0 = [A, mean, sigma]

# Estimation of parameters
coeff, var_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=p0)
    
# Get the fitted curve and plot
hist_fit = gauss.gaussFun(bin_centres, *coeff)
p = plt.plot(bin_centres,hist_fit,'k',label=format.info_gaussian('Fit',coeff[0],coeff[1],np.abs(coeff[2])))
    
# Error on the estimates
error_parameters = np.sqrt(np.array([var_matrix[0][0],var_matrix[1][1],var_matrix[2][2]]))
plt.figure(1); plt.title('Curve Fit of a sum of Gaussians')
p1 = plt.hist(points_tot,bins,facecolor='blue',label='Sum')
p2 = plt.hist(points,bins,facecolor='green',label=format.info_gaussian('Primary',max(hist),mean,sigma))
p3 = plt.hist(pointsp,bins,facecolor='red',label=format.info_gaussian('Secondary',max(histp),meanp,sigmap))
plt.legend(prop={'size':10}); plt.ylabel('Frequency'); plt.xlabel('Values')
plt.show()    