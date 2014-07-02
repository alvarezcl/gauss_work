# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:46:43 2014

@author: luis
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import gauss
import sys

# Beginning from an analytic expression, plot histogram
mean = 0; var = 1; sigma = np.sqrt(var); N = 10000
A = 1/np.sqrt((2*np.pi*var))
points = gauss.draw_1dGauss(mean,var,N)
bins = 100
hist, bin_edges = np.histogram(points,bins,density=True)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    
# Now draw from a minor gaussian
meanp = 0; varp = 1; sigmap = np.sqrt(varp); Np = N
pointsp = gauss.draw_1dGauss(meanp,varp,Np)
A_coeff = 1/2
pointsp = pointsp*A_coeff    
histp, bin_edgesp = np.histogram(pointsp,bins,density=True)
bin_centresp = (bin_edgesp[:-1] + bin_edgesp[1:])/2
    
plt.figure(1); plt.title('Separate Histograms')
plt.hist(points,bins,facecolor='green')
plt.hist(pointsp,bins,facecolor='red')
    
# Now the sum
points_tot = np.array(points.tolist()+pointsp.tolist())
bins_tot = 2*bins
hist_tot, bin_edges_tot = np.histogram(points_tot,bins_tot,density=True)
bin_centres_tot = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2
plt.figure(2); plt.title('Combined Histogram')
plt.hist(points_tot,bins_tot)    
    
p0 = [A, mean, sigma]
    
coeff, var_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=p0)
    
# Get the fitted curve
hist_fit = gauss.gaussFun(bin_centres, *coeff)
    
# Error on the estimates
error_parameters = np.sqrt(np.array([var_matrix[0][0],var_matrix[1][1],var_matrix[2][2]]))
    