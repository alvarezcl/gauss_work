# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:46:43 2014

@author: luis
"""

## This file draws from two gaussian distributions and attempts
## to find the best fit parameters. Useful for deblending.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss

# Draw from the major gaussian. Note the number N. It is
# the main parameter in obtaining your estimators.
mean = 0; var = 1; sigma = np.sqrt(var); N = 100
A = 1/np.sqrt((2*np.pi*var))
points = gauss.draw_1dGauss(mean,var,N)
bins = N/10
hist, bin_edges = np.histogram(points,bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    
# Now draw from a minor gaussian. Note Np
meanp = 0; varp = 1/4; sigmap = np.sqrt(varp); Np = 1000-N
pointsp = gauss.draw_1dGauss(meanp,varp,Np)
binsp = Np/10
histp, bin_edgesp = np.histogram(pointsp,bins)
bin_centresp = (bin_edgesp[:-1] + bin_edgesp[1:])/2
# Plot the histogram of each on one plot.
plt.figure(1); plt.title('Major Gaussian and Minor Gaussian')
plt.hist(pointsp,bins,facecolor='red',normed=True)
plt.hist(points,bins,facecolor='green',normed=True)

    
# Now implement the sum of the draws by concatenating the two arrays.
points_tot = np.array(points.tolist()+pointsp.tolist())
bins_tot = len(points_tot)/20
hist_tot, bin_edges_tot = np.histogram(points_tot,bins_tot,density=True)
bin_centres_tot = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2
# Plot the histogram of the sum
plt.figure(2); plt.title('Combined Histogram')
plt.hist(points_tot,bins_tot,normed=True)    
    
# Initial guess
p0 = [A, mean, sigma]

# Result of the fit
coeff, var_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=p0)
    
# Get the fitted curve
hist_fit = gauss.gaussFun(bin_centres, *coeff)
    
# Error on the estimates
error_parameters = np.sqrt(np.array([var_matrix[0][0],var_matrix[1][1],var_matrix[2][2]]))