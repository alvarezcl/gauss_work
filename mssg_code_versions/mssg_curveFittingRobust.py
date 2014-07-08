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
import plotutils
import ipdb

# Beginning from an analytic expression, throw points for a Gaussian
# (minimal comments here, see them in 1D_Gaussian_fit code for this part, it does the
# same thing here)
mean = 0; var = 1; sigma = np.sqrt(var); N = 10000
A = 1/np.sqrt((2*np.pi*var))
points = gauss.draw_1dGauss(mean,var,N) # Throw points from function using utility from gauss class

# Histogram Setup
lowsigmawidth = 6
hisigmawidth = 9
bin_size = 0.1; min_edge = mean- lowsigmawidth *sigma; max_edge = mean+ hisigmawidth*sigma
Nn = (max_edge-min_edge)/bin_size; Nplus1 = Nn + 1
# These are all the bin edges from min_edge to  max_edge -- they will be the same for the primary, secondary, and total Gaussians 
bins = np.linspace(min_edge, max_edge, Nplus1) 

# Use Numpy to do the binning
#    hist is the freq hist in nbins, bin_edges are the left edge coords of these bins
hist, bin_edges = np.histogram(points,bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    
# Now draw from a secondary gaussian, same as above, all subscript "_s" means secondary
mean_s = 2; var_s = 1; sigma_s = np.sqrt(var_s); N_s = N
points_s = gauss.draw_1dGauss(mean_s,var_s,N_s)    
hist_s, bin_edges_s = np.histogram(points_s,bins)
bin_centres_s = (bin_edges_s[:-1] + bin_edges_s[1:])/2
        
# Now the sum
points_tot = np.array(points.tolist()+points_s.tolist())
hist_tot, bin_edges_tot = np.histogram(points_tot,bins)
bin_centres_tot = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2

# Initial seed for fit  (putting the input params for the Gaussian into a vec)
paramvec = [A, mean, sigma]

# Estimation of parameters
params_out, covar_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=paramvec)
    
# Get the fitted curve and plot
hist_fit = gauss.gaussFun(bin_centres, *params_out)
p = plt.plot(bin_centres,hist_fit,'k',label=plotutils.info_gaussian('Fit',params_out[0],params_out[1],np.abs(params_out[2])))
    
# Error on the estimates
error_parameters = np.sqrt(np.array([covar_matrix[0][0],covar_matrix[1][1],covar_matrix[2][2]]))
plt.figure(1); plt.title('Curve Fit of a sum of Gaussians')
p1 = plt.hist(points_tot,bins,facecolor='blue',label='Sum')
p2 = plt.hist(points,bins,facecolor='green',label=plotutils.info_gaussian('Primary',max(hist),mean,sigma))
p3 = plt.hist(points_s,bins,facecolor='red',label=plotutils.info_gaussian('Secondary',max(hist_s),mean_s,sigma_s))
plt.legend(prop={'size':10}); plt.ylabel('Frequency'); plt.xlabel('Values')
plt.show()    

ipdb.set_trace()
