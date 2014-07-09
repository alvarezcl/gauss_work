# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 16:16:59 2014

@author: luis
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import plotutils

## This script extracts the parameters of the sum of 
## two 1D gaussians. 

# Total number of points hat can be thrown.
N = 10000
# Amplitude coefficents for each distribution
Acp = 1.0
Acs = 1.0
    
# Draw from primary gaussian
mean_p = 0; sigma_p = 1; var_p = sigma_p**2
N_p = N*Acp
A_p = 1/(np.sqrt(2*np.pi*var_p))
points = gauss.draw_1dGauss(mean_p,var_p,N_p)
    
# Now draw from a secondary gaussian
mean_s = 3; sigma_s = 1; var_s = sigma_s**2
N_s = N*Acs
A_s = 1/(np.sqrt(2*np.pi*var_s))
pointsp = gauss.draw_1dGauss(mean_s,var_s,N_s)

# Histogram Setup
bin_size = 0.1; 
min_edge = mean_p-5*sigma_p; max_edge = mean_s+5*sigma_s
Nn = (max_edge-min_edge)/bin_size; Nplus1 = Nn + 1
bins = np.linspace(min_edge, max_edge, Nplus1)

# Individual hist
hist_p = np.histogram(points,bins)
hist_s = np.histogram(pointsp,bins)

# Now concatenate the two data arrays and get the 
# bin centers.
points_tot = np.array(points.tolist()+pointsp.tolist())
hist_tot, bin_edges_tot = np.histogram(points_tot,bins)
bin_centers = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2.0

# Initial seed
p0 = [A_p,mean_p,sigma_p,A_s,mean_s,sigma_s]

# Estimation of parameters
params, var_matrix = curve_fit(gauss.sum_gauss_2_1D, bin_centers, hist_tot, p0=p0)

# Plot histogram and curve
x = np.linspace(min_edge,max_edge,N)
plt.figure(1)
plt.hist(points_tot,bins,facecolor='green',label='Sum')
plt.plot(x,gauss.sum_gauss_2_1D(x,*params),'r',lw=3,label='Curve_Fit')
plt.xlabel('Values'); plt.ylabel('Frequency'); plt.legend()
plt.show()
