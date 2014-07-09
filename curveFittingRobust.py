# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:46:43 2014

@author: luis
"""

## This script draws from two gaussians and attempts
## to find the best-fit from the concatenated data.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import plotutils

N = 10000
Ap = 1.0
As = 0.5

# Beginning from an analytic expression, plot histogram
mean = 0; sigma = 1; var = sigma**2; N_p = N*Ap*sigma
points = gauss.draw_1dGauss(mean,var,N_p)
    
# Now draw from a secondary gaussian
meanp = 5; sigmap = 1; varp = sigmap**2; N_s = N*As*sigmap
pointsp = gauss.draw_1dGauss(meanp,varp,N_s)    

# Histogram Setup
bin_size = 0.1; min_edge = mean-5*sigma; max_edge = meanp+5*sigmap
Nn = (max_edge-min_edge)/bin_size; Nplus1 = Nn + 1
bins = np.linspace(min_edge, max_edge, Nplus1)

# Return info from primary gaussian
hist, bin_edges = np.histogram(points,bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.0

# Return info from secondary gaussian
histp, bin_edgesp = np.histogram(pointsp,bins)
bin_centresp = (bin_edgesp[:-1] + bin_edgesp[1:])/2.0
        
# Now the sum
points_tot = np.array(points.tolist()+pointsp.tolist())
hist_tot, bin_edges_tot = np.histogram(points_tot,bins)
bin_centres_tot = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2.0

# Initial seed
p0 = [(max(hist)+max(histp))/2.0, (mean+meanp)/2.0, (sigma+sigmap)/2.0]
#p0 = [max(histp),meanp,sigmap]
#p0 = [max(histp),mean,sigmap]

# Estimation of parameters
coeff, var_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=p0)
    
# Get the fitted curve and plot
hist_fit = gauss.gaussFun(bin_centres, *coeff)
p = plt.plot(bin_centres,hist_fit,'k',label=plotutils.info_gaussian('Fit',coeff[0],coeff[1],np.abs(coeff[2])))
    
# Error on the estimates
error_parameters = np.sqrt(np.array([var_matrix[0][0],var_matrix[1][1],var_matrix[2][2]]))
plt.figure(1); plt.title('Curve Fit of a sum of Gaussians')
p1 = plt.hist(points_tot,bins,facecolor='blue',label='Sum')
p2 = plt.hist(points,bins,facecolor='green',label=plotutils.info_gaussian('Primary',max(hist),mean,sigma))
p3 = plt.hist(pointsp,bins,facecolor='red',label=plotutils.info_gaussian('Secondary',max(histp),meanp,sigmap))
plt.legend(prop={'size':10}); plt.ylabel('Frequency'); plt.xlabel('Values')
plt.show()    
