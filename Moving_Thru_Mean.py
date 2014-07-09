# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 11:46:43 2014

@author: luis
"""

## This script loops through different means for primary
## and secondary gaussians to determine best fit params
## primarily changing the mean.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import plotutils

width = 0.5
loops = 21
ind_var = []
stats = []
error = []

N = 10000
Arp = 1.0
Ars = 1.0
start_point = -5

for i in xrange(0,loops):
    
    # Draw from primary gaussian
    mean = 0; sigma = 1; var = sigma**2
    N_p = N*Arp*sigma
    points = gauss.draw_1dGauss(mean,var,N_p)
    
    # Now draw from a secondary gaussian
    meanp = mean + (start_point + width*i); sigmap = 1; varp = sigma**2
    N_s = N*Ars*sigmap
    pointsp = gauss.draw_1dGauss(meanp,varp,N_s)
    
    # Histogram Setup
    bin_size = 0.1; min_edge = mean-10*sigma; max_edge = mean+14*sigma
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
    p0 = ((max(hist)+max(histp))/2.0,(mean+meanp)/2.0,(sigma+sigmap)/2.0)
            
    # Iterate through different estimates in order to 
    # weed out outliers.
    numTrials = 100; 
    sig_est = []; sig_error = []
    mean_est = []; mean_error = []
    amp_est = []; amp_error = []
    tol = 1e9
    
    count = 0
    while count < numTrials:
        p_cat, var_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=p0)
        diag_errors = (var_matrix[0][0],var_matrix[1][1],var_matrix[2][2])
        # Only allow low-error estimates, toss outliers.       
        if diag_errors[0] < tol and diag_errors[1] < tol and diag_errors[2] < tol:
            sig_est.append(p_cat[2]) 
            sig_error.append(np.sqrt(diag_errors[2]))
            mean_est.append(p_cat[1]) 
            mean_error.append(np.sqrt(diag_errors[1]))
            amp_est.append(p_cat[0]) 
            amp_error.append(np.sqrt(diag_errors[0]))
            count = count + 1
            
    # Obtain the mean estimates of each estimator
    sig = np.abs(np.mean(sig_est)) 
    mean_est = np.mean(mean_est) 
    amp = np.mean(amp_est) 

    # Obtain the mean errors of each estimator
    sig_error = np.mean(sig_error) 
    mean_error = np.mean(mean_error)
    amp_error = np.mean(amp_error)

    # Plot
#    plt.figure(1); plt.title('Curve Fit of a sum of Gaussians')
#    x = np.linspace(min_edge,max_edge,1000)
#    plt.plot(x,gauss.gaussFun(x,*(amp,mean_est,sig)),label=plotutils.info_gaussian('Fit',amp,mean_est,sig))
#    p1 = plt.hist(points_tot,bins,facecolor='blue',label='Sum')
#    p2 = plt.hist(points,bins,facecolor='green',label=plotutils.info_gaussian('Primary',max(hist),mean,sigma))
#    p3 = plt.hist(pointsp,bins,facecolor='red',label=plotutils.info_gaussian('Secondary',max(histp),meanp,sigmap))
#    plt.legend(prop={'size':10}); plt.ylabel('Frequency'); plt.xlabel('Values')
#    plt.show()
#    
    # Append to data containers
    ind_var.append(meanp-mean)
    stats.append([mean_est,sig,amp])
    error.append([mean_error,sig_error,amp_error])

# Convert to NP arrays
ind_var = np.array(ind_var); stats = np.array(stats); error = np.array(error)

font = 20
font_title = 14
p2 = plt.figure(2); plt.xlabel('$\Delta\mu=\mu_s - \mu_p$',fontsize=font); plt.ylabel('$Fit\/\mu_{est}$',fontsize=font)
plt.scatter(ind_var,stats[:,0]) 
plt.errorbar(ind_var,stats[:,0],yerr=error[:,0])
plt.suptitle(plotutils.info_gaussian_Acp('Primary',Arp,mean,sigma)+ '\n' +
             'Secondary: $A_{cs}=%.2f$, $\mu\/\epsilon [-5,5]$, $\sigma=%.2f$'%(Ars,sigma),fontsize=font_title)
plt.show()

p3 = plt.figure(3); plt.xlabel('$\Delta\mu=\mu_s - \mu_p$',fontsize=font); plt.ylabel('$Fit\/\sigma_{est}$',fontsize=font)
plt.scatter(ind_var,stats[:,1]) 
plt.errorbar(ind_var,stats[:,1],yerr=error[:,1])
plt.suptitle(plotutils.info_gaussian_Acp('Primary',Arp,mean,sigma)+ '\n' +
             'Secondary: $A_{cs}=%.2f$, $\mu\/\epsilon [-5,5]$, $\sigma=%.2f$'%(Ars,sigma),fontsize=font_title)
plt.show()

p4 = plt.figure(4); plt.xlabel('$\Delta\mu=\mu_s - \mu_p$',fontsize=font); plt.ylabel('$Fit\/A_{est}$',fontsize=font)
plt.scatter(ind_var,stats[:,2])
plt.errorbar(ind_var,stats[:,2],yerr=error[:,2])
plt.suptitle(plotutils.info_gaussian_Acp('Primary',Arp,mean,sigma)+ '\n' +
             'Secondary: $A_{cs}=%.2f$, $\mu\/\epsilon [-5,5]$, $\sigma=%.2f$'%(Ars,sigma),fontsize=font_title)
plt.show()