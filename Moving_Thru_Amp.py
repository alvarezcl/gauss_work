# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:46:05 2014

@author: luis
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import plotutils

## Draw from 2 1D Gaussians and attempt to find 
## the parameters of the sum of their histograms 
## and notes the change in the estimates vs
## the true distributions sampled from.
## Primarily for change in sigma.

# Value for normed histograms: False is unnormed, True is normed
norm = False
# Total possible samples to draw
N = 10000
# Amplitude ratios for each gaussian and text
Arp = 1.0; Ars = 1.0;
# Loops for iterations
loops = 10
start = 0
increment = 0.1

# Empty arrays to insert data within loop
ind_var = []
stats = []
error = []

for i in xrange(0,loops+1):

    Ars = (start+i*increment)
    
    # Draw from the primary gaussian. Note the number N_p 
    # and how it takes into account Amplitude ratio,
    # N, and sigma. It is
    # the main parameter in obtaining your estimators.
    mean = 0; sigma = 1; var = sigma**2; N_p = N*Arp*sigma
    points = gauss.draw_1dGauss(mean,var,N_p)
    
    # Now draw from the secondary gaussian. Note Np and
    # loop parameter within the changing value of choice.
    # Adjust number of points thrown such that the amplitude
    # is preserved. Sigma prime changes, therefore preserving
    # the number of points increases the density of points/bin.
    # One remedy is to scale points thrown 
    meanp = 0; sigmap = 1; varp = sigmap**2; N_s = N*Ars*sigmap
    pointsp = gauss.draw_1dGauss(meanp,varp,N_s)    
    
    # Histogram parameters for each distribution
    bin_size = 0.1 
    min_edge = mean-9*sigma 
    max_edge = meanp+9*sigma
    Nn = (max_edge-min_edge)/bin_size; Nplus1 = Nn + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)
    
    # Histogram the first samples
    hist,bin_edges = np.histogram(points,bin_list,density=norm)
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.0
    width = bin_edges[1]-bin_edges[0]
    
    # Histogram the second samples
    histp,bin_edgesp = np.histogram(pointsp,bin_list,density=norm)
    bin_centersp = (bin_edgesp[:-1] + bin_edgesp[1:])/2.0
    widthp = bin_edgesp[1]-bin_edgesp[0]
    
    # Initial guess
    p0 = ((max(hist)+max(histp))/2.0,(mean+meanp)/2.0,(sigma+sigmap)/2.0)
    
    # Now histogram the concatenated sum
    points_sum = np.array(points.tolist()+pointsp.tolist())
    bins_sum = bin_list
    hist_sum,bin_edges_sum = np.histogram(points_sum,bins_sum,density=norm)
    bin_centers_sum = (bin_edges_sum[:-1] + bin_edges_sum[1:])/2.0

    # Iterate through different estimates in order to 
    # weed out outliers.
    numTrials = 100; 
    sig_est = []; sig_error = []
    mean_est = []; mean_error = []
    amp_est = []; amp_error = []
    tol = 1e5
    
    count = 0
    while count < numTrials:
        p_cat, var_matrix = curve_fit(gauss.gaussFun, bin_centers_sum, hist_sum, p0=p0)
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
    plt.figure(1); plt.title('Curve Fit of a sum of Gaussians')
    x = np.linspace(min_edge,max_edge,1000)
    plt.plot(x,gauss.gaussFun(x,*(amp,mean_est,sig)),label=plotutils.info_gaussian('Fit',amp,mean_est,sig))
    p1 = plt.hist(points_sum,bins_sum,facecolor='blue',label='Sum')
    p2 = plt.hist(points,bin_list,facecolor='green',label=plotutils.info_gaussian('Primary',max(hist),mean,sigma))
    p3 = plt.hist(pointsp,bin_list,facecolor='red',label=plotutils.info_gaussian('Secondary',max(histp),meanp,sigmap))
    plt.legend(prop={'size':10}); plt.ylabel('Frequency'); plt.xlabel('Values')
    plt.show()


    # Stats of interest:
    # Independent Parameters being changed: sigmap,meanp,max(histp)
    ind_var.append(Ars)

    # Parameters returned by curve_fit, with outliers taken out
    stats.append([sig,mean_est,amp])

    # Errors on those parameters
    error.append([sig_error,mean_error,amp_error]) 

# Convert to numpy arrays for plotting purposes
ind_var = np.array(ind_var) 
stats = np.array(stats) 
error = np.array(error)

# Plot information
font = 18
font_title = 14

p2 = plt.figure(2); plt.xlabel(r'$A_{cs}$',fontsize=font); plt.ylabel('$Fit\/\sigma_{est}$',fontsize=font)
plt.scatter(ind_var,stats[:,0]) 
plt.errorbar(ind_var,stats[:,0],yerr=error[:,0])
plt.suptitle(plotutils.info_gaussian_Acp('Primary',Arp,mean,sigma)+ '\n' +
             'Secondary: $A_{cs}\/\epsilon[%.1f,%.1f]$, $\mu=%.2f$, $\sigma=%.2f$'%(start,loops*increment,meanp,sigmap),fontsize=font_title)
plt.show()

p3 = plt.figure(3); plt.xlabel(r'$A_{cs}$',fontsize=font); plt.ylabel('$Fit\/\mu_{est}$',fontsize=font)
plt.scatter(ind_var,stats[:,1]) 
plt.errorbar(ind_var,stats[:,1],yerr=error[:,1])
plt.suptitle(plotutils.info_gaussian_Acp('Primary',Arp,mean,sigma)+ '\n' +
             'Secondary: $A_{cs}\/\epsilon[%.1f,%.1f]$, $\mu=%.2f$, $\sigma=%.2f$'%(start,loops*increment,meanp,sigmap),fontsize=font_title)
plt.show()

p4 = plt.figure(4); plt.xlabel(r'$A_{cs}$',fontsize=font); plt.ylabel('$Fit\/A_{est}$',fontsize=font)
plt.scatter(ind_var,stats[:,2]) 
plt.errorbar(ind_var,stats[:,2],yerr=error[:,2])
plt.suptitle(plotutils.info_gaussian_Acp('Primary',Arp,mean,sigma)+ '\n' +
             'Secondary: $A_{cs}\/\epsilon[%.1f,%.1f]$, $\mu=%.2f$, $\sigma=%.2f$'%(start,loops*increment,meanp,sigmap),fontsize=font_title)
plt.show()