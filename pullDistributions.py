# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:53:48 2014

@author: luis
"""

# This script calculates the mean and standard deviation for
# the pull distributions on the estimators that curve_fit returns

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss
import sys

numTrials = 15000
# Pull given by (a_j - a_true)/a_error)
error_vec_A = [] 
error_vec_mean = []
error_vec_sigma = []

# Loop to determine pull distribution
for i in xrange(0,numTrials):

    # Draw from gaussian distribution
    mean = 0; var = 1; sigma = np.sqrt(var); 
    N = 20000
    A = 1/np.sqrt((2*np.pi*var))
    points = gauss.draw_1dGauss(mean,var,N)
    bins = 2000
    hist, bin_edges = np.histogram(points,bins,density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    
    # Initial guess
    p0 = [5, 2, 4]
    
    coeff, var_matrix = curve_fit(gauss.gaussFun, bin_centres, hist, p0=p0)
    
    # Get the fitted curve
    hist_fit = gauss.gaussFun(bin_centres, *coeff)
        
    # Error on the estimates
    error_parameters = np.sqrt(np.array([var_matrix[0][0],var_matrix[1][1],var_matrix[2][2]]))
    
    # Obtain the error for each value: A,mu,sigma 
    A_std = (coeff[0]-A)/error_parameters[0]
    mean_std = ((coeff[1]-mean)/error_parameters[1])
    sigma_std = (np.abs(coeff[2])-sigma)/error_parameters[2]
    
    # Store results in container
    error_vec_A.append(A_std)
    error_vec_mean.append(mean_std)
    error_vec_sigma.append(sigma_std)

# Plot the distribution of each estimator        
plt.figure(1); plt.hist(error_vec_A,bins,normed=True); plt.title('Pull of A')
plt.figure(2); plt.hist(error_vec_mean,bins,normed=True); plt.title('Pull of Mu')
plt.figure(3); plt.hist(error_vec_sigma,bins,normed=True); plt.title('Pull of Sigma')

# Store key information regarding distribution 
mean_A = np.mean(error_vec_A); sigma_A = np.std(error_vec_A)    
mean_mu = np.mean(error_vec_mean); sigma_mu = np.std(error_vec_mean)    
mean_sigma = np.mean(error_vec_sigma); sigma_sig = np.std(error_vec_sigma)    
info = np.array([[mean_A,sigma_A],[mean_mu,sigma_mu],[mean_sigma,sigma_sig]])