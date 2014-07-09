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
import mssg_gauss
import plotutils

#### Beginning from an analytic expression, plot histogram

# Input vars
mean = 0; var = 1; sigma = np.sqrt(var); N = 10000

# Note we choose a normalized Gaussian with unit area underneath, amplitude determined fully by the FWHM
A = 1/np.sqrt((2*np.pi*var))

# Throw the points from the function using the utility function from the gauss class
points = mssg_gauss.draw_1dGauss(mean,var,N)

# Set num of bins
nbins = N/100

# Use Numpy to do the binning
#    hist is the freq hist in nbins, bin_edges are the left edge coords of these bins
hist, bin_edges = np.histogram(points,nbins) 

# Vars we'll need below
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2 # Take avg to get x-coords of bin centers
paramvec = [A,mean,sigma]                        # Put the input params for the Gaussian into a vec

#### Now we fit to this 1D Gaussian histo
# Curve Fit Estimators
# --- curve_fit works by taking the bin_centers, the thrown points in the hist, and the seeds for the fit function in paramvec, then fitting a standard Gaussian function defined in the gauss class to them
# --- The outputs are the params, and a covariance mat (which we don't use later in this specific code)
params_out, covar_matrix = curve_fit(mssg_gauss.gaussFunc, bin_centers, hist, p0=paramvec)

#### Plot the results

# Make the label for the Gaussian histo using input values
true_gauss = plotutils.info_gaussian('Input Gaussian',max(hist),mean,sigma)

# Make the actual histo (we won't use the name 'gausshisto' later again in this case)
gausshisto = plt.hist(points,nbins,facecolor='green',label= true_gauss)
# plt.legend() 

# Make the label for the fit Gaussian curve using output values
fit_gauss_label = plotutils.info_gaussian('Fit Gaussian',params_out[0],params_out[1],np.abs(params_out[2]))

# Make a vec of how many points we'll want to evaluate func at to make smooth curve, and the x domain
ncurvepoints = 1000
numsigmawidth = 5
x = np.linspace(mean-numsigmawidth*sigma,mean+numsigmawidth*sigma,ncurvepoints)

# Plot the fit Gaussian curve (lw = linewidth)
fitcurve = plt.plot(x,mssg_gauss.gaussFunc(x,*(params_out)),'r',lw=3,label=fit_gauss_label)

# Put the label into the legend box and fix the fontsize
plt.legend(prop={'size':12}) 

# Make the title and x-y axes labels
plt.title('Histogrammed Data with Curve Fit'); plt.ylabel('Frequency'); plt.xlabel('Value')

# Show the plot
plt.show()
