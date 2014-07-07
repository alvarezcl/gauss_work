# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 23:30:16 2014

@author: luis
"""
## This file contains a library of useful functions
## suited for obtaining values drawn from 1D and 2D
## gaussian distributions and analytic representations
## of them as well.

import numpy as np
import gauss
from scipy.stats import multivariate_normal

# Function returns a scalar evaluated at x with gaussian function.
def gauss_1d(x,mean,variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-(x-mean)**2/(2*variance))

# Define model function to be used to fit to the data above:
def gaussFunc(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def inverseGaussFun(x, *p):
    A, mu, sigma = p
    return np.sqrt(2*sigma**2*np.log(A/x)) + mu

# Draw from a 1D gaussian using built-in Numpy random normal dist function
# Feeding in mean, width, and num of points we want to throw
def draw_1dGauss(mean,var,N):
    scatter = np.random.normal(mean,np.sqrt(var),N)
    scatter = np.sort(scatter)
    return scatter

# Return the sum of two gaussians that vary a long differing dimensions.
def sum_gauss(x,mean1,var1,y,mean2,var2):
    return gauss_1d(x,mean1,var2) + gauss_1d(y,mean2,var2)

# Return the two-dimensional gaussian with mean array and cov matrix values.
def mult_gaussStats(x,y,mean,cov):
    assert cov[0][1] is cov[1][0]
    assert cov[0][1] <= cov[0][0]*cov[1][1]
    X,Y = np.meshgrid(x,y)
    pos = np.emptry(X.shape + (2,))
    pos[:,:,0] = X; pos[:,:,1] = Y
    rv = multivariate_normal(mean,cov)
    return rv, X, Y

# Return a gaussian distribution at an angle alpha from the x-axis
# from astroML
def mult_gaussFun(A,x,y,x0,y0,varx,vary,cov,rho,alpha):
    X,Y = np.meshgrid(x,y)
    assert rho != 1
    a = 1/(2*(1-rho**2)) # Normalization for Unity
    Z = A*np.exp(-a*((X-x0)**2/(varx)+(Y-y0)**2/(vary)-(2*rho/(np.sqrt(varx*vary)))*(X-x0)*(Y-y0)))
    return X,Y,Z
    
# Alternate parametrization from Wikipedia   
def mult_gaussFunAlt(A,x,y,x0,y0,varx,vary,alpha):
    assert alpha >= -np.pi/2
    assert alpha <= np.pi/2
    X,Y = np.meshgrid(x,y)
    a = np.cos(alpha)**2/(2*varx) + np.sin(alpha)**2/(2*vary)
    b = -np.sin(2*alpha)/(4*varx) + np.sin(2*alpha)/(4*vary)
    c = np.sin(alpha)**2/(2*varx) + np.cos(alpha)**2/(2*vary)
    Z = A*np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    return X,Y,Z

# Convert variances from pincipal axes coordinates to variances in x-y        
def transform_Var(var_p1,var_p2,alpha):
    assert alpha >= -np.pi/2
    assert alpha <= np.pi/2
    varx = var_p1*np.cos(alpha)**2 + var_p2*np.sin(alpha)**2
    vary = var_p1*np.sin(alpha)**2 + var_p2*np.cos(alpha)**2
    cov = (var_p1-var_p2)*np.sin(alpha)*np.cos(alpha)
    rho = cov/(np.sqrt(varx*vary))
    return varx,vary,cov,rho

# This function returns information on variances and covariance,
# in order to plot gaussians at any centroid oriented at an angle, alpha.
def mult_gaussPrincipal(A,x,y,x0,y0,var_p1,var_p2,alpha):
    """
    Parameters:
        A: Amplitude of Gaussian
        x: x Domain
        y: y Domain
        x0,y0: Centroid of Gaussian
        var_p1: Variance in semi-major/minor axis in P1 domain
        var_p2: Variance in semi-major/minor axis in P2 domain
        alpha: Orientation of distribution with respect to x
        Returns:
            X,Y,Z: Domain and Gaussian Functional Values in Z
            varx,vary,cov,rho: Variance in x,y, and the Covariance and Correlation Co.
            P1,P2,Zp: Domain and Gaussian Functional Values in Principal Axes Frame  
    """
    P1,P2,Zp = gauss.mult_gaussFunAlt(A,x,y,0,0,var_p1,var_p2,0)
    varx,vary,cov,rho = gauss.transform_Var(var_p1,var_p2,alpha)
    X,Y,Z = gauss.mult_gaussFun(A,x,y,x0,y0,varx,vary,cov,rho,alpha)
    return X,Y,Z,varx,vary,cov,rho,P1,P2,Zp
