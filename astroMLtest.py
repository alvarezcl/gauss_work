# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:30:10 2014

@author: luis
"""

# This code draws upon a bi-variate distribution and samples 
# that distribution in order to place them on a 2-D plot.
# Then, a method provided by astroML estimates the necessary
# parameters from the data, and a gaussian surface/contour is
# is drawn upon the data.

import astropy as apy
import astroML as aml
from astroML.stats import fit_bivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import gauss

# Produce a number of points in x-y from 1 bi-variate gaussian
# distribution. 
mean = [0,0]
cov = [[3,2],[2,3]]
N = 1000
x,y = np.random.multivariate_normal(mean,cov,N).T
Z = np.array([x,y])

# Fit method provided by astroML
mean,sigma1,sigma2,alpha = fit_bivariate_normal(x,y)

# Plot bi-variate gaussian with parameters and determine
# variances in x-y. The covariance and variances should
# approximate the original distribution drawn from.
domain = 10;
xCord = np.linspace(-domain,domain,N/domain)
yCord = np.linspace(-domain,domain,N/domain)
A = 1; 
X,Y,Z,varx,vary,cov,rho,P1,P2,P = gauss.mult_gaussPrincipal(A,xCord,yCord,mean[0],mean[1],sigma1**2,sigma2**2,alpha)
plt.hold()
V = [cov, 2*cov]
plt.hold()
plt.xlabel('x'); plt.ylabel('y'); plt.title('Distribution of Points with Gaussian Fit (As Contours)')
plt.plot(x,y,'x'); plt.axis('equal')
plt.contour(X,Y,Z,zorder=10)
plt.show()
