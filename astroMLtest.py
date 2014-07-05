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
mean = [1,1]
cov = [[3,1],[1,3]]
N = 1000
x,y = np.random.multivariate_normal(mean,cov,N).T

# Fit method provided by astroML
mean_est,sigma1,sigma2,alpha = fit_bivariate_normal(x,y)

# Plot bi-variate gaussian with parameters and determine
# variances in x-y. The covariance and variances should
# approximate the original distribution drawn from.
domain = 10;
xCord = np.linspace(-domain,domain,N/domain)
yCord = np.linspace(-domain,domain,N/domain)
A = 1; 
X,Y,Z,varx,vary,cov,rho,P1,P2,P = gauss.mult_gaussPrincipal(A,xCord,yCord,mean[0],mean[1],sigma1**2,sigma2**2,alpha)
sigma_x = np.sqrt(varx); sigma_y = np.sqrt(vary)
V = [cov, 2*cov]
plt.figure(1)
plt.xlabel('x'); plt.ylabel('y'); plt.title('Distribution of Points with Gaussian Fit (As Contours)')
lab = r'Estimated Parameters: $\sigma_x=%.2f$,$\sigma_y=%.2f$,$\sigma_{xy}=%.2f$,$\alpha=%.2f$' % (sigma_x,sigma_y,cov,alpha)
p1, = plt.plot(x,y,'x'); plt.axis('equal')
p2 = plt.contour(X,Y,Z,zorder=10); plt.legend([p1],[lab],prop={'size':12})
plt.text(0,-4,'Estimated Mean = (%.2f,%.2f)'%(mean_est[0],mean_est[1]))
plt.figure(2); plt.title('Contours of Estimated Gaussian\n In Principal Axes Frame')
p3 = plt.contour(P1,P2,P); plt.xlabel('P1'); plt.ylabel('P2')
plt.show()
