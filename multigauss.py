# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 21:20:27 2014

@author: luis
"""

## A script to plot 2 2D gaussians with different 
## parameters and plotting them in 3D as well as 
## the contour plot.

import gauss
import numpy as np
import format
import matplotlib.pyplot as plt

# This portion creates a gaussian at an angle -pi/4 relative to x-y space
x = y = np.linspace(-10,10,100)
A = 1
var_p1 = 4
var_p2 = 1
x0 = 0; y0 = 0
alpha = -np.pi/4
X,Y,Z,varx,vary,cov,rho,P1,P2,ZP = gauss.mult_gaussPrincipal(A,x,y,x0,y0,var_p1,var_p2,alpha)

# This portion creates a gaussian at an angle pi/4 relative to x-y space
Ap = 0.3
var_p1p = 5
var_p2p = 3
x0p = 4
y0p = 2
alphap = np.pi/4
Xp,Yp,Zp,varxp,varyp,covp,rhop,P1p,P2p,Zpp = gauss.mult_gaussPrincipal(Ap,x,y,x0p,y0p,var_p1p,var_p2p,alphap)

# Since the domains of the individual gaussians were the same,
# the surfaces interact in real space appropriately. The matrices
# Z and Zp are mapped onto the x-y domain.
fig = format.plot_contour(Zp+Z)
con = format.plot_3d(X,Y,Zp+Z)
plt.show()