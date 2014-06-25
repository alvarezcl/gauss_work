# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 21:20:27 2014

@author: luis
"""

# Code to test different gaussian distributions

import gauss
import numpy as np

x = y = np.linspace(-20,20,200)
x0 = 0
y0 = -5
varx = 1
vary = 5
A = 1 # Amplitude
alpha = 0
rho1,X,Y,Z = gauss.mult_gaussFunAlt(A,x,y,x0,y0,varx,vary,alpha)

xp0 = 10
yp0 = 0
varxp = 8
varyp = 2
alphap = 0
B = 0.5 # Amplitude
rho2,X,Y,Zp = gauss.mult_gaussFunAlt(B,x,y,xp0,yp0,varxp,varyp,alphap)

fig3 = gauss.plot_3d(X,Y,Z+Zp)
fig4 = gauss.plot_contour(Z+Zp)

# Plotting rho vs alpha (Note the divergence of rho)
a = np.linspace(-np.pi/2,np.pi/2,100)
rho = (varx-vary)*np.tan(2*a)/(2*np.sqrt(varx*vary))