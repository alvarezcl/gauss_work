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
y0 = 0
varx = 1
vary = 3
alpha = np.pi/2
X,Y,Z = gauss.mult_gaussFun(x,y,x0,y0,varx,vary,alpha)

xp0 = 5
yp0 = 5
varxp = 8
varyp = 2
alphap = 0
X,Y,Zp = gauss.mult_gaussFun(x,y,xp0,yp0,varxp,varyp,alphap)

fig1 = gauss.plot_3d(X,Y,Z+Zp)
fig2 = gauss.plot_contour(Z+Zp)
