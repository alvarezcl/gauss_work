# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 07:00:08 2014

@author: luis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gauss
import pylab as pl
import sys
from scipy.stats import multivariate_normal

fig1 = plt.figure(1)
ax1 = Axes3D(fig1)

x = y = np.linspace(-20,20,100)

X,Y = np.meshgrid(x,y)

gauss.plot_3d(X,Y,Z)

sys.exit()




Z = gauss.gauss_1d(Y,0,.1)

ax1.plot_surface(X,Y,Z)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
plt.show()


con1 = plt.figure(3)
im = pl.imshow(Z)
cset = pl.contour(Z,[0.1,0.2,0.5,0.6,0.7])

#cset = pl.contour(Z,[0.1,0.2])

pl.clabel(cset,inline=True)
pl.colorbar(im)
plt.title('Cont')
plt.show()

Zp = gauss.sum_gauss(X,0,1,Y,0,1)
fig2 = plt.figure(2)
ax2 = Axes3D(fig2)
ax2.plot_surface(X,Y,Zp)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
plt.show()

con2 = plt.figure(4)
im = pl.imshow(Zp)
cset = pl.contour(Zp,[0.1,0.2,0.5,0.6,0.7])
pl.clabel(cset,inline=True)
pl.colorbar(im)
plt.title('Cont-Sum')
plt.show()
