# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 09:36:08 2014

@author: luis
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gauss
import pylab as pl

x = y = np.linspace(-10,10,100)
mean = [0,0]
cov = [[1.58**2,1.5],[1.5,1.58**2]]
Z,X,Y = gauss.mult_gauss(x,y,mean,cov)
