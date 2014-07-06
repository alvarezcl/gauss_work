# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 18:30:16 2014

@author: luis
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss

## Draw from 2 1D Gaussians and attempt to find 
## the parameters of the sum of their histograms 
## with plotting of the histogram distributions 
## as well as the gaussians. Streamlined for 
## amplitude ratios and figures are much more presentable.

norm = False
N_tot = 10000
drp = 0.8; str_dr = '$/A_{rp}=%.1f$' % drp; str_ds = '$/A_{rs}=%.1f$' % (1-drp)

# Draw from the major gaussian. Note the number N. It is
# the main parameter in obtaining your estimators.
mean = 0; sigma = 1; var = sigma**2; N = N_tot*drp
points = gauss.draw_1dGauss(mean,var,N)

# Histogram parameters
bin_size = 0.1; min_edge = mean-6*sigma; max_edge = mean+9*sigma
Nn = (max_edge-min_edge)/bin_size; Nplus1 = Nn + 1
bin_list = np.linspace(min_edge, max_edge, Nplus1)

hist,bin_edges = np.histogram(points,bin_list,density=norm)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.0
width = bin_edges[1]-bin_edges[0]

# Now draw from a minor gaussian. Note Np
meanp = 3; sigmap = 1; varp = sigmap**2; Np = N_tot-N
pointsp = gauss.draw_1dGauss(meanp,varp,Np)
binsp = bin_list
histp,bin_edgesp = np.histogram(pointsp,bin_list,density=norm)
bin_centersp = (bin_edgesp[:-1] + bin_edgesp[1:])/2.0
widthp = bin_edgesp[1]-bin_edgesp[0]

# Initial guess
g = np.random.uniform(-5,5,1)
p0 = [g,g,g]

# Result of the fits for each gaussian
parametersPrim, var_matrixPrim = curve_fit(gauss.gaussFun, bin_centers, hist, p0=p0)
parametersSec, var_matrixSec = curve_fit(gauss.gaussFun, bin_centersp, histp, p0=p0)

# Obtain the parameters for each single gaussian
p = (parametersPrim[0],parametersPrim[1],parametersPrim[2])
pp = (parametersSec[0],parametersSec[1],parametersSec[2])

# Domain
x = np.linspace(min_edge,max_edge,1000)

# Obtain Gaussian Curves for estimated parameters from P and S
PrimGauss = gauss.gaussFun(x,*p); SecGauss = gauss.gaussFun(x,*pp)

# Check Amplitude Ratio for Secondary and make sure it is around 1-dr
Ar_hist_sec = max(histp)/float(max(hist)); Ar_param_sec = pp[0]/float(p[0])

# Now try the parameters of the concatenated sum
points_sum = np.array(points.tolist()+pointsp.tolist())
bins_sum = bin_list
hist_sum,bin_edges_sum = np.histogram(points_sum,bins_sum,density=norm)
bin_centers_sum = (bin_edges_sum[:-1] + bin_edges_sum[1:])/2.0
parametersSum, var_matrixSum = curve_fit(gauss.gaussFun, bin_centers_sum, hist_sum, p0=p0)
p_cat = (parametersSum[0],parametersSum[1],parametersSum[2])
# Obtain Gaussian Curve for estimated parameters of cat-ed data
catGauss = gauss.gaussFun(x,*p_cat)

# Sum of Primary and Secondary Gaussians
sumPSGauss = PrimGauss+SecGauss

# Now try to extract parameters from the data representing the sum of the primary
# and secondary gaussians 
param_Sum,err = curve_fit(gauss.gaussFun,x,sumPSGauss,p0)
sum_param_Gauss = gauss.gaussFun(x,*param_Sum)

# String formatting for the legend and the display figure
prim_Gauss_info = 'Primary: $\mu=%.1f$, $\sigma=%.1f$' % (mean,sigma)
sec_Gauss_info = 'Secondary: $\mu=%.1f$, $\sigma=%.1f$' % (meanp,sigmap)
cat_Gauss_info = 'Cat: $\mu=%.2f$, $\sigma=%.2f$' % (np.abs(p_cat[1]),np.abs(p_cat[2]))
sum_Gauss_info = 'Sum: $\mu=%.2f$, $\sigma=%.2f$' % (np.abs(param_Sum[1]),np.abs(param_Sum[2]))

# Plot each Gaussian:
# Primary
# Secondary
# Sum of Primary and Secondary
# Gaussian representing concatenated data
# Gaussian representing element-wise summed data.
plt.figure(1); plt.title('Histograms')
plt.hist(points_sum,bin_list,zorder=1,facecolor='cyan'); plt.hist(points,bin_list,normed=norm,facecolor='blue'); plt.hist(pointsp,bin_list,normed=norm,facecolor='green')
p4, = plt.plot(x,catGauss,'r',lw=2); plt.legend([p4],['Cat Fit'])
plt.title('Histogram with Curve Fit')
plt.figure(2)
p1, = plt.plot(x,PrimGauss,lw=2); p2, = plt.plot(x,SecGauss); p3, = plt.plot(x,sumPSGauss)
p4, = plt.plot(x,catGauss,lw=2); p5, = plt.plot(x,sum_param_Gauss)
plt.legend([p1,p2,p3,p4,p5],[prim_Gauss_info,sec_Gauss_info,'Sum of PS',cat_Gauss_info,sum_Gauss_info],loc=7,prop={'size':10})
plt.text(mean,max(PrimGauss),str_dr)
plt.text(meanp,max(SecGauss),str_ds)
plt.ylim(0,max(sumPSGauss))
plt.show()