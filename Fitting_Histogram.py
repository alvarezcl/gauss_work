# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 18:30:16 2014

@author: luis
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss

norm = False
N_tot = 10000
drp = 0.6; str_dr = '$/D_{rp}=%.1f$' % drp; str_ds = '$/D_{rs}=%.1f$' % (1-drp)

# Draw from the major gaussian. Note the number N. It is
# the main parameter in obtaining your estimators.
mean = 0; sigma = 1; var = sigma**2; N = N_tot*drp
points = gauss.draw_1dGauss(mean,var,N)
bins = N/50
hist,bin_edges = np.histogram(points,bins,density=norm)
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.0
width = bin_edges[1]-bin_edges[0]

# Now draw from a minor gaussian. Note Np
meanp = 3; sigmap = 1; varp = sigmap**2; Np = N_tot-N
pointsp = gauss.draw_1dGauss(meanp,varp,Np)
binsp = bins    
histp,bin_edgesp = np.histogram(pointsp,binsp,density=norm)
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
x = np.linspace(mean-8*sigma,meanp+8*sigmap,1000)

# Obtain Gaussian Curves for estimated parameters
PrimGauss = gauss.gaussFun(x,*p); SecGauss = gauss.gaussFun(x,*pp)

# Check Amplitude Ratio and make sure it is around 1-dr
Ar_hist_sec = max(histp)/float(max(hist)); Ar_param_sec = pp[0]/float(p[0])

# Now try the parameters of the concatenated sum
points_sum = np.array(points.tolist()+pointsp.tolist())
bins_sum = bins
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
plt.figure(1)
plt.hist(points,bins,normed=norm); plt.hist(pointsp,binsp,normed=norm)
plt.title('Gaussians')
plt.figure(2)
p1, = plt.plot(x,PrimGauss,lw=2); p2, = plt.plot(x,SecGauss); p3, = plt.plot(x,sumPSGauss)
p4, = plt.plot(x,catGauss); p5, = plt.plot(x,sum_param_Gauss)
plt.legend([p1,p2,p3,p4,p5],[prim_Gauss_info,sec_Gauss_info,'Sum of PS',cat_Gauss_info,sum_Gauss_info],loc=7,prop={'size':10})
plt.text(mean,max(PrimGauss),str_dr)
plt.text(meanp,max(SecGauss),str_ds)
plt.ylim(0,max(sumPSGauss))
plt.show()