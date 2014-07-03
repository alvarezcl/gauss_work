# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:45:42 2014

@author: luis
"""

## This file draws from two gaussian distributions and attempts
## to find the best fit parameters. Useful for deblending.

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gauss

loop = 11
info = []; num_Pts = []
N_tot = 1000

# Loop over number of points drawn from each distribution
for i in xrange(0,loop):
    
    # Draw from the major gaussian. Note the number N. It is
    # the main parameter in obtaining your estimators.
    mean = 0; sigma = 1; var = sigma**2; N = N_tot - 100*i
    A = 1/np.sqrt((2*np.pi*var))
    points = gauss.draw_1dGauss(mean,var,N)
        
    # Now draw from a minor gaussian. Note Np
    meanp = 2; sigmap = 2; varp = sigmap**2; Np = N_tot-N
    Ap = 1/np.sqrt((2*np.pi*varp))
    pointsp = gauss.draw_1dGauss(meanp,varp,Np)
        
    # Now implement the sum of the draws by concatenating the two arrays.
    points_tot = np.array(points.tolist()+pointsp.tolist())
    bins_tot = len(points_tot)/5
    hist_tot, bin_edges_tot = np.histogram(points_tot,bins_tot,density=True)
    bin_centres_tot = (bin_edges_tot[:-1] + bin_edges_tot[1:])/2.0
    # Plot the histogram of the sum
    #plt.figure(2); plt.title('Combined Histogram')
    #plt.hist(points_tot,bins_tot,normed=True)    
        
    # Initial guess
    p0 = [A, mean, sigma]
    
    # Result of the fit
    coeff, var_matrix = curve_fit(gauss.gaussFun, bin_centres_tot, hist_tot, p0=p0)
        
    # Get the fitted curve
    if np.mod(i,2) == 0:
        x = np.linspace(mean-15,meanp+15,200)
        hist_fit = gauss.gaussFun(x, *coeff)
        fig = plt.figure(5); plt.title('Gaussian Estimate')
        plt.plot(x,hist_fit)
        fig.canvas.draw()        
        
    # Error on the estimates
    error_parameters = np.sqrt(np.array([var_matrix[0][0],var_matrix[1][1],var_matrix[2][2]]))
    info.append(coeff)
    num_Pts.append([N,Np])


plt.figure(3) 
info = np.array(info); num_Pts = np.array(num_Pts)
p1 = plt.scatter(np.sort(num_Pts[:,0])/N_tot,info[:,0][::-1],c='b',marker='o')
plt.hold()
p2 = plt.scatter(np.sort(num_Pts[:,0])/N_tot,info[:,1][::-1],c='r',marker='x')
p3 = plt.scatter(np.sort(num_Pts[:,0])/N_tot,info[:,2][::-1],c='k',marker='v')
plt.legend([p3,p2,p1],["Sigma","Mu","Amplitude"],loc=1,borderaxespad=0)
plt.xlabel('Draw Ratio for Primary Gaussian'); plt.ylabel('Parameter Value')
plt.suptitle('Primary Gaussian Parameters: Mu = '+ str(mean) +' , Sigma = ' + str(sigma) + ', Amplitude = ' + str(A) +'\n Secondary Gaussian Parameters: Mu = '+ str(meanp) +' , Sigma = ' + str(sigmap) + ', Amplitude = ' + str(Ap))
plt.xlim((0,1.0))
plt.show()

plt.figure(5)
p1, = plt.plot(x,gauss.gaussFun(x,*(A,mean,sigma)),'o',lw=0.1)
p2, = plt.plot(x,gauss.gaussFun(x,*(Ap,meanp,sigmap)),'x',lw=0.1)
plt.legend([p1,p2],["Primary Gauss","Secondary Gauss"])