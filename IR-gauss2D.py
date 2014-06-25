import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as pl

#Irving Rodriguez
#06/25/2014
#Version 3
#This function plots a 2-Dimensional Gaussian and its projection based on 5 parameters:
#Centroid coordiantes (2 parameters)
#Angle between semi-major axis of ellipse and x-axis (1 parameter)
#Semi-major and semi-minor axes of 1-Sigma contour (2 parameters)


def main():
    
    xMean = 3
    yMean = 4 #Centorid coordinates
    alpha = np.pi/3 #Angle between semi-major axis of ellipse and x-axis
    major = 4
    minor = 3 #Semi-major and semi-minor axes of 1-Sigma contour
    
    u = [xMean, yMean]
    o = comVarMat(major, minor, alpha)
    x, y = np.mgrid[-10:10:.05, -10:10:.05]
    
    gauss = make2DGaussian(x, y, u, o)
    plotGaussian(x, y, gauss)

#Computes the matrix of second moments using the geometry of the elliptical contour at an angle alpha to the x-axis in real space.
#Input: None
#Returns: 2x2 Symmetric Matrix of Second Moments
def comVarMat(major, minor, alpha):
    oXX = (major**2 * (np.cos(alpha))**2 + minor**2 * (np.sin(alpha))**2)**(.5)
    oYY = (major**2 * (np.sin(alpha))**2 + minor**2 * (np.cos(alpha))**2)**(.5)
    oXY = ((major**2 - minor**2)*np.sin(alpha)*np.cos(alpha))**.5
    o = [[oXX, oXY], [oXY, oYY]]
    return o
    
#Constructs the values of a 2-Dimensional Gaussian distribution for a given coordinate system.
#Input: x and y axes values, 2x1 matrix of centroid coordinates, 2x2 matrix of second moments
#Output: Bivariate Gaussian over the given coordinate system.
def make2DGaussian(x, y, u, o):
    pos = np.empty(x.shape + (2,))
    pos[:,:,0] = x
    pos[:,:,1] = y
    
    gauss = multivariate_normal.pdf(pos, u, o)
    
    return gauss
    
#Plots a given 2D Gaussian and its contours for a given coordinate system.
#Input: x and y axes values, 2D Gaussian distribution values
#Output: 3D Surface Plot and 2D Contour plot
def plotGaussian(x, y, gauss):
    fig, ax = pl.subplots(1, 2, figsize=(12,6))
    
    ax[0] = fig.add_subplot(121, projection='3d')
    ax[0].plot_surface(x, y, gauss)
    ax[0].axis('tight')
    
    ax[1].contourf(x, y, gauss)
    ax[1].set_xlim([-10,10])
    ax[1].set_ylim([-10,10])    

    pl.show()
    
if __name__ == "__main__":
    main()