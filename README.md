gauss_work
==========

Work centered around Gaussian Distributions

2_1D_Gaussian_Fit:
Produce histogram of samples drawn from two gaussians and attempt to find fit of sum of gaussians.

1D_Gaussian_Fit: 
Produce histogram of samples drawn from gaussian and plot gaussian corresponding to the samples.

astroMLtest: 
Draw from 2D Gaussian. Obtain parameters from the scatter and convert into variances in x and y (including cov)

curveFittingRobust: 
Draw from 2 1D Gaussians and attempt to find the parameters of the
sum of their histogram. Result is plotted with a curve, primary, secondary, and sum.

Fitting_Histogram:
Draw from 2 1D Gaussians and attempt to find the parameters of the sum of their histogram with plotting of the histogram distributions as well as the gaussians. Streamlined for amplitude ratios and figures are much more presentable.

Fitting_Summing_Gaussians:
Draws from 2 1D Gaussians and attempts to find the parameters of the sum of the distributions. Key difference is the loop through the different amplitude ratios. 

plotutils:
Formatting library for strings and information in plots as well as producing plots given inputs.

GalSimInstallation:
Instructions for installing GalSim

gauss:
Function library for various uses ranging from drawing from 1D and 2D gaussians and for their analytic definitions.

Moving_Thru_Amp:
Script to change the amplitude coefficient on a secondary gaussian. Produces an estimate of the best fit parameters over
the change in amplitude.

Moving_Thru_Mean:
Script to change the mean on a secondary gaussian. Produces an estimate of the best fit parameters over
the change in mean.

Moving_Thru_Sigma:
Script to change sigma on a secondary gaussian. Produces an estimate of the best fit parameters over
the change in sigma.

multigauss:
A script to plot 2 2D gaussians with different parameters and plotting them in 3D as well as the contour plot.

plotting_bar_2dhist:
A script to draw from 2 2D distributions and attempts to plot the points using 2d histograms and 3D bar graphs, from binned data.

pullDistributions:
Script runs through a number of trials to produce parameters for samples drawn from known gaussians. Pull distributions for the amplitude, mean, and sigma are produced to determine if they converge to a standard normal random variable. 

parameterEstimation_1D (Deprecated):
Incorrect version of Fitting_Summing_Gaussians


