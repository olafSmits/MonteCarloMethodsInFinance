
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
from scipy.stats import norm

def graphicalComparisonPdf(X, modelPdf, scale = True, xMin = None, xMax = None, axes_object=None, nBins=None):
    _X = X[np.logical_not(np.isnan(X))]
    if xMax is None:
        xMax = np.max(_X) # default parameter of xMax
    if xMin is None:
        xMin = np.min(_X) # default parameter of xMin
    nPlot = 1000
    xPlot = np.linspace(xMin, xMax, nPlot)
    yPlot = modelPdf(xPlot)
    if nBins is None:
		nBins = np.min([np.sqrt(X.size), 40])  
    widthHistogram          = np.max(_X)- np.min(_X)
    averageHeightHistogram  = _X.size/nBins
    areaHistogram           = widthHistogram*averageHeightHistogram
    
    pdfScaleFactor = areaHistogram if not scale else 1 
    # if scale = False we rescale modelPDF(x) by the area of the histogram
    # if scale = True the histogram is scaled, such that its area is 1 (as is the case for modelPDF(x))
	
    if axes_object is None:
	    fig = plt.figure()
	    ax = fig.add_subplot(111)
    else:
	    ax = axes_object
	
    _, _, p = ax.hist(_X, bins=nBins, normed = scale)
    l, = ax.plot(xPlot, yPlot * pdfScaleFactor, 'r', linewidth=3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('pdf(x)')
    ax.set_xlim(xMin, xMax)
    if scale:
        plt.legend([l, p[0]], ['pdf(x)', 'scaled histogram'], loc='best')
    else:
        plt.legend([l, p[0]], ['scaled pdf(x)', 'histogram'], loc='best')
    return ax

def multivariateGaussianRand(M, mu, Sigma):
    """
    multivariateGaussianRand: Generate random numbers from a D-dimensional Gaussian

    INPUT:
         M : size of the sample
        mu : vector of means   [D,1]
     Sigma : covariance matrix [D,D]

    OUTPUT:
         Z : Sample from N(mu,Sigma) Gaussian  [M,D]  
    """
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    
    D = mu.size
    
    L = cholesky(Sigma).T
    
    ## Generate M samples of D-dimensional vectors of independent Gaussians 
    X = np.random.randn(D, M)
    
    
    ## Transform vectors into vectors with proper mean and correlations
    Z = mu[:, np.newaxis] + np.dot(L, X)
    return Z.T
	
	
def  priceAsianArithmeticMeanCallMC_withControlVariate(S0,K,r,T,sigma,M,N):
    """
    priceOptionMC: Black-Scholes price of a generic option providing a payoff.
    
    INPUT:
       S0 : Initial value of the underlying asset
        r : Risk-free interest rate 
        T : Time to expiry 
    sigma : Volatility 
        M : Number of simulations
        N : Number of observations
    payoff_function : payoff function of the option
    
    OUTPUT:
     price_MC : MC estimate of the price of the option in the Black-Scholes model  
     stdev_MC : MC estimate of the standard deviation  

    """
    
    ## Generate M x N samples from N(0,1)
    X = np.random.randn(M, N)

    ## Simulate M trajectories in N steps
    deltaT = T / N
    e = np.exp((r-0.5*sigma**2) * deltaT  + sigma * np.sqrt(deltaT) * X)
    S = np.cumprod(np.c_[S0 * np.ones((M,1)), e], axis=1)        
        
    def arithmeticMean(S):
        S_mean = np.mean(S[:, 1:], 1)
        return np.where(S_mean < K, 0, S_mean - K)
        
    def geometricMean(S):
        S_mean = np.exp(np.mean(np.log(S[:, 1:]), 1))
        return np.where(S_mean < K, 0, S_mean - K)    
        
        
    payoff_ar, price_ar, std_ar = priceOptionMCWithSAsInput(S, r, T, M, arithmeticMean)
    payoff_geom, price_geom, std_geom = priceOptionMCWithSAsInput(S, r, T, M, geometricMean)
    price_geom_exact = priceAsianGeometricMeanCall(S0,K,r,T,sigma,N)
    
    covarianceMatrix = np.cov(np.c_[payoff_ar, payoff_geom].T)
    
    var_ar = covarianceMatrix[0,0]
    var_geom = covarianceMatrix[1,1]
    cov_ar_geom = covarianceMatrix[0,1]
    
    corr_ar_geom = cov_ar_geom / np.sqrt(var_ar * var_geom)
    
    
    price_MC = price_ar - cov_ar_geom / var_geom * (price_geom - price_geom_exact)
    std_MC = std_ar * np.sqrt(1 - corr_ar_geom**2)
    return price_MC, std_MC
	
	
def priceOptionMCWithSAsInput(S, r, T, M, payoff_function):
    """
    priceOptionMC: Black-Scholes price of a generic option providing a payoff.
    INPUT:
        S : A set of pre-computed simulations of Brownian motion
        r : Risk-free interest rate
        M : Number of simulations
        T : Time to maturity
    payoff_function : payoff function of the option
    
    OUTPUT:
     price_MC : MC estimate of the price of the option in the Black-Scholes model  
     stdev_MC : MC estimate of the standard deviation  

    """
   
    ## Compute the payoff for each trajectory
    payoff = payoff_function(S)

    
    ## MC estimate of the price and the error of the option
    discountFactor = np.exp(-r*T);
        
    price_MC = discountFactor * np.mean(payoff)
    stdev_MC = discountFactor * np.std(payoff)/np.sqrt(M)
    return payoff, price_MC, stdev_MC

def priceAsianGeometricMeanCall(S0,K,r,T,sigma,N):
    """
    priceAsianGeometricMeanCall: Price of a Asian call option on the geometric mean in the Black-Scholes model
    INPUT:
       S0 : Initial value of the underlying asset
        K : Strike 
        r : Risk-free interest rate 
        T : Time to expiry 
    sigma : Volatility 
        N : Number of monitoring times

    OUTPUT:
    price : Price of the option in the Black-Scholes model  
    """

    ## Auxiliary parameters
    r_GM     = 0.5 * (r * (N+1) / N - sigma**2 * (1.0 - 1.0/N**2) / 6.0)
    sigma_GM = sigma * np.sqrt((2.0 * N**2 + 3.0 * N + 1.0) / (6.0 * N**2))
    
    d_plus  = np.log(S0/(K*np.exp(-r_GM*T)))/(sigma_GM*np.sqrt(T)) + sigma_GM*np.sqrt(T)/2.0
    d_minus = d_plus - sigma_GM*np.sqrt(T)
    
    
    ## Pricing formula
    price = np.exp(-r*T)*(S0*np.exp(r_GM*T)*norm.cdf(d_plus)-K*norm.cdf(d_minus))
    
    return price
	
def priceEuropeanCall(S0, K, r, T, sigma):
    """
    Price of a European call option in the Black-Scholes model
    
    INPUT:
       S0 : Initial value of the underlying asset
        K : Strike 
        r : Risk-free interest rate 
        T : Time to expiry 
    sigma : Volatility 
    *args : extra arguments can be passed, but are not used

    OUTPUT:
    price : Price of the option in the Black-Scholes model  
    """
    
    discountedStrike = np.exp(-r * T) * K
    totalVolatility = sigma * np.sqrt(T)
    
    d_minus = np.log(S0 / discountedStrike) / totalVolatility - .5 * totalVolatility
    d_plus = d_minus + totalVolatility
    
    # The extra zero in the return is the variance in price, which is of course zero. This way the
    # function behaves similar to the Monte Carlo methods defined below which have non-zero variance
    return S0 * norm.cdf(d_plus) - discountedStrike * norm.cdf(d_minus)