import numpy as np
from scipy.stats import norm

class BlackScholes():
    def __init__(self, r, sigma):
        """
        Parameters
        ----------
        r : float
            Risk-free rate
        mu : float
            Drift
        sigma : float
            Volatility
        """
        self.r = r
        self.sigma = sigma
    
    def getCallPrice(self, K, S, tau, q = 0.):
        """
        Computes the price of a European call that, optionally, pays a dividend `q`.
        
        Parameters
        ----------
        K : float
            Strike
        S : float
            Stock spot price
        tau : float
            Time to maturity (years)
        q : float
            Continuous dividend rate
            
        Returns
        -------
        float
            Price of the European call.
        """
        q1 = 1. / (self.sigma * np.sqrt(tau))
        q2 = np.log(S / K)
        q3 = (self.r - q + 0.5 * self.sigma ** 2) * tau
        q4 = (self.r - q - 0.5 * self.sigma ** 2) * tau
        d1 = q1 * (q2 + q3)
        d2 = q1 * (q2 + q4)
        return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)
    
    def getPutPrice(self, K, S, tau, q = 0.):
        """
        Computes the price of a European put that, optionally, pays a dividend `q`.
        We use put-call parity to price the put.
        
        Parameters
        ----------
        K : float
            Strike
        S : float
            Stock spot price
        tau : float
            Time to maturity (years)
        q : float
            Continuous dividend rate
            
        Returns
        -------
        float
            Price of the European put.
        """
        return self.getCallPrice(K, S, tau, q) + K * np.exp(-self.r * tau) - S * np.exp(-q * tau)
