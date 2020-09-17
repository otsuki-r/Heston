import cmath
import numpy as np
import scipy.integrate

class Heston():
    def __init__(self, r, v0, kappa, theta, eta, rho, sigma):
        """
        Parameters
        ----------
        r : float
            Risk-free rate
        v0 : float
            Initial volatility level
        kappa : float
            Volatility mean-reversion rate
        theta : float
            Long-term volatility level
        eta : float
            Volatility risk parameter
        rho : float
            Correlation coefficient of Brownian motions
        sigma : float
            Vol-of-vol

        """
        self.r = r
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.sigma = sigma

        self.b = [kappa + eta - rho * sigma, kappa + eta]
        self.u = [0.5, -0.5]
    
    def makeCharacteristicFunctions(self, S, tau, q = 0.):
        """
        Creates the characteristic functions of the logarithmic terminal stock price
        under the two measures appearing in the option prices.
        
        Parameters
        ----------
        tau : float
            Time to maturity
        S : float
            Stock spot price
        q : float, optional
            Continuous stock dividend rate (per year)
            
        Returns
        -------
        list
            Two characteristic functions
        """
        def _makeKthCharacteristicFunction(k):
            """
            Constructs the characteristic function under the kth measure (k = 0, 1).
            
            Parameters
            ----------
            k : int
                Characteristic function index
                
            Returns
            -------
            callable
                Characteristic function of the kth measure.
            """
            def _psi(phi):
                """
                By Albrecher's little trap formulation which produces a slightly
                more stable integrand for numerical integration when compared to
                Heston's original formulation
                """
                q1 = self.b[k] - self.rho * self.sigma * phi * 1j
                q2 = 2 * self.u[k] * phi * 1j - phi**2
                d = cmath.sqrt(q1 * q1 - self.sigma**2 * q2)
                c = (q1 - d) / (q1 + d)
                
                c1 = (self.r - q) * phi * tau * 1j
                c2 = self.kappa * self.theta / (self.sigma ** 2)
                c3 = (q1 - d) * tau
                c4 = cmath.log((1. - c * cmath.exp(-d * tau))/(1. - c))
                C = c1 + c2 * (c3 - 2 * c4)

                d1 = 1./ self.sigma**2
                d2 = q1 - d
                d3 = (1. - cmath.exp(-d * tau)) / (1 - c * cmath.exp(-d * tau))
                D = d1 * d2 * d3
                
                return cmath.exp(C + D * self.v0 + phi * np.log(S) * 1j)
            return _psi
        
        return [_makeKthCharacteristicFunction(k) for k in range(2)]
    
    def getDensity(self, S, tau, xT, lower, upper):
        """
        Computes the value of the pdf of the log terminal stock price equalling xT.
        
        The pdf is obtained by a Fourier transform of the characteristic function.
        Since the imaginary part must vanish, we focus only on the real part.
        
        Note that the characteristic function, and hence pdf, changes depending on
        the spot price S and time to maturity tau and so we need to additionally
        specify the paramters of the pdf that is being used.
        
        Parameters
        ----------
        S : float
            Spot price
        tau : float
            Time to maturity (years)
        xT : float
            Log-terminal stock price
        lower : float, optional
            Lower limit of Fourier integral
        upper : float, optional
            Upper limit of Fourier integral
            
        Returns
        -------
        float
            Value of the pdf of the terminal stock price, given spot price S,
            eavulated at xT
        """
        def _makeFourierIntegrand(xT):
            """
            Helper function to return the integrand of the Fourier transform
            
            Returns
            -------
            callable
                Integrand of the Fourier transform of the characteristic function
            """
            def _fourierIntegrand(phi):
                _, characteristic_function = self.makeCharacteristicFunctions(S, tau)
                return (np.exp( -phi * xT * 1j) * characteristic_function(phi)).real
            return _fourierIntegrand
        
        # f is a callable representing the integrand of the Fourier transform.
        f = _makeFourierIntegrand(xT)
        
        return 1. / np.pi * scipy.integrate.quad(f, lower, upper)[0]
    
    def getItmProbabilities(self, K, S, tau, q, lower, upper):
        """
        Computes the probabilities of the terminal price exceeding the strike
        under the two measures. Note that the characteristic function and CDF
        are related by the Gil-Pelaez theorem.
        
        Parameters
        ----------
        K : float
            Strike price
        S : float
            Stock spot price
        tau : float
            Time to maturity (years)
        q : float
            Continuous stock dividend rate (per year)
        lower : float, optional
            Lower limit of integration
        upper : float, optional
            Upper limit of integration
        
        Returns
        -------
        list
            List of the two probabilties used in pricing European options.
        """
        def makeGilPelaezIntegrands():
            """
            Helper function to construct the integrands appearing in the
            Gil-Pelaez theorem.
            
            Returns
            -------
            list
                List of two functions, as a function of phi, representing
                the two integrands in the Gil-Pelaez theorem.
            """
            psi = self.makeCharacteristicFunctions(S, tau, q)
            i_log_K = np.log(K) * 1j
        
            def _makeKthGilPelaezIntegrand(k):
                """
                Constructs the Gil-Pelaez integrand for the kth characteristic function (k = 0, 1)
                
                Returns
                -------
                callable
                    The Gil-Pelaez integrand as a function of phi.
                """
                def _gilPelaezIntegrand(phi):
                    return ((cmath.exp(- phi * i_log_K) * psi[k](phi))/ (phi * 1j)).real
                return _gilPelaezIntegrand
            return [_makeKthGilPelaezIntegrand(k) for k in range(2)]
        
        # `integrands` is a list of the callables representing the integrands of
        # the Gil-Pelaez theorem.
        integrands = makeGilPelaezIntegrands()
        integrals = [scipy.integrate.quad(integrands[k], lower, upper)[0] for k in range(2)]
        return [0.5 + 1/np.pi * I for I in integrals]
    
    def getCallPrice(self, K, S, tau, q = 0., lower = 1e-8, upper = 1e2):
        """
        Computes the price of a European call option under the Heston model.
        
        Parameters
        ----------
        K : float
            Strike price
        S : float
            Stock spot price
        tau : float
            Time to maturity (years)
        q : float
            Continuous stock dividend rate (per year)
        
        Returns
        -------
        float
            Call price
        """
        P1, P2 = self.getItmProbabilities(K, S, tau, q, lower, upper)
        return S * np.exp(- q * tau) * P1 - K * np.exp(-self.r * tau) * P2
    
    def getPutPrice(self, K, S, tau, q = 0., lower = 1e-8, upper = 1e2):
        """
        Computes the price of a European put option under the Heston model.
        Uses put-call parity.
        
        Parameters
        ----------
        K : float
            Strike
        S : float
            Stock spot price
        tau : float
            Time to maturity (years)
        q : float
            Continuous stock dividend rate (per year)
            
        Returns
        -------
        float
            Put price
        """
        return self.getCallPrice(K, S, tau, q, lower, upper) + K * np.exp(-self.r * tau) - S * np.exp(-q * tau)
