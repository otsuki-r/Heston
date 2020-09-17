# Heston
Short study on the effect of the various parameters in the Heston model (stochastic volatility) in option pricing.

The bulk of the code is held in `Models/Heston.py` and `Models/BlackScholes.py`. A demonstration of the code and some short studies on the effect of the various parameters on option pricing is described in `Heston.ipyb`. Additionally, we study how the introduction of stochastic volatility can introduce the volatitility smile in the terminal stock price distribution and the impact this has in modifying the pricing of European options over the Black Scholes model.

We cover much of the theory in the two accompanying PDFs `Heston.pdf` and `RNPricing.pdf` which covers the derivation of the Heston pricing formula and the notion of risk-neutral pricing respectively. In particular, we have opted to compute the probabilities of landing ITM under the various measures by computing the CDF from the characteristic function (which can be obtained analytically) via the Gil-Pelaez theorem rather than Monte-Carlo pricing, say.
