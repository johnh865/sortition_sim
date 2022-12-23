# -*- coding: utf-8 -*-

import scipy
import numpy as np
from scipy.stats import norm, lognorm

import pdb


class IncomeDistribution:
    """Create lognorm income distribution.
    
    Statistical literacy and the lognoraml distribution:
    https://www.researchgate.net/publication/327971358
    
    """
    def __init__(self, mean=86220, median=61713, random_state=None):
        mu = np.log(median)
        sigma = np.sqrt(2 * (np.log(mean) - mu))
        self.sigma = sigma
        self.scale = np.exp(mu)
        self.random_state = random_state
        self.mean = mean
        self.median = median
        self.mu = mu
        
    
    def pdf(self, x):
        """Probability density function."""
        return lognorm.pdf(x, s=self.sigma, scale=self.scale)
    
    def ppf(self, q):
        """Percent point function (inverse of cdf - percentiles"""
        return lognorm.ppf(q, s=self.sigma, scale=self.scale)
    
    
    def cdf(self, x):
        """Cumulative density function"""
        return lognorm.cdf(x, s=self.sigma, scale=self.scale)    
    
    
    def income_from_parameter(self, x: float) -> float:
        """Given standard normal deviate number `x`, calculate associated income."""
        return np.exp(self.sigma * x + self.mu)
        
        
    def rvs(self, n):
        """Draw `n` samples."""
        s = self.sigma
        rs = self.random_state
        return lognorm.rvs(s=s, scale=self.scale, size=n, random_state=rs)
    
    
    def __call__(self, n):
        return self.rvs(n)
    
    
    
def saving_model_00(income: float) -> float:
    """Initial savings model."""
    base_cost_of_living = 40000
    spend_ratio = 0.20
    
    spending = base_cost_of_living + income * spend_ratio
    savings = income - spending
    return savings
    
    
def test_income():
    
    dist = IncomeDistribution(random_state=0)
    sample = dist(10000)
    mean = dist.mean
    median = dist.median
    ratio1 = np.abs(mean - sample.mean()) / mean
    ratio2 = np.abs(median - np.median(sample)) / median
    print('Accuracy on mean = ', ratio1)
    print('Accuracy on median = ', ratio2)
    
    assert ratio1 < 0.03
    assert ratio2 < 0.03
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test_income()
    
    
    dist = IncomeDistribution(random_state=0)
    x = np.linspace(0, 0.5e6, 1000)
    y = dist.pdf(x)
    y2 = dist.cdf(x)
    plt.subplot(2,1,1)
    plt.plot(x / dist.median * 100, y)
    plt.xlabel('Income (% Median)')
    plt.ylabel('PDF')
    plt.title('PDF of LogNorm Income Distribution')
    plt.tight_layout()
    plt.ylim(0, None)
    plt.xlim(0, None)
    plt.grid()
    
    
    plt.subplot(2,1,2)
    plt.plot(x / dist.median * 100, y2)
    plt.xlabel('Income (% Median)')
    plt.ylabel('CDF')
    plt.title('CDF of LogNorm Income Distribution')
    plt.tight_layout()
    plt.ylim(0, None)
    plt.xlim(0, None)
    plt.grid()
    