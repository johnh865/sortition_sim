# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import default_rng, RandomState
from scipy.stats import norm, rayleigh, lognorm
import matplotlib.pyplot as plt

from model_v2.model import Citizens


class CrimeProbOccur:
    """Model of likelihood of crime occurrence.
    
    Attributes
    ----------
    base_probability : float
        Base probability of occurrence of crime [0 to 1].
    rayleigh_loc : float
        Rayleigh distribution parameter
    rayleigh_scale : float
        Rayleight distribution parameter
    pdf_factor : float
        Maximum probability of crime prevention [0 to 1].
    """
    def __init__(self, rng: RandomState=None):
        if rng is None:
            rng = RandomState()
            
        self.rng = rng
        self.base_probability = 0.2
        self.rayleigh_loc = 0.05
        self.rayleigh_scale = 0.40
        self.pdf_factor = 0.75
        
    
    def spend_prevent_pdf(self, x):
        """input spending per capita, return probability of crime prevention."""
        f = self.pdf_factor
        loc = self.rayleigh_loc
        scale = self.rayleigh_scale
        
        return rayleigh.cdf(x, loc=loc, scale=scale) * f
    
    
    
    def crime_locs(self, citizens: Citizens, spend: float) -> np.ndarray:
        """Get citizen index locations where crimes will be committed. 
        
        Parameters
        ----------
        citizens : Citizens
            Citizens. 
        spend : float
            Government spending to prevent crime.

        Returns
        -------
        locs : ndarray[bool]
            Index mask locations of citizens where crime is committed.

        """
        
        prob_base = self.base_probability
        num = len(citizens)
        
        samples = self.rng.uniform(low=0, high=1, size=num)
        prob_prevent = self.spend_prevent_pdf(spend)
        prob_crime = prob_base * (1 - prob_prevent)
        
        locs = samples < prob_crime
        return locs

        
        
class LossModel:
    def __init__(self, mean=86220/61713, median=1, rng: RandomState=None):
        
        self.mean = mean
        self.median = median
        
        mu = np.log(median)
        sigma = np.sqrt(2 * (np.log(mean) - mu))
        self.sigma = sigma
        self.scale = np.exp(mu)
        
        
        if rng is None:
            rng = default_rng()
        self.rng = rng
        self.loc = 0
        self.scale = 1
    
    
    def rvs(self, n: int):
        return lognorm.rvs(
            scale=self.scale, s=self.sigma, size=n, random_state=self.rng)
    
    
    def pdf(self, x): 
        return lognorm.pdf(x, scale=self.scale, s=self.sigma,)
        
    

def test_spend_model():    
    x = np.linspace(0, 2, 1001)
    cpo = CrimeProbOccur()
    y = cpo.spend_prevent_pdf(x)
    
    plt.figure()
    plt.plot(x*100, y)
    plt.xlabel('Gov Protection Spend per capita [% mean income]')
    plt.ylabel('Probabability of preventing act of crime.')
    plt.grid()
    
    
    
def test_crime_occur():
    
    pass

def test_loss_model():
    x = np.linspace(0, 5, 1001)
    lmodel = LossModel()
    y2 = lmodel.pdf(x)

    plt.figure()
    plt.plot(x, y2)
    plt.xlabel('Income lost for unprevented crime event')
    plt.ylabel('PDF')
    
if __name__ == '__main__':
    test_spend_model()
    