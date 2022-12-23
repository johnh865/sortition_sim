# -*- coding: utf-8 -*-
"""
n1 = Corrupt Punisher
n2 = Corrupt Forgiver
n3 = Innocent Punisher
n4 = Innocent Forgiver
"""
import pdb

from src.models import Model1, Model1Param
from numpy.random import default_rng
import numpy as np

def test1():
    """test simple model of population of corrupt punishers with 2 term limit."""
    params = Model1Param(bias=0.0,
                         m_detect=0.0,
                         l_detect=0.0,
                         n1=10, 
                         num_leaders=3,
                         term_limit=2,
                         num_sessions=100)
    
    rstate = default_rng(0)
    model = Model1(params, rstate=rstate)
    model.sessions
    actions = model.sessions.history.actions
    
    assert actions.will_steal.all()
    
    # Cannot punish at 0th session because nobody to punish. 
    will_punish = actions.will_punish.to_numpy()
    assert will_punish[0] == False
    
    # For other iterations, cycle through punish/don't punish after each 2 
    # terms end. 
    for ii in range(1, len(actions) // 2):
        assert will_punish[2*ii - 1] == False
        assert will_punish[2*ii] == True
        
        
    
    leaders = model.sessions.history.leader
    
    # Test to make sure 2 term limit is respected for leaders 
    for ii in range(len(leaders) // 2):
        leaders1 = leaders[2*ii]
        leaders2 = leaders[2*ii + 1]
        
        assert np.all(leaders1 == leaders2)
        
        
        
        
    # Test to make sure 
if __name__ == '__main__':
    
    import logging
    logging.basicConfig(level = logging.DEBUG)    
    test1()
    