# -*- coding: utf-8 -*-

from src import base
import numpy as np
from numpy.random import Generator
from dataclasses import dataclass
from functools import cached_property

@dataclass
class Model1Param:
    bias : float
    m_detect : float
    l_detect : float
    
    
    n1 : int = 0
    n2 : int = 0
    n3 : int = 0
    n4 : int = 0
    
    num_leaders : int = 11
    num_sessions : int = 10
    
    start_money : float = 0.0
    member_reward : float = 1
    leader_reward : float = 2
    punishment : float = 30
    term_limit : int = 25
    max_steal : float = 8
        
    
class Model1:
    def __init__(self, parameters: Model1Param, rstate: Generator):
        self.parameters = parameters
        self._build_member_types()
        self.rstate = rstate
        
        
    def _build_member_types(self):
        parameters = self.parameters
        
        bias = parameters.bias
        m_detect = parameters.m_detect
        l_detect = parameters.l_detect
        
        # Corrupt Punisher
        self._member1 = base.Member(
            is_corrupt = True, 
            will_punish = True, 
            corrupt_detect_bias = bias,
            corrupt_detect_as_member = m_detect,
            corrupt_detect_as_leader = l_detect, 
            )
        
        # Corrupt Forgiver
        self._member2 = base.Member(
            is_corrupt = True, 
            will_punish = False, 
            corrupt_detect_bias = bias,
            corrupt_detect_as_member = m_detect,
            corrupt_detect_as_leader = l_detect, 
            )
        
        # Innocent Punisher
        self._member3 = base.Member(
            is_corrupt = False, 
            will_punish = True, 
            corrupt_detect_bias = bias,
            corrupt_detect_as_member = m_detect,
            corrupt_detect_as_leader = l_detect, 
            )
        
        # Innocent Forgiver
        self._member4 = base.Member(
            is_corrupt = False, 
            will_punish = False, 
            corrupt_detect_bias = bias,
            corrupt_detect_as_member = m_detect,
            corrupt_detect_as_leader = l_detect, 
            )
        
        
    @cached_property
    def sessions(self):
        params = self.parameters
        election = base.Election(
            
                self.members, 
                num_win=params.num_leaders,
                rstate=self.rstate
            )
        settings = base.SessionSettings(
            
                start_money = params.start_money,
                member_reward = params.member_reward,
                leader_reward = params.leader_reward,
                punishment = params.punishment,
                term_limit = params.term_limit,
                max_steal = params.max_steal,
            )
        sessions1 = base.Sessions(
            
                members = self.members,
                election = election,
                settings = settings,
                num = params.num_sessions
            )
        return sessions1.run()
    

    @cached_property
    def members(self):
        n1 = self.parameters.n1
        n2 = self.parameters.n2
        n3 = self.parameters.n3
        n4 = self.parameters.n4
        
        m1 = self._member1.uniform(n1)
        m2 = self._member2.uniform(n2)
        m3 = self._member3.uniform(n3)
        m4 = self._member4.uniform(n4)
        return base.Members.concat([m1, m2, m3, m4])
    
    
    

