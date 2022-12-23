# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from numpy.random import default_rng, RandomState
from scipy.stats import lognorm, uniform, norm
from model_v2 import income

from typing import Union, Protocol


class LeaderPolicyType(Protocol):
    tax_flat : float
    tax_income_pct : float
    leader_pay : float
    leader_punish_pay_threshold: float
    leader_punish_amount : float    
    

class LeaderPolicySeriesType(LeaderPolicyType, pd.Series):
    pass




    
@dataclass
class LeaderPolicy:
    """
    tax_flat : float
        Flat tax amount on all citizens
    tax_income_pct : float
        Tax amount as percent of income
    leader_pay : float
        Amount of money to pay leadership 
    leader_punish_pay_threshold: float
    leader_punish_amount : float
    
    
    """
    
    
    # defense_budget : int
    tax_flat : float
    tax_income_pct : float
    leader_pay : float
    leader_punish_pay_threshold: float
    leader_punish_amount : float
    
    
class LeaderPolicies(LeaderPolicy):
    
    
    def dataframe(self):
        d = {}
        d['tax_flat'] = self.tax_flat
        d['tax_income_pct'] = self.tax_income_pct
        d['leader_pay'] = self.leader_pay
        d['leader_punish_pay_threshold'] = self.leader_punish_pay_threshold
        d['leader_punish_amount'] = self.leader_punish_amount
        return pd.DataFrame(data=d)    
    

@dataclass
class Citizen:
    """
    income : float
        Base amount of income citizen generates per turn.
        
    money : float
        Current amount of wealth.
    
    health_pts : float
        Current amount of health points (max 100). Citizen dies at zero.
    health_pt_cost : float
        Money cost of health point.
    health_pt_loss : float
        Amount of health points lost per turn. Citizen must consume money 
        in order to regain lost health. 
        
    age : float
        Current age
    max_lifespan : float
        Maximum citizen age before death
        
    leader_policy : LeaderPolicy
        Policies citizen would implement if he were a leader. 
    """
    
    income : float
    money : float

    health_pts : float
    health_pt_loss : float
    health_pt_cost : float

    age : float
    max_lifespan : float
    
    leader_policy : LeaderPolicies
    
    
    
class CitizenDataFrameType(pd.DataFrame):
    income : float
    money : float

    health_pts : float
    health_pt_loss : float
    health_pt_cost : float

    age : float
    max_lifespan : float



class Citizens(Citizen):
    """Vectorized Citizen.\n"""
    
    
    
    def __len__(self):
        return len(self.income)
    
    
    def dataframe(self) -> CitizenDataFrameType:
        d = {}
        d['income'] = self.income
        d['money'] = self.money
        d['health_pts'] = self.health_pts
        d['health_pt_loss'] = self.health_pt_loss
        d['health_pt_cost'] = self.health_pt_cost
        d['age'] = self.age
        d['max_lifespan'] = self.max_lifespan
        return pd.DataFrame(data=d)
    
    
    
    def next(self, income, hp_loss, ):
        money = self.money + income
        
        health_pts = self.health_pts - hp_loss - self.health_pt_loss
        age = self.age + 1 
        
        # Calculate money needed to heal
        damage = 100 - health_pts
        heal_cost = self.health_pt_cost * damage
        
        # Calculate which citizens cannot fully heal
        health_spend = np.minimum(heal_cost, money)
        health_spend = np.maximum(health_spend, 0)
        healing = health_spend / self.health_pt_cost
        health_pts = health_pts + healing
        
        return Citizens(income=self.income,
                        money = money,
                        health_pts = health_pts,
                        health_pt_loss = self.health_pc_loss,
                        health_pt_cost = self.health_pt_cost,
                        age = age,
                        max_lifespan = self.max_lifespan
                        
                        )
        
        
    
        
    
    @staticmethod
    def gen_uniform1(size: int, rng: RandomState=None):
        distribution = income.IncomeDistribution(
            mean = 1.0,
            median = 61713 / 86220,
            random_state = rng,
            )
        # Set Citizen Attributes
        incomes = distribution.rvs(size)
        moneys = rng.uniform(low=0, high=50, size=size)

        health_pts = np.ones(size) * 100
        health_pt_loss = rng.uniform(low=10, high=40, size=size)
        health_pt_cost = np.ones(size) * 20
        
        median_income = distribution.median
        
        ages = rng.uniform(low=0, high=65, size=size)
        max_lifespan = rng.uniform(40, 80, size=size)
        # Set Leadership Policy Attributes
        tax_flat = incomes * rng.uniform(low=0, high=0.5, size=size)
        tax_income_pct = rng.uniform(low=0, high=50, size=size)
        
        leader_pay = rng.uniform(
            low = median_income * 0.5,
            high = median_income * 10,
            size = size,
            )
        leader_punish_pay_threshold = rng.uniform(
            low = 0, 
            high = median_income * 10,
            size = size
            )
        leader_punish_amount = rng.uniform(
            low = 0.5,
            high = median_income * 10, 
            size = size)
        
        leader_policy = LeaderPolicies(
            tax_flat = tax_flat, 
            tax_income_pct = tax_income_pct, 
            leader_pay = leader_pay, 
            leader_punish_pay_threshold = leader_punish_pay_threshold, 
            leader_punish_amount = leader_punish_amount)
        
        
        citizens = Citizens(
            income=incomes,
            money = moneys,
            health_pts = health_pts,
            health_pt_loss = health_pt_loss,
            health_pt_cost = health_pt_cost,
            age = ages,
            max_lifespan = max_lifespan,
            leader_policy = leader_policy,
            )
        return citizens
    

class Government:
    def __init__(self, num_leaders: int, citizens: Citizens, rng: RandomState):
        self.num_leaders = num_leaders
        self.rng = rng
        self.citizens = citizens
        
        
    @cached_property
    def _leader_locs(self):
        """Leader index locations of Citizens."""
        num = len(self.citizens)
        locs = self.rng.choice(num, size=self.num_leaders, replace=False)
        return locs
    
    
    @cached_property
    def leader_policies(self) -> pd.DataFrame:
        policy = self.citizens.leader_policy
        
        df = policy.dataframe()
        df = df.iloc[self._leader_locs]
        return df
    
    
    @cached_property
    def median_policy(self) -> LeaderPolicySeriesType:
        return self.leader_policies.median()
        



    
class Model2:
    def __init__(self, num_citizens, num_leaders):
        self._rng = RandomState(seed=0)
        self._citizens = Citizens.gen_uniform1(
            num_citizens, 
            rng=self._rng
            )
        self._government = Government(
            num_leaders = num_leaders, 
            citizens = self._citizens,
            rng = self._rng
            )
        
    @property
    def government(self):
        return self._government
        
    
    @property
    def citizens(self):
        return self._citizens
    
    def next(self):
        policy = self.government.median_policy
        
        
        df_citizens = self.citizens.dataframe()
        
        income = df_citizens.income
        money = df_citizens.money
        money = money + income
        
        

class ModelState:
    def __init__(self, citizens: Citizens, government: Government, rng: RandomState):
        self.citizens = citizens
        
        self.government = government
        self.rng = rng
        
        self.policy = self.government.median_policy
        
        
    def next(self):
        income = self.citizens.income
        money = self.citizens.money - income
        

        
    @cached_property    
    def tax_citizen(self):
        income = self.citizens.income
        tax_flat = self.policy.tax_flat
        tax_income_pct = self.policy.tax_income_pct
        tax = tax_flat + income * tax_income_pct / 100        
        return tax
    
    
    @cached_property
    def tax_net_revenue(self):
        return self.tax_citizen.sum()    
    
    
    @cached_property
    def leader_net_cost(self):
        cost = self.policy.leader_pay.sum()
        if cost > self.tax_net_revenue:
            cost = self.tax_net_revenue
        return cost
            
    
    @cached_property
    def leader_salary(self):
        return self.leader_net_cost / self.government.num_leaders
  
    def crime_model(self):
        
        prevention_pdf_loc = 0.2
        prevention_pdf_scale = 0.5

        
        budget = self.tax_net_revenue - self.leader_net_cost
        budget_per_person = budget / len(self.citizens)
        
        rng = self.rng

        
        prevention_prob = norm.pdf(
            x = budget_per_person,
            loc = prevention_pdf_loc, 
            scale = prevention_pdf_scale
            )
        
        

    
    

    
        

        
    


    
    
    
if __name__ == '__main__':
    rng = RandomState(seed=0)
    citizens = Citizens.gen_uniform1(100, rng=rng)
    gov = Government(10, citizens=citizens, rng=rng)
    