# -*- coding: utf-8 -*-
import pdb
from src import base
from src.base import Member, Members, Election, Session, SessionSettings
import pandas as pd
import numpy as np

m1 = Member(
    is_corrupt = False,
    will_punish = False,
    corrupt_detect_bias = 0,
    corrupt_detect_as_member = 0.5,
    corrupt_detect_as_leader = 1.0,
    )

m2 = Member(
    is_corrupt = True,
    will_punish = False,
    corrupt_detect_bias = 0,
    corrupt_detect_as_member = 0.0,
    corrupt_detect_as_leader = 1.0,
    )

m3 = Member(
    is_corrupt = True,
    will_punish = True,
    corrupt_detect_bias = 0.5,
    corrupt_detect_as_member = 0.25,
    corrupt_detect_as_leader = 1.0,
    )



members = Members([m1, m2, m3])

def test_defaults():
    e = Election(members, 1, )
    assert len(e.last_leaders) == 0
    
def test_remove_probabilities():
    
    rstate = np.random.default_rng(0)
    
    
    last_leaders = np.array([1])
    e = Election(members, 1, True, last_leaders, rstate=rstate)
    probabilities = e.remove_probabilities
    assert np.isclose(probabilities[0, 0], 0.5)
    assert np.isclose(probabilities[0, 1], 0.0)
    assert np.isclose(probabilities[0, 2], 0.75)
    

def test_election():
    
    rstate = np.random.default_rng(0)
    last_leaders = np.array([1])
    
    e = Election(members, 1, False, last_leaders, rstate=rstate)
    e.leaders_to_keep
    e.leaders_to_remove
    
    
    
# if __name__ == '__main__':
def test_simulation1():
    """Test simulation where leaders are always corrupt, members will never
    vote against leader."""
    m = Member(is_corrupt=True, 
               will_punish=False, 
               corrupt_detect_bias=0.0,
               corrupt_detect_as_member=0.0,
               corrupt_detect_as_leader=0.0)
    
    members = Members([m]*100)
    election = Election(members, num_win=1, )
    settings = SessionSettings(
        start_money=0, 
        member_reward=1, 
        leader_reward=2, 
        punishment=30,
        max_steal=10000,)
    
    session = Session(election=election, settings=settings)
    session.moneys2
    
    assert session._amount_to_steal[0] == 99
    assert session.will_steal == True
    assert session.will_punish == False
    assert session.moneys2[session.leaders] == 101
    assert np.all(session.moneys2[session.member_ids_not_leaders] == 0)
    
    # Step to next session
    session2 = session.next

    assert session2.election.new_leaders[0] == session.leaders[0]
    assert np.all(session2.election.remove_probabilities == 0)
    assert session2.election.votes_against.sum() == 0
    
    assert session2.will_steal == True
    assert session2.will_punish == False
    assert session2.moneys2[session.leaders] == 202
    assert np.all(session2.moneys2[session2.member_ids_not_leaders] == 0)
    

def test_simulation2():
    """Test simulation where leaders are never corrupt, but members will 
    vote randomly."""
    m = Member(is_corrupt=False, 
               will_punish=False,
               corrupt_detect_bias=0.5, 
               corrupt_detect_as_member=0.0,
               corrupt_detect_as_leader=0.0,)
    members = Members([m]*500)
    election = Election(members, num_win=10)
    settings = SessionSettings(
        start_money=0, 
        member_reward=1, 
        leader_reward=2, 
        punishment=30)    
    
    # In the initial election there is nobody to vote against
    assert election.votes_against.size == 0
    session1 = Session(election=election, settings=settings)
    assert session1.moneys1.sum() == 0
    assert (session1.moneys2[session1.leaders] == 2).all()
    assert (session1.moneys2[session1.member_ids_not_leaders] == 1).all()
    
    session2 = session1.next
    # With lots of voters the votes for or against should be about 50/50
    assert abs(session2.election.votes_against.sum(axis=1).mean() - 250) < 5
    set_remove = set(session2.election.leaders_to_remove)
    set_new = set(session2.election.new_leaders)
    set_keep = set(session2.election.leaders_to_keep)
    assert set_remove.isdisjoint(set_new)
    assert set_keep <= set_new
    
    assert session2.will_punish == False
    assert session2.will_steal == False
    
    assert session2.moneys2.sum() == session1.moneys2.sum()*2
    

    
    
if __name__ == '__main__':    
    import logging
    logging.basicConfig(level = logging.DEBUG)
    # logger = logging.getLogger()
    
    
    test_simulation2()
    test_remove_probabilities()
    test_election()
    test_simulation1()

