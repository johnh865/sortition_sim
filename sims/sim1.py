# -*- coding: utf-8 -*-
"""Sim1

Simple simulation testing 4 possible personality types.

"""
import pdb, traceback, sys, code
import logging

from src import base

logging.basicConfig(level=logging.DEBUG)

bias = 2.0
m_detect = 0.1
l_detect = 0.9

member1  = base.Member(
    is_corrupt = True, 
    will_punish = True, 
    corrupt_detect_bias = bias,
    corrupt_detect_as_member = m_detect,
    corrupt_detect_as_leader = l_detect, 
    )

member2  = base.Member(
    is_corrupt = True, 
    will_punish = False, 
    corrupt_detect_bias = bias,
    corrupt_detect_as_member = m_detect,
    corrupt_detect_as_leader = l_detect, 
    )

member3  = base.Member(
    is_corrupt = False, 
    will_punish = True, 
    corrupt_detect_bias = bias,
    corrupt_detect_as_member = m_detect,
    corrupt_detect_as_leader = l_detect, 
    )

member4  = base.Member(
    is_corrupt = False, 
    will_punish = False, 
    corrupt_detect_bias = bias,
    corrupt_detect_as_member = m_detect,
    corrupt_detect_as_leader = l_detect, 
    )


def generate_members(n1, n2, n3, n4):
    m1 = member1.uniform(n1)
    m2 = member2.uniform(n2)
    m3 = member3.uniform(n3)
    m4 = member4.uniform(n4)
    return base.Members.concat([m1, m2, m3, m4])


members = generate_members(25, 25, 25, 25)
election = base.Election(members, 11,)
settings = base.SessionSettings()

sessions = base.Sessions(
    members = members, 
    election = election, 
    settings = settings,
    num = 1000).run()

h = sessions.history.votes
a = sessions.history.actions
terms  = sessions.terms