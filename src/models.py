"""
Accountability Behaviors
------------------------

* Level 1: Dumb punishment
 - Voter will vote against all leaders if steal vote is successful.  
* Level 2: Smart punishment
 - Voter will vote against specific leaders that voted to steal. 
* Level 3: Proactive punishment
 - Voter will not vote for leaders that have `is_corrupt` personality. 

Leader Behaviors
----------------
* Leaders will vote steal only when they have enough votes to do it. 

"""

import pdb
from dataclasses import dataclass
from functools import cached_property
from typing import Union
import logging

import numpy as np
import pandas as pd 

DEFAULT_RNG = np.random.default_rng(0)

logger = logging.getLogger(__name__)

    
def leader_remove_probability(
        
        is_corrupt: bool, 
        ability: float,
        bias: float,
        
        ) -> float:
    """Member probability to vote against corrupt leader. 

    Parameters
    ----------
    is_corrupt : bool
        Corrupt status of leader.
    ability : float
        Detection ability of member.
    base : float
        Member base probability to vote for removal .

    Returns
    -------
    probability : float
        Member probability to vote for leader removal from [0 to 1].
    """
    detection = (2 * is_corrupt - 1.0) * ability
    probability = bias + detection
    probability = np.clip(probability, a_min=0, a_max=1.0)    
    return probability
    
 


@dataclass
class Member:
    """A member of a democratic collective. Store immutable personality 
    properties of the member. 
    
    is_corrupt : bool
        As leader, will always steal if True, or not if False.
        
    will_punish : bool
        As leader, will punish stealers if True, or forgive if False
        
    corrupt_detect_as_member : float
        Corruption detection ability from 0 to 1 (max).
        
    corrupt_detect_as_leader : float
        Corruption detection ability from 0 to 1 (max). 
        
    corrupt_detect_bias : float
        Corruption detection probability bias from 0 to 1.
         - At 0.0, will always re-elect.
         - At 0.5, no bias 
         - At 1.0, will always punish.
         
    """    
    
    is_corrupt : bool
    will_punish : bool
    
    corrupt_detect_bias : float
    corrupt_detect_as_member : float
    corrupt_detect_as_leader : float
    
    
    # personality: Personality

    # is_leader: bool = False
    # was_last_leader: bool = False
    
    @classmethod
    def _field_types(cls):
        fields = cls.__dataclass_fields__
        return {key : fields[key].type for key in fields}
    
    
    def uniform(self, num: int) -> 'Members':
        """Members : Create uniform Members out of this template Member."""
        return Members([self] * num)


class Members:
    """Store members data into numpy arrays and dataframes. 
    
    Parameters
    ----------
    members : list[Member], optional
        Input individual members attributes. The default is None.
    df : pd.DataFrame, optional
        Input if a DataFrame of member attributes are available. 
        The default is None.
        
    Attributes
    ----------
    See `Member` for list of descriptions of vectorized attributes of `Member`.
    
    dataframe : pandas.DataFrame
        Member attributes collected into DataFrame
        
    member_num : int
        Number of members `m`
        
    member_index : ndarray[int] shape(m,)
        Member unique ID's.
    """    
    
    _IS_CORRUPT = 'is_corrupt'
    _WILL_PUNISH = 'will_punish'
    _CORRUPT_DETECT_BIAS = 'corrupt_detect_bias'
    _CORRUPT_DETECT_AS_MEMBER = 'corrupt_detect_as_member'
    _CORRUPT_DETECT_AS_LEADER = 'corrupt_detect_as_leader'
    _DF_TYPES = Member._field_types()
    

    
    def __init__(self, members: list[Member]=None, df: pd.DataFrame=None):

        
        if df is None:
            df = pd.DataFrame(members)
        self.dataframe = df
        
        self.is_corrupt = df['is_corrupt'].values
        self.will_punish = df['will_punish'].values
        self.corrupt_detect_bias = df['corrupt_detect_bias'].values
        self.corrupt_detect_as_member = df['corrupt_detect_as_member'].values
        self.corrupt_detect_as_leader = df['corrupt_detect_as_leader'].values
        
        self.member_num = len(df)
        self.member_index = df.index.values
        
     
    def __len__(self):
        return self.member_num
    
    
    # @classmethod
    # def uniform(cls, num: int, is_corrupt: bool, will_punish: bool, 
    #             corrupt_detect_bias, corrupt_detect_as_member, 
    #             corrupt_detect_as_leader, 
    #             ):
    #     df = {}
    #     ones = np.ones(num)
        
    #     df[cls._IS_CORRUPT] = ones * is_corrupt
    #     df[cls._WILL_PUNISH] = ones * will_punish
    #     df[cls._CORRUPT_DETECT_BIAS] = ones * corrupt_detect_bias
    #     df[cls._CORRUPT_DETECT_AS_MEMBER] = ones * corrupt_detect_as_member
    #     df[cls._CORRUPT_DETECT_AS_LEADER] = ones * corrupt_detect_as_leader
        
    #     df = cls.validate_df(df)
    #     return Members(df=df)
        
    
    @classmethod
    def validate_df(cls, df: pd.DataFrame):
        return df.astype(cls._DF_TYPES)
    
    
    @classmethod
    def concat(cls, a: list['Members']) -> 'Members':
        """Concatenate several `Members` together into new `Members`."""
        dataframes = [m.dataframe for m in a]
        df = pd.concat(dataframes, ignore_index=True)
        return Members(df=df)
  
    
class Election:
    """Run an election.
    
    Parameters
    ----------
    members : Members
        Members input data.
    num_win : int
        Number of winners in an election.
    stole : bool, optional
        Whether or not previous leaders stole. The default is False.
    last_leaders : np.ndarray, optional
        Member ID's of last leaders. The default is None.
    rstate : np.random.Generator, optional
        Random number generator. The default is DEFAULT_RNG.
    disqualified :  np.ndarray[int], optional
        Member ID's of members that are disqualified from running in election.
    
        The default is (). 
    """
    
    def __init__(self,
        members: Members,
        num_win: int,
        stole: bool = False,
        last_leaders: np.ndarray = None,
        rstate: np.random.Generator = DEFAULT_RNG,
        disqualified: Union[bool, np.ndarray] = None, 
        ):
        self.members = members
        self.stole = stole
        
        if last_leaders is None:
            last_leaders = np.array([], dtype=int)
        if disqualified is None:
            disqualified = np.array([], dtype=int)
            
        self.disqualified = disqualified
        self.last_leaders = last_leaders
        self.num_win = num_win
        self.rstate = rstate
        
        self.vote_threshold = len(members) // 2 + 1
        
        # Generate the properties that need random numbers for determinism.
        self.rolls
        # self.new_leaders
        
        
        # 


    @cached_property
    def remove_probabilities(self) -> np.ndarray:
        """array[float] shape (l, m) : Probability [0 to 1] that member `m` 
        will vote against leader `l`."""
        members = self.members
        leaders = self.last_leaders 
        bias = members.corrupt_detect_bias
        
        # Return empty array if no existing leaders
        if len(leaders) == 0:
            return np.zeros((0, len(members)))
        
        # Assume Level 2 accountability. 
        # Detection can only activate if leaders successfully stole. 
        if not self.stole:
            prob = np.clip(bias, a_min=0.0, a_max=1.0)
            probabilities = np.ones((len(leaders), 1))
            probabilities = probabilities * prob[None, :]
            return probabilities    
        
        # Generate probabilties
        corruption = members.is_corrupt
        leader_corruption = corruption[leaders]
        ability = members.corrupt_detect_as_member
        
        probabilities = []
        for is_corrupt in leader_corruption:
            
            prob = leader_remove_probability(is_corrupt, ability, bias)
            
            # leaders never vote against themselves. Set their prob = 0
            prob[leaders] = 0
            probabilities.append(prob)
        return np.array(probabilities)
    
    
    @cached_property
    def rolls(self):
        """array[float] shape (l, m) : Random numbers to compare to remove 
        probabilities."""
        rolls = self.rstate.uniform(
            
            low = 0, 
            high = 1,
            size = (len(self.last_leaders), len(self.members))
            
            )
        return rolls
    
    
    @cached_property
    def votes_against(self) -> np.ndarray:
        """array[bool] shape (l, m) : Votes cast against leaders."""
        probabilities = self.remove_probabilities
        rolls = self.rolls
        return rolls < probabilities
    
    
    @cached_property
    def vote_percentages(self) -> np.ndarray:
        """array[int] shape(l,) : Vote count against leaders."""
        if self.votes_against.size == 0:
            return np.zeros(self.num_win)
        
        return self.votes_against.sum(axis=1) / len(self.members)
    
    
    @cached_property
    def _leaders_disqualified_bool(self):
        """array[bool] shape (l,) : Leaders to be disqualified."""

        member_index = np.intersect1d(self.last_leaders , self.disqualified)
        leader_index = np.searchsorted(self.last_leaders, member_index)
        out = np.zeros(len(self.last_leaders), dtype=bool)
        
        out[leader_index] = True
        return out
    
    
    @cached_property
    def _leaders_to_remove_bool(self):
        """array[bool] shape (l,) : Leader should be removed if True"""
        
        votes_against = self.votes_against
        vote_count = votes_against.sum(axis=1)
        to_remove = vote_count >= self.vote_threshold

        # Also remove disqualified 
        to_remove = to_remove | self._leaders_disqualified_bool
        return to_remove
    
    
    @cached_property
    def leaders_to_remove(self):
        """array[int] : Member ID of leaders to be removed from office."""
        return self.last_leaders[self._leaders_to_remove_bool]
    
    
    
    # @cached_property
    # def removal_accuracy(self):
    
    
    @cached_property
    def leaders_to_keep(self):
        """array[int] : Member ID of leaders to be kept."""
        return self.last_leaders[~self._leaders_to_remove_bool]
    
    
    @cached_property
    def corrupt_reelected(self):
        """array[int] : Member ID of corrupt leaders to be kept."""
        kept = self.leaders_to_keep
        is_corrupt = self.members.is_corrupt[kept]
        return kept[is_corrupt]
    
    
    @cached_property
    def new_leaders(self):
        """array[int] : Member ID of new leaders to select at random."""
        
        # Eligible members for election
        member_index = self.members.member_index
        member_index = np.setdiff1d(member_index, self.disqualified)
        member_index = np.setdiff1d(member_index, self.last_leaders)
        
        replace_num = self.num_win - len(self.leaders_to_keep)
        if replace_num > 0:
            rand_leaders = self.rstate.choice(
               
                member_index, 
                size = replace_num, 
                replace = False
                )
            out = np.append(self.leaders_to_keep, rand_leaders)
        else:
            out = self.leaders_to_keep
        return np.sort(out)


@dataclass
class SessionSettings:    
    
    start_money: float = 0.0
    member_reward : float = 1
    leader_reward : float = 2
    # steal_reward : float = 10
    punishment : float = 30
    term_limit : int = 25
    
    

class Session:
    """Run actions the leaders will perform."""
    def __init__(self,
                 election: Election,
                 settings: SessionSettings,
                 moneys: np.ndarray=None,
                 terms: np.ndarray=None,
                 ):
        
        self.election = election
        self.settings = settings
        self.members = election.members
        self.leaders = election.new_leaders
        self.num_win = election.num_win

        if moneys is None:
            self.moneys1 = np.ones(len(self.members)) * settings.start_money
        else:
            self.moneys1 = moneys
            
        if terms is None:
            terms = np.zeros(len(self.members), dtype=int)
        
        self.last_leaders = election.last_leaders
        self.last_stole = election.stole
        self.last_terms = terms


            
    @cached_property   
    def disqualified(self):
        """Member ID's disqualified from future elections."""
        return np.nonzero(self.terms >= self.settings.term_limit)[0]
    
    
    @cached_property   
    def terms(self):
            
        out = self.last_terms.copy()
        leaders = self.election.new_leaders
        out[leaders] += 1             
        return out
    
            
    @cached_property   
    def next(self):
        """Session : Step into next election and session."""
        election2 = Election(
            
            members = self.members, 
            num_win = self.num_win,
            stole = self.will_steal,
            last_leaders = self.leaders,
            rstate = self.election.rstate,
            disqualified=self.disqualified
            )
        
        new_terms = self.terms % self.settings.term_limit
        session2 = Session(
            
            election = election2, 
            settings = self.settings,
            moneys = self.moneys2,
            terms = new_terms,
            
            )
        return session2
    
    
       
    @cached_property
    def _amount_to_steal(self):
        """float : Amount to steal from each member if `self.will_steal`==True."""
        member_num = len(self.members) - self.num_win

        income = member_num * self.settings.member_reward
        return income / self.num_win
            
            
    @cached_property    
    def member_ids_not_leaders(self):
        """ndarray[int] : Member ID's that are not leaders."""
        return np.delete(self.members.member_index, self.leaders)
    
    
    @cached_property
    def rewards(self):
        """ndarray[float] shape(m,) : Reward for each member at end of session."""
        leader_reward = self.settings.leader_reward
        member_reward = self.settings.member_reward
        member_ids = self.member_ids_not_leaders
        punishment = self.settings.punishment
        leaders = self.leaders
        
        out = np.zeros(len(self.members))
        
        # Leaders steal from workers 
        if self.will_steal:        
            out[leaders] = self._amount_to_steal + leader_reward
            
        else:
            out[leaders] = leader_reward
            out[member_ids] = member_reward
            
        # Leaders vote to punish former stealers. 
        if self.will_punish:
            punish_ids = self.leaders_to_punish
            out[punish_ids] = -punishment - self._amount_to_steal
            
        return out

            
    
    @cached_property
    def moneys2(self):
        """ndarray[float] shape(m,) : Money balance for each member at end of session."""
        return self.moneys1 + self.rewards


    @cached_property
    def votes_to_steal(self):
        """ndarray[int] : Member ID's of corrupt leaders who vote to steal."""
            
        mask = self.members.is_corrupt[self.leaders]
        return self.leaders[mask]
    
        
    @cached_property
    def votes_to_punish(self):
        """ndarray[int] : Will leader vote to punish?"""
        if self.last_stole:
            
            # Leaders will not punish themselves
            # former_leaders = self.election.last_leaders
            # former_corruption = self.members.is_corrupt[former_leaders]
            former_corrupted = self.election.corrupt_reelected
            leaders2 = np.setdiff1d(self.leaders, former_corrupted)
            
            mask = self.members.will_punish[leaders2]
            
            return leaders2[mask]
        else:
            return np.array([], dtype=int)
        
        
    @cached_property
    def will_steal(self):
        """bool : Determine if Sesssion will vote to steal."""
        return self.votes_to_steal.size / len(self.leaders) > 0.5
    
    
    @cached_property
    def will_punish(self):
        """bool : Determine if Session will vote to punish last leaders."""
        return self.votes_to_punish.size / len(self.leaders) > 0.5
    
    
    @cached_property
    def _leaders_to_punish_vote(self):
        """Election : Calculations for punishment vote."""
        
        # To calculate leaders to punish, use data from pseudo Election.
        num_win = self.election.num_win
        last_leaders = self.election.last_leaders
        df = self.members.dataframe.iloc[self.leaders]
        members1 = Members(df=df)
        
        # Swap in leadership corruption detection
        members1.corrupt_detect_as_member = members1.corrupt_detect_as_leader
    
        e1 = Election(members1, num_win=num_win, last_leaders=last_leaders)
        return e1
    
    
    @cached_property
    def leaders_to_punish(self):
        """ndarray[int] shape(l,) : Former leaders who will be punished."""
        if self.will_punish:
            return self._leaders_to_punish_vote.leaders_to_remove
        else:
            return np.array([], dtype=int)
            
        

class Sessions:
    def __init__(self,
                 members: Members, 
                 election: Election,
                 settings: SessionSettings,
                 num: int):
        self.members = members
        self.settings = settings
        
        session1 = Session(election=election, settings=settings)
        self.sessions = [session1]
        
        # Store terms served here. 
        self.num = num
        self.terms = np.zeros((num + 1, len(self.members)), dtype=int)
        
        
        
    def run(self):

        for ii in range(self.num):
            logger.info(
                f'Sesssion {ii}'
                 '-------------'
                )
            
            session = self.sessions[-1]                
            # leaders = session.leaders
            # limit = self.settings.term_limit
            # self.terms[ii + 1, leaders] = self.terms[ii, leaders] + 1
            
            next_session = session.next
            self.sessions.append(next_session)
            
            # After 1 disqualification reset term limit
            # self.terms[ii + 1] = self.terms[ii + 1] % limit

        # self.terms = self.terms[1:]
        return self
    
    
    def leader_history(self):
        hist = []
        for session in self.sessions:
            leaders = np.sort(session.election.new_leaders)
            hist.append(leaders)
        return np.array(hist)
    
    
    def reward_history(self):
        hist = []
        for session in self.sessions:
            r = session.rewards
            hist.append(r)
        return np.array(hist)
    
    
    def action_history(self):
        df = pd.DataFrame()
        sessions = self.sessions
        df['will_punish'] = [s.will_punish for s in sessions]
        df['will_steal'] = [s.will_steal for s in sessions]
        df['num_removed'] = [len(s.election.leaders_to_remove) for s in sessions]
        return df
    

    
    def vote_history(self):
        hist = []
        for session in self.sessions:
            r = session.election.vote_percentages
            hist.append(r)
        return np.array(hist)
            
    

        
        
    

    
    
    
    

