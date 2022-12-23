# -*- coding: utf-8 -*-

import pandas as pd
import pandera as pa
from pandera.typing import Series
from typing import Protocol



class Type1(Protocol):
    def meth(self) -> int:
        ...
        
    @property
    def x(self) -> int:
        ...

class Type2(Protocol):
    x : int
    y : int

def mow(a : Type2):
    a.x
    a.y    
        
        