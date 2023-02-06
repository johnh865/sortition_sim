# -*- coding: utf-8 -*-

import numpy as np 
from dataclasses import dataclass

TILE_LAND = 1
TILE_WATER = 0
TILE_MOUNTAIN = 2


@dataclass
class Tile:
    xloc : int
    yloc : int
    food_power : int
    lumber_power : int
    type : int
    
    

    