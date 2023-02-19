# -*- coding: utf-8 -*-

import numpy as np 
from dataclasses import dataclass
import polars as pl
import pandas as pd
import pdb


TILE_LAND = 1
TILE_WATER = 0
TILE_MOUNTAIN = 2


TILE_COLUMN_TYPES = 0



@dataclass
class Tile:
    xloc : int
    yloc : int
    food_power : int
    lumber_power : int
    type : int
    
    
@dataclass
class Tiles:
    df : pd.DataFrame
    
    @property
    def xloc(self):
        return self.df.get_column('xloc')
    
    
    @property
    def yloc(self):
        return self.df.get_column('yloc')
    
    
    
def test_world_tiles():
    rows = 10
    cols = 10
    types = np.zeros((rows, cols), dtype=int)
    foods = np.zeros((rows, cols), dtype=int)
    lumbers = np.zeros((rows, cols), dtype=int)


    types[4:7, 4:7] = 1
    foods[4:6, 4:6] = 40
    lumbers[5:7, 4:5] = 60
    
    iarr = np.arange(rows)
    yarr = np.arange(cols)
    xg, yg = np.meshgrid(iarr, yarr)
    
    d = {}
    d['xloc'] = xg.ravel()
    d['yloc'] = yg.ravel()
    d['type'] = types.ravel()
    d['food_power'] = foods.ravel()
    d['lumber_power'] = lumbers.ravel()
    
    df = pd.DataFrame(d)
    tiles = Tiles(df)
    
    pdb.set_trace()
    
#test_world_tiles()