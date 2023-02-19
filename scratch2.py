# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import pandas as pd


d = {}
x = np.arange(10)
y = x **2
z = np.sin(x)
d['x'] = x
d['y'] = y
d['z'] = z

df = pd.DataFrame(d)


