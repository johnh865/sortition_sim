import polars as pl
import numpy as np 

d = {}
x = np.linspace(0, 1, 11)
y = x**2
z = np.sin(x)
d['i'] = np.arange(len(x))
d['x'] = x
d['y'] = y
d['z'] = z
df = pl.DataFrame(d)

filter1 = pl.col('i').is_between(4, 6)


df2 = df.filter(filter1)



df.get_column('i')

df3 = df.with_columns([(pl.col('x') + 42)])