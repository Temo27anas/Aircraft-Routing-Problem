

import pandas as pd
demand_df = pd.read_csv('data/demand.csv', index_col=0)

#make symmetric (copy upper triangle to lower triangle)
for i in demand_df.index:
    for j in demand_df.columns:
        if i != j:
            demand_df.loc[j, i] = demand_df.loc[i, j]

demand_df.to_csv('data/demand_sym.csv')
