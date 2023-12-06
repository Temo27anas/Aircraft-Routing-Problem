import pandas as pd
import numpy as np
import pulp

# Load the distance data and demand data
# For demonstration purposes, we'll assume the data is already loaded in distance_df and demand_df DataFrames
# Replace 'path_to_distance_matrix.csv' and 'path_to_demand_matrix.csv' with actual file paths
distance_df = pd.read_csv('data/airport_distance_matrix.csv', index_col=0)
demand_df = pd.read_csv('data/demand.csv', index_col=0)
#print(distance_df)
#print(demand_df)

# Numeric data provided
fleet = 6
bt = 10  # hours/day
seats = 70
lto = 0.33  # min
speed = 510  # Km/h
load_factor = 0.8
cask = 1.2  # Mad
Yield = 1.6  # Mad/RPK


# Create the linear programming problem
prob = pulp.LpProblem("Airline_Profit_Maximization", pulp.LpMaximize)

# Decision variables
#X_{ij}:the direct flow from airport i to airport j.
x_vars = pulp.LpVariable.dicts("x", ((i, j) for i in demand_df.index for j in demand_df.columns), lowBound=0, cat='Integer',)
#W_{ij}:the flow from airport i to airport j that transfers at the hub
w_vars = pulp.LpVariable.dicts("w", ((i, j) for i in demand_df.index for j in demand_df.columns), lowBound=0, cat='Integer')
#Z_{ij}: the number of flights from airport i to airport j.
z_vars = pulp.LpVariable.dicts("z", ((i, j) for i in demand_df.index for j in demand_df.columns), lowBound=0, cat='Integer')



# Objective Function \text{Maximize Profits} = \sum_{i \in N} \sum_{j \in N} [\text{Yield} \cdot d_{ij} \cdot (x_{ij} + w_{ij}) - \text{CASK} \cdot d_{ij} \cdot s \cdot z_{ij}]

profits = pulp.lpSum([Yield * distance_df.loc[i, j] * (x_vars[(i, j)] + w_vars[(i, j)]) - cask * distance_df.loc[i, j] * seats * z_vars[(i, j)] for i in demand_df.index for j in demand_df.columns])
prob += profits

# Constraints
def handle_hub(i, j):
    if i == 'CMN':
        g_i = 0
    else:
        g_i = 1
    if j == 'CMN':
        g_j = 0
    else:
        g_j = 1
    return g_i, g_j

#ONE: w_{ij} \leq q_{ij} \cdot g_{i} \cdot g_{j} \quad \forall i, j \in N
for i in demand_df.index:
    for j in demand_df.columns:
        #if airport i is CMN, then g_i = 1, otherwise g_i = 0
        g_i, g_j = handle_hub(i, j)
        prob += w_vars[(i, j)] <= demand_df.loc[i, j] * g_i * g_j

#TWO: x_{ij} + w_{ij} \leq q_{ij} \quad \forall i, j \in N
for i in demand_df.index:
    for j in demand_df.columns:
        prob += x_vars[(i, j)] + w_vars[(i, j)] <= demand_df.loc[i, j]

#FixTHREE:x_{ij} + \sum_{m \in N} w_{im} \cdot (1 - g_{j}) + \sum_{m \in N} w_{mj} \cdot (1 - g_{i}) \leq z_{ij} \cdot s \cdot \text{LF} \quad \forall i, j \in N
for i in demand_df.index:
    for j in demand_df.columns:
        g_i, g_j = handle_hub(i, j)
        transfer_in = pulp.lpSum([w_vars[(i, m)] * (1 - g_j) for m in demand_df.index])
        transfer_out = pulp.lpSum([w_vars[(m, j)] * (1 - g_i) for m in demand_df.index])
        prob += x_vars[(i, j)] + transfer_in + transfer_out <= z_vars[(i, j)] * seats * load_factor, f"Capacity_Constraint_{i}_{j}"

#FOUR:\sum_{j \in N} z_{ij} = \sum_{j \in N} z_{ji} \quad \forall i \in N
for i in demand_df.index:
    prob += pulp.lpSum([z_vars[(i, j)] for j in demand_df.columns]) == pulp.lpSum([z_vars[(j, i)] for j in demand_df.columns])

#FIVE:\sum_{i \in N} \sum_{j \in N} \left(\frac{d_{ij}}{sp} + \text{LTO}\right) \cdot z_{ij} \leq \text{BT} \cdot \text{AC}
prob += pulp.lpSum([(distance_df.loc[i, j] / speed + lto)* z_vars[(i, j)] for i in demand_df.index for j in demand_df.columns]) <= bt * fleet


# Solve the problem
prob.solve()

# The status of the solution
print("Status:", pulp.LpStatus[prob.status])

# Each of the variables is printed with its resolved optimum value
for v in prob.variables():
    if v.varValue != 0:
        print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Total Profit = ", pulp.value(prob.objective))
