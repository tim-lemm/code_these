import numpy as np
import pandas as pd

def generate_od_matrix (
        size_od:int,
        od_scenario:str = "RANDOM_OD",
        seed:int = 69,
        max_demand: int = 1000
)-> pd.DataFrame:
    if od_scenario not in ["RANDOM_OD", "1OD","2OD"]:
        raise ValueError("Invalid OD scenario. You can choose between RANDOM_OD, 1OD, 2OD")
    np.random.seed(seed)
    od_matrix = pd.DataFrame(0, index=range(1, size_od), columns=range(1,size_od))
    if od_scenario == "1OD":
        od_matrix.loc[1, 16] = max_demand
    elif od_scenario == "2OD":
        od_matrix.loc[1, 16] = max_demand
        od_matrix.loc[13, 4] = max_demand
    elif od_scenario == "RANDOM_OD":
        list_i = [1, 2, 3, 4, 5, 8, 9, 12, 13, 15, 14, 16]
        for i in list_i:
            for j in list_i:
                if i != j:
                    od_matrix.loc[i, j] = np.random.randint(20, max_demand)
    return od_matrix