import numpy as np
import pandas as pd

def generate_od_df (
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

def convert_od_df_to_matrix(od_df: pd.DataFrame)-> np.ndarray:
    size_od = len(od_df)+1
    od_matrix = np.zeros((size_od, size_od))
    for origin in od_df.index:
        for destination in od_df.index:
            od_matrix[origin, destination] = od_df.loc[origin, destination]
    return od_matrix

def convert_od_matrix_to_df(od_matrix: np.ndarray)-> pd.DataFrame:
    size_od = len(od_matrix)
    od_df = pd.DataFrame(index=range(1, size_od), columns=range(1, size_od))
    for origin in od_df.index:
        for destination in od_df.index:
            od_df.loc[origin, destination] = od_matrix[origin, destination]
    return od_df