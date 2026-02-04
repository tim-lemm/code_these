import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from plotting import *
from od_matrix_generator import *
from eaquilibrea_interface import *
from network_processing import *
from src.utils_sta import ta_due

CURRENT_DIR = "/Users/tristan.lemoalle/Documents/Thèse/Code/code_these/"

edge_df, node_df = import_network(CURRENT_DIR + "data/edges_small_grid_2.csv", CURRENT_DIR + "data/nodes_small_grid_2.csv")

list_od_scenarios=['1OD','2OD',"RANDOM_OD"]
list_max_demand = [500,1000,2000,3000,4000,5000]
list_algorithm = ['bfw', 'fw', 'msa']

df_results = pd.DataFrame()
for algorithm in list_algorithm:
    for scenario in list_od_scenarios:
        for max_demand in list_max_demand:
            name = f"{algorithm}_{scenario}_{max_demand}"
            od_df = generate_od_df(17, od_scenario=scenario, max_demand=max_demand)
            od_df_eaq = convert_to_eaquilibrae_od_matrix(od_df)
            results_ta_due = ta_due(edge_df,
                                    od_df_eaq,
                                    algorithm=algorithm,
                                    max_iter=500,
                                    tolerance=1e-4,
                                    time_field='free_flow_time_car',
                                    cost_field='free_flow_time_car',
                                    capacity_field='capacity_cars',
                                    verbose=True)
            df_results[name] = results_ta_due

df_results.to_json(f"{CURRENT_DIR}output/sensitivity_due.json")