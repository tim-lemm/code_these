import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from plotting import *
from od_matrix_generator import *
from eaquilibrea_interface import *
from network_processing import *
from src.utils_sta import ta_due, ta_stochastic
from traffic import mode_choice

CURRENT_DIR = ""

edge_df, node_df = import_network(CURRENT_DIR + "data/edges_small_grid_2.csv", CURRENT_DIR + "data/nodes_small_grid_2.csv")

list_od_scenarios=['1OD','2OD',"RANDOM_OD"]
list_max_demand = [500,1000,2000,3000,4000,5000]

### Processing sensitivity analysis on ta_due
list_algorithm = ['bfw', 'fw', 'msa']

# df_results = pd.DataFrame()
# for algorithm in list_algorithm:
#     for scenario in list_od_scenarios:
#         for max_demand in list_max_demand:
#             name = f"{algorithm}_{scenario}_{max_demand}"
#             od_df = generate_od_df(17, od_scenario=scenario, max_demand=max_demand)
#             od_df_eaq = convert_to_eaquilibrae_od_matrix(od_df)
#             results_ta_due = ta_due(edge_df,
#                                     od_df_eaq,
#                                     algorithm=algorithm,
#                                     max_iter=500,
#                                     tolerance=1e-4,
#                                     time_field='free_flow_time_car',
#                                     cost_field='free_flow_time_car',
#                                     capacity_field='capacity_cars',
#                                     verbose=True)
#             df_results[name] = results_ta_due

# df_results.to_json(f"{CURRENT_DIR}output/sensitivity_due.json")

###processing sensitivity analysis on ta_sto

# list_algorithm = ['bfsle', 'lp']
# list_max_route = [1,2,3,4,5]
# df_results = pd.DataFrame()
# for algorithm in list_algorithm:
#     for scenario in list_od_scenarios:
#         for max_demand in list_max_demand:
#
#             od_df = generate_od_df(17, od_scenario=scenario, max_demand=max_demand)
#             od_df_eaq = convert_to_eaquilibrae_od_matrix(od_df)
#             for max_route in list_max_route:
#                 name = f"{algorithm}_{max_route}_{scenario}_{max_demand}"
#                 results_ta_sto = ta_stochastic(edge_df,
#                                                od_df_eaq,
#                                                mode='bikes',
#                                                time_field='free_flow_time_bike',
#                                                cost_field='free_flow_time_bike',  ### LENGTH OR LENGTH_BI?
#                                                algorithm=algorithm,
#                                                max_routes=max_route,
#                                                capacity_field='capacity_bikes',
#                                                verbose=True)
#                 df_results[name] = results_ta_sto
#
# df_results.to_json(f"{CURRENT_DIR}output/sensitivity_sto.json")


### getting total demand from the different scenario of OD

# df_demand = pd.DataFrame(columns=['scenario','max_demand','total_demand'])
#
# rows_list = []
# for scenario in list_od_scenarios:
#     for max_demand in list_max_demand:
#         od_matrix = generate_od_df(17, scenario, max_demand=max_demand)
#         rows_list.append({
#             'scenario': scenario,
#             'max_demand': max_demand,
#             'total_demand': od_matrix.sum().sum()
#         })
# df_demand = pd.DataFrame(rows_list)
#
# df_demand.to_json(f"{CURRENT_DIR}output/demand.json")

### Processing mode_choice sensitivity
# parameters for mode choice
list_asc_bike = [-2.5,-1,0]
beta_time = -0.01
ASC_car = 0
mu_mode = 1.0
max_iter_mode_choice = 5
plot = False
size_od = 17

# df_results_mc = pd.DataFrame()
#
# for ASC_bike in list_asc_bike:
#     for scenario in list_od_scenarios:
#         for max_demand in list_max_demand:
#             name = f"{scenario}_{max_demand}_{ASC_bike}"
#             od_df = generate_od_df(size_od, od_scenario=scenario, max_demand=max_demand)
#             od_df_eaq = convert_to_eaquilibrae_od_matrix(od_df)
#
#             result_df,_,_ = mode_choice(edge_df,
#                                         node_df,
#                                         od_df,
#                                         beta_time=beta_time,
#                                         mu_mode=mu_mode,
#                                         max_iter_mode_choice=max_iter_mode_choice,
#                                         ASC_bike=ASC_bike,
#                                         plot=plot)
#             df_results_mc[name] = {"results_df":result_df}
#
# df_results_mc.to_json(f"{CURRENT_DIR}output/sensitivity_mc.json")


max_demand = 5000
df_results_mc_beta = pd.DataFrame()
list_beta_time = [-0.0001, -0.001, -0.01, -0.1]
for scenario in list_od_scenarios:
    for ASC_bike in list_asc_bike:
        for beta_time in list_beta_time:
            name = f'{scenario}_{max_demand}_{ASC_bike}_{beta_time}'
            od_df = generate_od_df(size_od, od_scenario=scenario, max_demand=max_demand)
            od_df_eaq = convert_to_eaquilibrae_od_matrix(od_df)

            result_df, _, _ = mode_choice(edge_df,
                                          node_df,
                                          od_df,
                                          beta_time=beta_time,
                                          mu_mode=mu_mode,
                                          max_iter_mode_choice=max_iter_mode_choice,
                                          ASC_bike=ASC_bike,
                                          plot=plot)
            df_results_mc_beta[name] = {"results_df":result_df}

df_results_mc_beta.to_json(f"{CURRENT_DIR}output/sensitivity_mc_beta.json")

#TODO: test with different version of mode_choice (order, weights etc...)