from pathlib import Path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import warnings
import logging
import sys

from scripts.plotting import plot_network, plot_od_matrix
from scripts.traffic import skimming
from utils_sta import ta_due, ta_stochastic, plot_vc_histogram
from scripts.network_processing import *
from scripts.plotting import *
from scripts.od_matrix_generator import *
from scripts.eaquilibrea_interface import *

from aequilibrae.paths import RouteChoice
from aequilibrae.matrix import AequilibraeMatrix

CURRENT_DIR = "C:/Users/Tristan/Documents/ENTPE/Thèse/code_thése/code_these/"

warnings.filterwarnings('ignore')
logging.getLogger("aequilibrae").setLevel(logging.ERROR)

edge_df = pd.read_csv(CURRENT_DIR + "data/edges_small_grid_2.csv")
node_df = pd.read_csv(CURRENT_DIR + "data/nodes_small_grid_2.csv")

edge_df = calculate_length(node_df, edge_df)
edge_df["length"] *= 10
edge_df["type_bike"] = None
edge_df["speed_bike"] /= 3.6
edge_df["speed_car"] /= 3.6
edge_df["travel_time_bike"] = edge_df["length"] / edge_df["speed_bike"]
edge_df["free_flow_time"] = edge_df["length"] / edge_df["speed_car"]
edge_df["travel_time_car"] = edge_df["free_flow_time"]
edge_df["capacity_cars"] = 3000
edge_df["capacity_bikes"] = 99999
edge_df["alpha"] = 0.15
edge_df["beta"] = 4

plot_network(edge_df, node_df,
             node_id_col='node',
             node_label=True,
             color_col_num='travel_time_car',
             base_width=1,
             legend=True,
             title="Network with Free Flow Time",
             figsize=(8, 8))
od_scenario = "RANDOM_OD"
size_od = max(node_df['node']) + 1
od_matrix = generate_od_matrix(size_od)

od_matrix_eaq = convert_to_eaquilibrae_od_matrix(od_matrix)
plot_od_matrix(od_matrix_eaq, edge_df, node_df, title=f"{od_scenario} OD Matrix on Network", figsize=(8, 8),
               label=False)

results_due = ta_due(edge_df, od_matrix_eaq, algorithm='bfw', max_iter=500, tolerance=1e-4,
                     time_field='travel_time_car', cost_field='travel_time_car', capacity_field='capacity_cars',
                     verbose=True)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plot_network(results_due['network'], node_df,
             node_id_col='node',
             node_label=True,
             color_col_num='flow',
             base_width=0.1,
             width_col='flow',
             width_scale=100,
             legend=True,
             title=f"Network with flow ({od_scenario} scenario)",
             figsize=(8, 8),
             cmap='Reds',
             vmax=results_due['network']['flow'].max(),
             vmin=results_due['network']['flow'].min(),
             ax=axes[0])
plot_network(results_due['network'], node_df,
             node_id_col='node',
             node_label=True,
             color_col_num='ratio',
             base_width=0.5,
             width_col='ratio',
             width_scale=100,
             legend=True,
             title=f"Network with v/c ratio ({od_scenario} scenario)",
             figsize=(8, 8),
             cmap='RdYlGn_r',
             vmax=results_due['network']['ratio'].max(),
             vmin=0,
             ax=axes[1])

results_sto = ta_stochastic(edge_df, od_matrix_eaq, mode='bike',
                            time_field='travel_time_bike', cost_field='travel_time_bike',
                            capacity_field='capacity_bikes',
                            algorithm='bfsle', max_routes=3, max_depth=100, verbose=True)
fig, ax = plt.subplots(figsize=(16, 8))
plot_network(results_sto['network'], node_df,
             node_id_col='node',
             node_label=True,
             color_col_num='flow',
             base_width=1,
             width_col='flow',
             width_scale=4,
             legend=True,
             title=f"Network with flow ({od_scenario} scenario)",
             figsize=(8, 8),
             cmap='Greens',
             vmax=results_sto['network']['flow'].max(),
             vmin=results_sto['network']['flow'].min(),
             ax=ax)
plt.show()

# parameters for mode choice
beta_time = -0.01
ASC_car = 0
ASC_bike = -2.5

impact_coef = 1
mu_mode = 1.0
max_iter_mode_choice = 10
plot = True

edge_df["flow_car"] = 0
edge_df["flow_bike"] = 0

results_df = pd.DataFrame({'iteration': [0],
                           'modal_share_car': [np.nan],
                           'modal_share_bike': [np.nan],
                           'total_travel_time_car': [np.nan],
                           'total_travel_time_bike': [np.nan],
                           'travel_time_per_car': [np.nan],
                           'travel_time_per_bike': [np.nan]})

j = 0

while j < max_iter_mode_choice:
    if plot:
        print(f"\n--- Mode Choice Loop {j + 1} ---")
    j += 1

    skim_car = skimming(edge_df, time_field='travel_time_car', size_od=size_od)
    skim_bike = skimming(edge_df, time_field='travel_time_bike', size_od=size_od)
 #TODO: continuer à segmenter le code
    # Calculate utilities and mode share for each OD pair
    od_mode_shares_skim = np.zeros((size_od, size_od, 2))  # 3rd dimension: 0 for car, 1 for bike
    total_car_skim = 0
    total_bike_skim = 0

    for origin in range(1, size_od - 1):
        for destination in range(1, size_od - 1):
            if origin != destination and od_matrix.loc[origin, destination] > 0:
                # Utilities
                V_car = ASC_car + beta_time * skim_car[origin - 1, destination - 1]
                V_bike = ASC_bike + beta_time * skim_bike[
                    origin - 1, destination - 1]  # TO MODIFY FOR AN OTHER UTILITY FUNCTION

                # Probabilities using logit model
                exp_car = np.exp(mu_mode * V_car)
                exp_bike = np.exp(mu_mode * V_bike)
                P_car = exp_car / (exp_car + exp_bike)
                P_bike = exp_bike / (exp_car + exp_bike)

                od_mode_shares_skim[origin - 1, destination - 1, 0] = P_car
                od_mode_shares_skim[origin - 1, destination - 1, 1] = P_bike

                # Accumulate total mode shares
                total_car_skim += od_matrix.loc[origin, destination] * P_car
                total_bike_skim += od_matrix.loc[origin, destination] * P_bike
    if plot:
        print(
            f"Mode shares with skimming: Car = {total_car_skim / (total_car_skim + total_bike_skim) * 100 :.3f} %, Bike = {total_bike_skim / (total_car_skim + total_bike_skim) * 100:.3f}%")
    # Update OD matrices based on mode shares
    updated_od_car = pd.DataFrame(0, index=range(1, size_od + 1), columns=range(1, size_od + 1))
    updated_od_bike = pd.DataFrame(0, index=range(1, size_od + 1), columns=range(1, size_od + 1))

    for origin in range(1, size_od - 1):
        for destination in range(1, size_od - 1):
            if origin != destination:
                total_od = od_matrix.loc[origin, destination]
                P_car = od_mode_shares_skim[origin - 1, destination - 1, 0]
                P_bike = od_mode_shares_skim[origin - 1, destination - 1, 1]
                updated_od_car.loc[origin, destination] = total_od * P_car
                updated_od_bike.loc[origin, destination] = total_od * P_bike

    updated_od_car = convert_to_eaquilibrae_od_matrix(updated_od_car)
    updated_od_bike = convert_to_eaquilibrae_od_matrix(updated_od_bike)

    #run traffic assignments with updated OD matrices
    car_results_mode_choice = ta_due(
        edge_df,
        updated_od_car,
        algorithm='bfw',
        time_field='travel_time_car',
        cost_field='travel_time_car',
        capacity_field='capacity_cars',
        max_iter=500,
        tolerance=1e-4,
        verbose=True
    )

    edge_df = car_results_mode_choice['network'].copy()
    edge_df["flow_car"] = edge_df["flow"]
    edge_df["ratio_flow_capacity_car"] = edge_df["ratio"]
    edge_df = pd.DataFrame.from_dict(edge_df)
    if plot:
        plot_vc_histogram(edge_df, capacity_col='capacity_cars', bins=20)

    bike_results_mode_choice = ta_stochastic(
        edge_df,
        updated_od_bike,
        mode='bikes',
        time_field='travel_time_bike',
        cost_field='travel_time_bike',  ### LENGTH OR LENGTH_BI?
        algorithm='bfsle',
        max_routes=3,
        capacity_field='capacity_bikes',
        verbose=True
    )

    edge_df = bike_results_mode_choice['network'].copy()
    edge_df["flow_bike"] = edge_df["flow"]
    edge_df = pd.DataFrame.from_dict(edge_df)

    #calculate congested time for cars
    calculate_congested_time(edge_df, flow_name='flow_car', free_flow_time_name='free_flow_time',
                             capacity_name="capacity_cars", congested_time_name='travel_time_car', alpha=0.15, beta=4)
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        plot_network(edge_df, node_df, width_col='flow_car', color_col_num='flow_car', cmap='Reds',
                     title=f'Car flows- Mode Choice Assignment - Loop {j}', node_size=3, colorbar_label='Flow (cars)',
                     base_width=1, width_scale=50, ax=axes[0, 0])
        plot_network(edge_df, node_df, width_col='flow_bike', color_col_num='flow_bike', cmap='Greens',
                     title=f'Bike flows - Mode Choice Assignment - Loop {j}', node_size=3,
                     colorbar_label='Flow (bikes)', base_width=1, width_scale=50, ax=axes[0, 1])
        plot_network(edge_df, node_df, color_col_num='travel_time_car', cmap='hot_r',
                     title=f'Car Travel Time - Loop {j}', node_size=3, colorbar_label='Travel Time (s)', base_width=1,
                     ax=axes[1, 0])
        plot_network(edge_df, node_df, color_col_num='travel_time_bike', cmap='hot_r',
                     title=f'Bike Travel Time - Loop {j}', node_size=3, colorbar_label='Travel Time (s)', base_width=1,
                     ax=axes[1, 1])
        plt.show()

    results_df.loc[j - 1, 'iteration'] = j
    results_df.loc[j - 1, 'modal_share_car'] = 100 * total_car_skim / od_matrix.values.sum()
    results_df.loc[j - 1, 'modal_share_bike'] = 100 * total_bike_skim / od_matrix.values.sum()
    total_travel_time_car = (edge_df['travel_time_car'] * edge_df['flow_car']).sum()
    total_travel_time_bike = (edge_df['travel_time_bike'] * edge_df['flow_bike']).sum()
    results_df.loc[j - 1, 'total_travel_time_car'] = total_travel_time_car
    results_df.loc[j - 1, 'total_travel_time_bike'] = total_travel_time_bike
    results_df.loc[j - 1, 'travel_time_per_car'] = total_travel_time_car / updated_od_car.values.sum()
    results_df.loc[j - 1, 'travel_time_per_bike'] = total_travel_time_bike / updated_od_bike.values.sum()

print(
    f"Mode shares with skimming: Car = {total_car_skim / (total_car_skim + total_bike_skim) * 100 :.3f} %, Bike = {total_bike_skim / (total_car_skim + total_bike_skim) * 100:.3f}%")
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
plot_network(edge_df, node_df, width_col='flow_car', color_col_num='flow_car', cmap='Reds',
             title=f'Car flows- Mode Choice Assignment - Loop {j}', node_size=3, colorbar_label='Flow (cars)',
             base_width=1, width_scale=50, ax=axes[0, 0])
plot_network(edge_df, node_df, width_col='flow_bike', color_col_num='flow_bike', cmap='Greens',
             title=f'Bike flows - Mode Choice Assignment - Loop {j}', node_size=3, colorbar_label='Flow (bikes)',
             base_width=1, width_scale=50, ax=axes[0, 1])
plot_network(edge_df, node_df, color_col_num='travel_time_car', cmap='hot_r', title=f'Car Travel Time - Loop {j}',
             node_size=3, colorbar_label='Travel Time (s)', base_width=1, ax=axes[1, 0])
plot_network(edge_df, node_df, color_col_num='travel_time_bike', cmap='hot_r', title=f'Bike Travel Time - Loop {j}',
             node_size=3, colorbar_label='Travel Time (s)', base_width=1, ax=axes[1, 1])
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
results_df.plot.line(x='iteration', y=['modal_share_car', 'modal_share_bike'], title='Evolution of modal shares',
                     ax=axes[0])
results_df.plot.line(x='iteration', y=['total_travel_time_car', 'total_travel_time_bike'],
                     title='Evolution of travel times', ax=axes[1])
axes[0].set_ylabel('Modal Share (%)')
axes[1].set_ylabel('Total Travel Time (s)')
axes[0].set_xticks(results_df['iteration'])
axes[1].set_xticks(results_df['iteration'])
axes[0].grid(True)
axes[1].grid(True)
plt.show()
