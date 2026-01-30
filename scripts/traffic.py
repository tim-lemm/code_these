import networkx as nx
import numpy as np
import pandas as pd
from scripts.plotting import *
from scripts.od_matrix_generator import *
from scripts.eaquilibrea_interface import *
from scripts.network_processing import *
from src.utils_sta import ta_due, ta_stochastic

def _create_empty_skim_matrice (size_od:int):
    return np.zeros((size_od, size_od))

def _create_graph_for_skimming (edge_df: pd.DataFrame, time_field)-> nx.Graph:
    graph = nx.DiGraph()
    for _, edge in edge_df.iterrows():
        if edge['a_node'] != edge['b_node']:
            graph.add_edge(edge['a_node'], edge['b_node'], time=edge[time_field])
    return graph

def create_empty_result_df_mc()-> pd.DataFrame:
    return pd.DataFrame({'iteration': [0],
                        'modal_share_car': [np.nan],
                        'modal_share_bike': [np.nan],
                        'total_travel_time_car': [np.nan],
                        'total_travel_time_bike': [np.nan],
                        'travel_time_per_car': [np.nan],
                        'travel_time_per_bike': [np.nan]})

def update_result_df_mc (results_df:pd.DataFrame,
                         j,
                         modal_share_car,
                         modal_share_bike,
                         total_travel_time_car,
                         total_travel_time_bike,
                         total_car_skim,
                         total_bike_skim) -> pd.DataFrame:
    results_df.loc[j - 1, 'iteration'] = j
    results_df.loc[j - 1, 'modal_share_car'] = modal_share_car
    results_df.loc[j - 1, 'modal_share_bike'] = modal_share_bike
    results_df.loc[j - 1, 'total_travel_time_car'] = total_travel_time_car
    results_df.loc[j - 1, 'total_travel_time_bike'] = total_travel_time_bike
    results_df.loc[j - 1, 'travel_time_per_car'] = total_travel_time_car / total_car_skim
    results_df.loc[j - 1, 'travel_time_per_bike'] = total_travel_time_bike / total_bike_skim
    return results_df

def utility_car(dist, ASC, beta_time):
    return ASC + beta_time * dist

def utility_bike(dist, ASC, beta_time):
    return ASC + beta_time * dist


def skimming (edge_df: pd.DataFrame, size_od, time_field:str = 'time'):
    skim_matrice = _create_empty_skim_matrice(size_od)
    graph = _create_graph_for_skimming(edge_df, time_field)
    for nodes_o in graph.nodes(data=True):
        for nodes_d in graph.nodes(data=True):
            if nodes_o[0] != nodes_d[0]:
                try:
                    skim_matrice[nodes_o[0], nodes_d[0]] = nx.shortest_path_length(graph, source=nodes_o[0],
                                                                                    target=nodes_d[0], weight='time')
                except:
                    skim_matrice[nodes_o[0], nodes_d[0]] = 9999
    return skim_matrice

def calculate_proba_matrice (skim_matrice_car, skim_matrice_bike, ASC_car:float, ASC_bike:float, beta_time:float, mu_mode:float, size_od:int):
    prob_matrice_car = _create_empty_skim_matrice(size_od)
    prob_matrice_bike = _create_empty_skim_matrice(size_od)
    for origin in range(size_od):
        for destination in range(size_od):
                # Utilities
                V_car = utility_car(skim_matrice_car[origin, destination], ASC_car, beta_time)
                V_bike = utility_bike(skim_matrice_bike[origin, destination], ASC_bike, beta_time)
                
                # Probabilities using logit model
                exp_car = np.exp(mu_mode * V_car)
                exp_bike = np.exp(mu_mode * V_bike)
                P_car = exp_car / (exp_car + exp_bike)
                P_bike = exp_bike / (exp_car + exp_bike)

                prob_matrice_car[origin, destination] = P_car
                prob_matrice_bike[origin, destination] = P_bike
    return prob_matrice_car, prob_matrice_bike

def plot_mc_results(edge_df, node_df, results_df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plot_network(edge_df, node_df, width_col='flow_car', color_col_num='flow_car', cmap='Reds',
                 title=f'Car flows- Mode Choice Assignment ', node_size=3, colorbar_label='Flow (cars)',
                 base_width=1, width_scale=50, ax=axes[0, 0])
    plot_network(edge_df, node_df, width_col='flow_bike', color_col_num='flow_bike', cmap='Greens',
                 title=f'Bike flows - Mode Choice Assignment ', node_size=3, colorbar_label='Flow (bikes)',
                 base_width=1, width_scale=50, ax=axes[0, 1])
    plot_network(edge_df, node_df, color_col_num='travel_time_car', cmap='hot_r', title=f'Car Travel Time',
                 node_size=3, colorbar_label='Travel Time (s)', base_width=1, ax=axes[1, 0])
    plot_network(edge_df, node_df, color_col_num='travel_time_bike', cmap='hot_r', title=f'Bike Travel Time',
                 node_size=3, colorbar_label='Travel Time (s)', base_width=1, ax=axes[1, 1])
    plt.show()

    _,axes = plt.subplots(1, 2, figsize=(20, 7))
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

def mode_choice (edge_df, node_df, od_df,
                 beta_time = -0.01,
                 ASC_car = 0,
                 ASC_bike = -2.5,
                mu_mode = 1.0,
                max_iter_mode_choice = 3,
                plot = True) :

    od_matrix = convert_od_df_to_matrix(od_df)
    size_od = len(od_matrix)
    results_df = create_empty_result_df_mc()
    j = 0
    while j < max_iter_mode_choice:
        if plot:
            print(f"\n--- Mode Choice Loop {j + 1} ---")
        j += 1

        skim_car = skimming(edge_df, time_field='travel_time_car', size_od=size_od)
        skim_bike = skimming(edge_df, time_field='travel_time_bike', size_od=size_od)
        # Calculate utilities and mode share for each OD pair
        prob_matrice_car, prob_matrice_bike = calculate_proba_matrice(skim_car, skim_bike, ASC_car, ASC_bike, beta_time,
                                                                      mu_mode, size_od)
        od_matrix_car = od_matrix * prob_matrice_car
        od_matrix_bike = od_matrix * prob_matrice_bike

        total_car_skim = od_matrix_car.sum()
        total_bike_skim = od_matrix_bike.sum()

        total_skim = total_car_skim + total_bike_skim

        modal_share_car = (total_car_skim / total_skim) * 100
        modal_share_bike = (total_bike_skim / total_skim) * 100

        od_df_car = convert_od_matrix_to_df(od_matrix_car)
        od_df_bike = convert_od_matrix_to_df(od_matrix_bike)
        updated_od_car = convert_to_eaquilibrae_od_matrix(od_df_car)
        updated_od_bike = convert_to_eaquilibrae_od_matrix(od_df_bike)

        # run traffic assignments with updated OD matrices
        car_results_mode_choice = ta_due(
            edge_df,
            updated_od_car,
            algorithm='bfw',
            time_field='free_flow_time',
            cost_field='travel_time_car',
            capacity_field='capacity_cars',
            max_iter=500,
            tolerance=1e-4,
            verbose=plot
        )
        total_travel_time_car = car_results_mode_choice['total_travel_time']
        edge_df = car_results_mode_choice['network'].copy()
        edge_df["flow_car"] = edge_df["flow"]
        edge_df["ratio_flow_capacity_car"] = edge_df["ratio"]
        edge_df = pd.DataFrame.from_dict(edge_df)

        bike_results_mode_choice = ta_stochastic(
            edge_df,
            updated_od_bike,
            mode='bikes',
            time_field='travel_time_bike',
            cost_field='length_bi',  ### LENGTH OR LENGTH_BI?
            algorithm='bfsle',
            max_routes=3,
            capacity_field='capacity_bikes',
            verbose=plot
        )
        total_travel_time_bike = bike_results_mode_choice['total_travel_time']
        edge_df = bike_results_mode_choice['network'].copy()
        edge_df["flow_bike"] = edge_df["flow"]
        edge_df = pd.DataFrame.from_dict(edge_df)

        # calculate congested time for cars
        update_network(edge_df, flow_name='flow_car', free_flow_time_name='free_flow_time',
                       capacity_name="capacity_cars", congested_time_name='travel_time_car', alpha=0.15, beta=4)

        results_df = update_result_df_mc(results_df, j, modal_share_car, modal_share_bike, total_travel_time_car,
                                         total_travel_time_bike, total_car_skim, total_bike_skim)

    print(
        f"Mode shares with skimming: Car = {total_car_skim / (total_car_skim + total_bike_skim) * 100 :.3f} %, Bike = {total_bike_skim / (total_car_skim + total_bike_skim) * 100:.3f}%")
    if plot:
        plot_mc_results(edge_df, node_df, results_df)

    return results_df, updated_od_car, updated_od_bike