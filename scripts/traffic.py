import networkx as nx
import numpy as np
import pandas as pd

def _create_empty_skim_matrice (size_od:int):
    return np.zeros((size_od, size_od))

def _create_graph_for_skimming (edge_df: pd.DataFrame, time_field)-> nx.Graph:
    graph = nx.DiGraph()
    for _, edge in edge_df.iterrows():
        if edge['a_node'] != edge['b_node']:
            graph.add_edge(edge['a_node'], edge['b_node'], time=edge[time_field])
    return graph

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

def calculate_proba_matrice (skim_matrice_car, skim_matric_bike, ASC_car:float, ASC_bike:float, beta_time:float, mu_mode:float):
    proba_matrice = np.zeros((size_od, size_od, 2))  # 3rd dimension: 0 for car, 1 for bike
    # TODO: continuer implenetation du calcul de la matrice des proba