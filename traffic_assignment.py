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

from traffic import *
from src.utils_sta import ta_due, ta_stochastic, plot_vc_histogram
from network_processing import *
from plotting import *
from od_matrix_generator import generate_od_df, convert_od_df_to_matrix, convert_od_matrix_to_df
from eaquilibrea_interface import *

from aequilibrae.paths import RouteChoice
from aequilibrae.matrix import AequilibraeMatrix

#TODO: rename file

CURRENT_DIR = "/Users/tristan.lemoalle/Documents/Thèse/Code/code_these/"

warnings.filterwarnings('ignore')
logging.getLogger("aequilibrae").setLevel(logging.ERROR)

edge_df, node_df = import_network(CURRENT_DIR + "data/edges_small_grid_2.csv", CURRENT_DIR + "data/nodes_small_grid_2.csv")
plot_network(edge_df, node_df,
             node_id_col='node',
             node_label=True,
             color_col_num='travel_time_car',
             base_width=1,
             legend=True,
             title="Network with Free Flow Time",
             figsize=(8, 8))
plt.show()

od_df = generate_od_df(17, od_scenario="1OD", max_demand=3000)
od_df_eaq = convert_to_eaquilibrae_od_matrix(od_df)
plot_od_matrix(od_df_eaq, edge_df, node_df)
plt.show()

## ta_due tests
results_ta_due = ta_due(edge_df,
                        od_df_eaq,
                        algorithm='bfw',
                        max_iter=500,
                        tolerance=1e-4,
                        time_field='free_flow_time_car',
                        cost_field='free_flow_time_car',
                        capacity_field='capacity_cars',
                        verbose=True)

plot_network(results_ta_due['network'], node_df,color_col_num='flow', width_col='flow', base_width=0.1, width_scale=25, cmap="Reds", vmax=3000)

plt.show()
plot_vc_histogram(results_ta_due['network'], capacity_col='capacity_cars', bins=20)

## ta_sto
results_ta_sto = ta_stochastic(edge_df,
                                od_df_eaq,
                                mode='bikes',
                                time_field='free_flow_time_bike',
                                cost_field='free_flow_time_bike',  ### LENGTH OR LENGTH_BI?
                                algorithm='bfsle',
                                max_routes=3,
                                capacity_field='capacity_bikes',
                                verbose=True)

plot_network(results_ta_sto['network'], node_df,color_col_num='flow', width_col='flow', base_width=0.1, width_scale=25, cmap="Greens", vmax=3000)
plt.show()
