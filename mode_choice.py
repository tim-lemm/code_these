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

#TODO: impement a parameter() function or config file
#TODO: update readme

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


# parameters for mode choice
beta_time = -0.01
ASC_car = 0
ASC_bike = -2.5
mu_mode = 1.0
max_iter_mode_choice = 1
plot = True
size_od = max(node_df['node']) + 1

od_df = generate_od_df(size_od, od_scenario="RANDOM_OD", max_demand=1000)

result_df, updated_od_car, updated_od_bike = mode_choice(edge_df, node_df, od_df, beta_time=beta_time, mu_mode=mu_mode, max_iter_mode_choice=max_iter_mode_choice, plot=plot)