import warnings
import logging
from traffic import *
from network_processing import *
from plotting import *
from od_matrix_generator import generate_od_df
from config import parameter

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
             legend=False,
             title="Network with Free Flow Time",
             figsize=(8, 8))


# parameters for mode choice
parameter_dict = parameter()
beta_time = parameter_dict['beta_time']
ASC_car = parameter_dict['ASC_car']
ASC_bike = parameter_dict['ASC_bike']
mu_mode = parameter_dict['mu_mode']
max_iter_mode_choice = parameter_dict['max_iter_mode_choice']
plot = True

size_od = max(node_df['node']) + 1

od_df = generate_od_df(size_od, od_scenario="RANDOM_OD", max_demand=200)

result_df, updated_od_car, updated_od_bike = mode_choice(edge_df,
                                                         node_df,
                                                         od_df,
                                                         beta_time,
                                                         ASC_car,
                                                         ASC_bike,
                                                         mu_mode=mu_mode,
                                                         max_iter_mode_choice=max_iter_mode_choice,
                                                         plot=plot)
