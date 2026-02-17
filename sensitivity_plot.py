import matplotlib.pyplot as plt
import pandas as pd
from od_matrix_generator import generate_od_df
from plotting import *
from network_processing import import_network
from eaquilibrea_interface import *



CURRENT_DIR = ""

df_results_due = pd.read_json(f"{CURRENT_DIR}output/sensitivity_due.json")
df_results_sto = pd.read_json(f"{CURRENT_DIR}output/sensitivity_sto.json")
df_results_mc = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc.json")
df_results_mc_beta = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_beta.json")
df_results_mc_order = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_order.json")
df_results_mc_time_cost = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_time_cost.json")
df_results_mc_skim_cost = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_skim_cost.json")
df_results_mc_order_2 = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_order_2.json")
df_results_mc_weight_bi = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_weight_bi.json")
df_demand = pd.read_json(f"{CURRENT_DIR}output/demand.json")
edge_df, node_df = import_network(CURRENT_DIR + "data/edges_small_grid_2.csv", CURRENT_DIR + "data/nodes_small_grid_2.csv", capacity_car=1000)


list_od_scenarios=['1OD','2OD',"RANDOM_OD"]
list_max_demand = [100,200,300,400,500,750,1000,1250,1500,1750,2000,2250]
list_asc_bike = [-2.5,-1,0]
list_beta_time = [-0.0001,-0.001,-0.01,-0.1]

### plot of od scenarios
max_demand = 300

fig, axes = plt.subplots(1,3, figsize=(30,10))
j = 0
for od_scenario in list_od_scenarios:
    od_df = generate_od_df(17, od_scenario=od_scenario, max_demand=max_demand)
    od_df = convert_to_eaquilibrae_od_matrix(od_df)
    plot_od_matrix(od_df,edge_df, node_df, ax=axes[j], title=od_scenario)
    j+=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/od_scenarios.png")

### Plots for ta_due
## Plotting convergence

list_algorithm = ['bfw', 'fw', 'msa']
fig, axes = plt.subplots(3,3, figsize=(30,20))
j = 0
for algorithm in list_algorithm:
    i = 0
    for scenario in list_od_scenarios:
        for max_demand in list_max_demand:
            pd.DataFrame.from_dict(df_results_due[f"{algorithm}_{scenario}_{max_demand}"]['convergence']).plot.line(x='iteration', y='rgap', ax=axes[i,j], label=max_demand)
        axes[i,j].set_title(f"{algorithm}_{scenario}")
        axes[i,j].set_xlabel("")
        axes[i,j].set_ylabel("")
        axes[i,j].grid(alpha=0.5)
        i +=1
    j+=1
axes[2,1].set_xlabel("Iteration")
axes[1,0].set_ylabel("rgap")
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_ta_due_algo.png")

## Plotting total travel time vs total demand

fig, axes = plt.subplots(3,3, figsize=(30,20))
j = 0
for algorithm in list_algorithm:
   i = 0
   for scenario in list_od_scenarios:
       x = []
       y = []
       for max_demand in list_max_demand:
           y.append(df_results_due[f"{algorithm}_{scenario}_{max_demand}"]["total_travel_time"])
           x.append(df_demand[(df_demand["scenario"] == scenario) & (df_demand["max_demand"] == max_demand)]["total_demand"].values[0])
       axes[i,j].plot(x, y, marker='o')
       axes[i,j].set_title(f"{algorithm}_{scenario}")
       axes[i,j].set_xlabel("")
       axes[i,j].set_ylabel("")
       axes[i,j].grid(alpha=0.5)
       i +=1
   j+=1
axes[2,1].set_xlabel("Total Demand")
axes[1,0].set_ylabel("Total Travel Time")
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_ta_due_algo_tt_vs_tdemand.png")

list_max_demand = [100,200,300,400,500,1000,2000]
### Plot for ta_sto
list_algorithm = ["bfsle","lp"]
fig, axes = plt.subplots(3,2, figsize=(30,20))
i = 0
for scenario in list_od_scenarios:
   j = 0
   for algorithm in list_algorithm:
       x = []
       y_1 = []
       y_2 = []
       y_3 = []
       y_4 = []
       y_5 = []
       for max_demand in list_max_demand:
           x.append(df_demand[(df_demand["scenario"] == scenario) & (df_demand["max_demand"] == max_demand)]["total_demand"].values[0])
           y_1.append(df_results_sto[f"{algorithm}_1_{scenario}_{max_demand}"]["total_travel_time"])
           y_2.append(df_results_sto[f"{algorithm}_2_{scenario}_{max_demand}"]["total_travel_time"])
           y_3.append(df_results_sto[f"{algorithm}_3_{scenario}_{max_demand}"]["total_travel_time"])
           y_4.append(df_results_sto[f"{algorithm}_4_{scenario}_{max_demand}"]["total_travel_time"])
           y_5.append(df_results_sto[f"{algorithm}_5_{scenario}_{max_demand}"]["total_travel_time"])
       axes[i,j].plot(x, y_1, marker='o', color='blue', label='1')
       axes[i,j].plot(x, y_2, marker='o', color='red', label='2')
       axes[i,j].plot(x, y_3, marker='o', color='green', label='3')
       axes[i,j].plot(x, y_4, marker='o', color='yellow', label='4')
       axes[i,j].plot(x, y_5, marker='o', color='pink', label='5')
       axes[i,j].legend(title="Number of max_routes")
       axes[i,j].set_title(f"{algorithm}_{scenario}")
       axes[i,j].set_xlabel("")
       axes[i,j].set_ylabel("")
       axes[i,j].grid(alpha=0.5)
       j +=1
   i+=1
axes[2,1].set_xlabel("Total Demand")
axes[1,0].set_ylabel("Total Travel Time")
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_ta_sto.png")

### plot sensitivity for mode choice
## plot mode share for iteration - test of od and ASC
list_max_demand = [100,200,300,400,500,750,1000,1250,1500,1750,2000,2250]
fig, axes = plt.subplots(3,3, figsize=(30,30))
x = [0,1,2,3,4]
j=0
for ASC_bike in list_asc_bike:
    i = 0
    for scenario in list_od_scenarios:
        for max_demand in list_max_demand:
            y = df_results_mc[f"{scenario}_{max_demand}_{ASC_bike}"]["results_df"]["modal_share_bike"]
            y = list(y.values())
            axes[i,j].plot(x, y, marker='o', label=max_demand)
            axes[i,j].legend(title="max demand")
            axes[i,j].set_title(f"{scenario}_{ASC_bike}")
            axes[i,j].set_xlabel("Iteration")
            axes[i,j].set_ylabel("Modal Share of bike (%)")
            axes[i,j].grid(alpha=0.5)
        i +=1
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_od_asc.png")

## plot mode share for iteration - test for beta and ASC
max_demand = 500
fig, axes = plt.subplots(3,3, figsize=(30,30))
x = [0,1,2,3,4]
j = 0
for ASC_bike in list_asc_bike:
    i = 0
    for scenario in list_od_scenarios:
        for beta in list_beta_time :
            y = df_results_mc_beta[f"{scenario}_{max_demand}_{ASC_bike}_{beta}"]["results_df"]["modal_share_bike"]
            y = list(y.values())
            axes[i,j].plot(x, y, marker='o', label=beta)
        axes[i,j].legend(title="beta")
        axes[i,j].set_title(f"{scenario}_{ASC_bike}")
        axes[i,j].set_xlabel("Iteration")
        axes[i,j].set_ylabel("Modal Share of bike (%)")
        axes[i,j].grid(alpha=0.5)
        i +=1
    j += 1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_beta_asc.png")

## plot for different version of mode_choice function

fig, axes = plt.subplots(2,3, figsize=(30,20))

ASC_bike = -2.5
x = [0,1,2,3,4]
list_version = ["mc1", "mc2", "mc4"]
list_bis = ["","bis"]
j=0
for version in list_version:
    i = 0
    for bis in list_bis:
        for max_demand in list_max_demand:
            y = df_results_mc_order[f'{scenario}_{max_demand}_{ASC_bike}_{version}{bis}']["results_df"]["modal_share_bike"]
            y = list(y.values())
            axes[i, j].plot(x, y, marker='o', label=max_demand)
        axes[i,j].legend(title="max demand")
        axes[i,j].set_title(f"{version}{bis}")
        axes[i,j].set_xlabel("Iteration")
        axes[i,j].set_ylabel("Modal Share of bike (%)")
        axes[i,j].grid(alpha=0.5)
        i +=1
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_od_version.png")

## plot for different time and cost field for ta_due

fig, axes = plt.subplots(2,3, figsize=(30,20))
ASC_bike = -2.5
beta_time = -0.01
scenario = "RANDOM_OD"
list_cost_field = ["free_flow_time_car", "travel_time_car","length"]
list_time_field = ["free_flow_time_car", "travel_time_car"]


j=0
for cost_field in list_cost_field:
    i = 0
    for time_field in list_time_field:
        for max_demand in list_max_demand:
            y = df_results_mc_time_cost[ f'{scenario}_{max_demand}_{ASC_bike}_{time_field}_{cost_field}']["results_df"]["modal_share_bike"]
            y = list(y.values())
            axes[i, j].plot(x, y,marker='o', label=max_demand)
        axes[i,j].legend(title="max demand")
        axes[i,j].set_title(f"{time_field}_{cost_field}")
        axes[i,j].set_xlabel("Iteration")
        axes[i,j].set_ylabel("Modal Share of bike (%)")
        axes[i,j].grid(alpha=0.5)
        i +=1
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_cost_due.png")

# plot for different skim cost
fig, axes = plt.subplots(3,4, figsize=(40,30))
ASC_bike = -2.5
beta_time = -0.01
scenario = "RANDOM_OD"
list_cost_bike = ["travel_time_bike","free_flow_time_bike","length","length_bi"]
list_cost_car = ["travel_time_car","free_flow_time_car","length"]


j=0
for cost_bike in list_cost_bike:
    i = 0
    for cost_car in list_cost_car:
        for max_demand in list_max_demand:
            y = df_results_mc_skim_cost[ f'{scenario}_{max_demand}_{ASC_bike}_{cost_bike}_{cost_car}']["results_df"]["modal_share_bike"]
            y = list(y.values())
            axes[i, j].plot(x, y,marker='o', label=max_demand)
        axes[i,j].legend(title="max demand")
        axes[i,j].set_title(f"Skim cost bike : {cost_bike} / Skim cost car : {cost_car}")
        axes[i,j].set_xlabel("Iteration")
        axes[i,j].set_ylabel("Modal Share of bike (%)")
        axes[i,j].grid(alpha=0.5)
        i +=1
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_cost_skim.png")

fig, axes = plt.subplots(2,4, figsize=(30,20))

ASC_bike = -2.5
x = [0,1,2,3,4,5,6,7,8,9]
list_version = ["mc1", "mc2", "mc3","mc4"]
demand = 500

j=0
for version in list_version:
    y = df_results_mc_order_2[f'{scenario}_{demand}_{ASC_bike}_{version}']["results_df"]["modal_share_bike"]
    y = list(y.values())
    axes[0, j].plot(x, y, marker='o', label = demand)
    axes[0,j].legend(title="max demand")
    axes[0,j].set_title(f"{version}")
    axes[0,j].set_xlabel("Iteration")
    axes[0,j].set_ylabel("Modal Share of bike (%)")
    axes[0,j].grid(alpha=0.5)

    y_bike_skim = df_results_mc_order_2[f'{scenario}_{demand}_{ASC_bike}_{version}']["results_df"]["total_bike_skim"]
    y_bike_skim = list(y_bike_skim.values())
    y_car_skim = df_results_mc_order_2[f'{scenario}_{demand}_{ASC_bike}_{version}']["results_df"]["total_car_skim"]
    y_car_skim = list(y_car_skim.values())
    axes[1, j].bar(x, y_bike_skim, width=0.5, color='b', label='bike skim')
    axes[1, j].bar(x, y_car_skim, width=0.5, color='r', bottom=y_bike_skim, label='car skim')

    axes[1, j].set_title(f"{version}")
    axes[1, j].set_xlabel("Iteration")
    axes[1, j].set_ylabel("Total skim")
    axes[1, j].grid(alpha=0.5)
    axes[1, j].legend(title="Total skim")
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_od_version_with_bar_plot.png")

scenario = "RANDOM_OD"
fig, axes = plt.subplots(1,12, figsize=(60,10))
ASC_bike = -2.5
x = [0,1,2,3,4,5,6,7,8,9]
list_weight = [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0, -0.05, -0.1, -0.15]

j=0
for max_demand in list_max_demand:
    for weight in list_weight:
        y = df_results_mc_weight_bi[f'{scenario}_{max_demand}_{ASC_bike}_{weight}']["results_df"]["modal_share_bike"]
        y = list(y.values())
        axes[j].plot(x, y, marker='o', label = weight)
        axes[j].legend(title="weight bi")
        axes[j].set_title(f"max_demand {max_demand}")
        axes[j].set_xlabel("Iteration")
        axes[j].set_ylabel("Modal Share of bike (%)")
        axes[j].grid(alpha=0.5)
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_RANDOM_OD_weight_bi.png")

fig, axes = plt.subplots(1,12, figsize=(60,10))
ASC_bike = -2.5
x = [0,1,2,3,4,5,6,7,8,9]
list_weight = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4]

j=0
for max_demand in list_max_demand:
    for weight in list_weight:
        y = df_results_mc_weight_bi[f'{scenario}_{max_demand}_{ASC_bike}_{weight}']["results_df"]["modal_share_bike"]
        y = list(y.values())
        axes[j].plot(x, y, marker='o', label = weight)
        axes[j].legend(title="weight bi")
        axes[j].set_title(f"max_demand {max_demand}")
        axes[j].set_xlabel("Iteration")
        axes[j].set_ylabel("Modal Share of bike (%)")
        axes[j].grid(alpha=0.5)
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_RANDOM_OD_weight_bi_2.png")

scenario = "1OD"
fig, axes = plt.subplots(1,12, figsize=(60,10))
ASC_bike = -2.5
x = [0,1,2,3,4,5,6,7,8,9]
list_weight = [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0, -0.05, -0.1, -0.15]

j=0
for max_demand in list_max_demand:
    for weight in list_weight:
        y = df_results_mc_weight_bi[f'{scenario}_{max_demand}_{ASC_bike}_{weight}']["results_df"]["modal_share_bike"]
        y = list(y.values())
        axes[j].plot(x, y, marker='o', label = weight)
        axes[j].legend(title="weight bi")
        axes[j].set_title(f"max_demand {max_demand}")
        axes[j].set_xlabel("Iteration")
        axes[j].set_ylabel("Modal Share of bike (%)")
        axes[j].grid(alpha=0.5)
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_1OD_weight_bi.png")

fig, axes = plt.subplots(1,12, figsize=(60,10))
ASC_bike = -2.5
x = [0,1,2,3,4,5,6,7,8,9]
list_weight = [0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4]

j=0
for max_demand in list_max_demand:
    for weight in list_weight:
        y = df_results_mc_weight_bi[f'{scenario}_{max_demand}_{ASC_bike}_{weight}']["results_df"]["modal_share_bike"]
        y = list(y.values())
        axes[j].plot(x, y, marker='o', label = weight)
        axes[j].legend(title="weight bi")
        axes[j].set_title(f"max_demand {max_demand}")
        axes[j].set_xlabel("Iteration")
        axes[j].set_ylabel("Modal Share of bike (%)")
        axes[j].grid(alpha=0.5)
    j +=1
plt.tight_layout()
plt.savefig(f"{CURRENT_DIR}output/img/sensitivity_mc_1OD_weight_bi_2.png")