import matplotlib.pyplot as plt
import pandas as pd
from od_matrix_generator import generate_od_df



CURRENT_DIR = ""

df_results_due = pd.read_json(f"{CURRENT_DIR}output/sensitivity_due.json")
df_results_sto = pd.read_json(f"{CURRENT_DIR}output/sensitivity_sto.json")
df_results_mc = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc.json")
df_results_mc_beta = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_beta.json")
df_results_mc_order = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_order.json")
df_results_mc_time_cost = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_time_cost.json")
df_results_mc_skim_cost = pd.read_json(f"{CURRENT_DIR}output/sensitivity_mc_skim_cost.json")
df_demand = pd.read_json(f"{CURRENT_DIR}output/demand.json")

list_od_scenarios=['1OD','2OD',"RANDOM_OD"]
list_max_demand = [500,1000,2000,3000,4000,5000]
list_asc_bike = [-2.5,-1,0]
list_beta_time = [-0.0001,-0.001,-0.01,-0.1]

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

fig, axes = plt.subplots(3,3, figsize=(30,30))
x = [0,1,2,3,4]
j=0
for ASC_bike in list_asc_bike:
    i = 0
    for scenario in list_od_scenarios:
        y_500 = df_results_mc[f"{scenario}_500_{ASC_bike}"]["results_df"]["modal_share_bike"]
        y_500 = list(y_500.values())
        y_1000 = df_results_mc[f"{scenario}_1000_{ASC_bike}"]["results_df"]["modal_share_bike"]
        y_1000 = list(y_1000.values())
        y_2000 = df_results_mc[f"{scenario}_2000_{ASC_bike}"]["results_df"]["modal_share_bike"]
        y_2000 = list(y_2000.values())
        y_3000 = df_results_mc[f"{scenario}_3000_{ASC_bike}"]["results_df"]["modal_share_bike"]
        y_3000 = list(y_3000.values())
        y_4000 = df_results_mc[f"{scenario}_4000_{ASC_bike}"]["results_df"]["modal_share_bike"]
        y_4000 = list(y_4000.values())
        y_5000 = df_results_mc[f"{scenario}_5000_{ASC_bike}"]["results_df"]["modal_share_bike"]
        y_5000 = list(y_5000.values())
        axes[i,j].plot(x, y_500, marker='v', color='blue', label='500')
        axes[i,j].plot(x, y_1000, marker='o', color='red', label='1000')
        axes[i,j].plot(x, y_2000, marker='^', color='green', label='2000')
        axes[i,j].plot(x, y_3000, marker='>', color='yellow', label='3000')
        axes[i,j].plot(x, y_4000, marker='<', color='pink', label='4000')
        axes[i,j].plot(x, y_5000, marker='h', color='brown', label='5000')
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
max_demand = 5000
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