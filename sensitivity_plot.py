import matplotlib.pyplot as plt
import pandas as pd
from od_matrix_generator import generate_od_df

CURRENT_DIR = ""

df_results_due = pd.read_json(f"{CURRENT_DIR}output/sensitivity_due.json")
df_results_sto = pd.read_json(f"{CURRENT_DIR}output/sensitivity_sto.json")
df_demand = pd.read_json(f"{CURRENT_DIR}output/demand.json")

list_od_scenarios=['1OD','2OD',"RANDOM_OD"]
list_max_demand = [500,1000,2000,3000,4000,5000]

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
plt.show()

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
plt.show()

### Plot for ta_sto
list_algorithm = ['bfsle', 'lp']
list_max_route = [1,2,3,4,5]
fig, axes = plt.subplots(3,2, figsize=(30,20))
j = 0
for algorithm in list_algorithm:
   i = 0
   for scenario in list_od_scenarios:
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
       axes[i, j].plot(x, y_4, marker='o', color='yellow', label='4')
       axes[i, j].plot(x, y_5, marker='o', color='pink', label='5')
       axes[i,j].legend(title="Number of max_routes")
       axes[i,j].set_title(f"{algorithm}_{scenario}")
       axes[i,j].set_xlabel("")
       axes[i,j].set_ylabel("")
       axes[i,j].grid(alpha=0.5)
       i +=1
   j+=1
axes[2,1].set_xlabel("Total Demand")
axes[1,0].set_ylabel("Total Travel Time")
plt.tight_layout()
plt.show()


#TODO: create similar plots for sensitivity of mode choice




