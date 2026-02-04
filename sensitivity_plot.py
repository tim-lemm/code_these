import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = "/Users/tristan.lemoalle/Documents/Thèse/Code/code_these/"

df_results = pd.read_json(f"{CURRENT_DIR}output/sensitivity_due.json")
print(type(df_results))

list_od_scenarios=['1OD','2OD',"RANDOM_OD"]
list_max_demand = [500,1000,2000,3000,4000,5000]
list_algorithm = ['bfw', 'fw', 'msa']


fig, axes = plt.subplots(3,3, figsize=(30,20))
j = 0
for algorithm in list_algorithm:
    i = 0
    for scenario in list_od_scenarios:
        for max_demand in list_max_demand:
            pd.DataFrame.from_dict(df_results[f"{algorithm}_{scenario}_{max_demand}"]['convergence']).plot.line(x='iteration', y='rgap', ax=axes[i,j], label=max_demand)
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

fig, axes = plt.subplots(3,3, figsize=(30,20)) #TODO: plot bar plot for total travel time for every scenario, algorithm and max_demand
j = 0
#for algorithm in list_algorithm:
#    i = 0
#    for scenario in list_od_scenarios:
#        for max_demand in list_max_demand:
#            df_results[f"{algorithm}_{scenario}_{max_demand}"].plot.bar(x='total_travel_time', ax=axes[i,j], label=max_demand)
#        axes[i,j].set_title(f"{algorithm}_{scenario}")
#        axes[i,j].set_xlabel("")
#        axes[i,j].set_ylabel("")
#        axes[i,j].grid(alpha=0.5)
#        i +=1
#    j+=1
#axes[2,1].set_xlabel("Iteration")
#axes[1,0].set_ylabel("rgap")
#plt.tight_layout()
#plt.show()

#TODO: create similar plots for sensitivity of stochastic assignment
#TODO: create similar plots for sensitivity of mode choice