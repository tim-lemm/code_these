import numpy as np

def calculate_length(node_df, edge_df):
    """ Calculate Euclidean length of edges based on node coordinates. """
    lengths = []
    for edge in edge_df.itertuples():
        a_node = node_df.loc[node_df['node'] == edge.a_node]
        b_node = node_df.loc[node_df['node'] == edge.b_node]
        length = np.sqrt((a_node['x'].values[0] - b_node['x'].values[0])**2 + (a_node['y'].values[0] - b_node['y'].values[0])**2)
        lengths.append(length)
    return edge_df.assign(length=lengths)

def calculate_congested_time(edges_df, free_flow_time_name="free_flow_time", congested_time_name="congested_time", flow_name="flow", capacity_name="capacity", alpha=0.15, beta=4):
    """Calculate congested travel time using BPR function.
    𝑇=𝑇0(1+α(𝑉/𝐶)^β)
    where:
    - 𝑇 is the congested travel time
    - 𝑇0 is the free-flow travel time
    - 𝑉 is the volume (flow)
    - 𝐶 is the capacity
    - α and β are parameters
    """
    edges_df[congested_time_name]=edges_df[free_flow_time_name]*(1+alpha*(edges_df[flow_name]/edges_df[capacity_name])**beta)
    return edges_df