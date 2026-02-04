"""
Unified Transport Assignment Utilities (utils_sta.py)
======================================================
Consolidated module for static traffic assignment and visualization.

All assignment functions follow consistent naming (ta_*) and interface:
- Input: edges_gdf (GeoDataFrame), od_gdf (GeoDataFrame with origin/destination/demand)
- Output: dict with 'network', 'flow', 'time', 'total_travel_time', etc.

Mode choice functions (mc_*) for logsum calculation and demand splitting.
"""

import time
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString, Polygon

from aequilibrae.paths import RouteChoice, Graph, TrafficAssignment, TrafficClass
from aequilibrae.matrix import AequilibraeMatrix
import logging
# logging.getLogger("aequilibrae").setLevel(logging.ERROR)
# logging.getLogger("aequilibrae").setLevel(logging.CRITICAL + 1)
# or, to disable all logging globally:
logging.disable(logging.CRITICAL)

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_demand_matrix(od_gdf, name='matrix'):
    """Convert OD GeoDataFrame to AequilibraeMatrix."""
    zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))
    od_matrix = np.zeros((zones, zones))

    for _, row in od_gdf.iterrows():
        od_matrix[int(row['origin']) - 1, int(row['destination']) - 1] = row['demand']

    demand = AequilibraeMatrix()
    demand.create_empty(zones=zones, matrix_names=[name], memory_only=True)
    demand.matrix[name][:, :] = od_matrix
    demand.index[:] = np.arange(1, zones + 1).astype(int)
    demand.computational_view([name])

    return demand

def _build_graph_for_stochastic(network_df, zones, time_field, cost_field, capacity_field):
    """Build graph for stochastic assignment."""
    network_df = network_df.copy()
    network_df['direction'] = 1
    network_df.index = list(range(len(network_df)))
    network_df["link_id"] = network_df.reset_index().index + 1
    network_df = network_df.astype({"a_node": "int64", "b_node": "int64", "direction": "int64", "link_id": "int64"})

    g = Graph()
    g.cost = network_df[cost_field].values
    g.capacity = network_df[capacity_field].values
    g.free_flow = network_df[time_field].values

    g.network = network_df
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(np.arange(1, zones + 1).astype(int))
    g.set_graph(cost_field)
    g.cost = np.array(g.cost, copy=True)
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id

    return g

# =============================================================================
# TRAFFIC ASSIGNMENT FUNCTIONS (DUE, STOCHASTIC (uncongested), OPTIMAL STRATEGIES)
# =============================================================================

def ta_due(edges_gdf, od_gdf, algorithm='bfw', max_iter=500, tolerance=1e-4,
           time_field='free_flow_time', cost_field='free_flow_time', capacity_field='capacity', verbose=False):
    """
    Deterministic User Equilibrium (DUE) traffic assignment.
    """
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df['alpha'] = network_df['alpha'].fillna(0.15)
    network_df['beta'] = network_df['beta'].fillna(4.0)
    network_df[capacity_field] = network_df[capacity_field].fillna(99999)
    network_df[time_field] = network_df[time_field].fillna(100.00)

    zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))
    demand = _build_demand_matrix(od_gdf)

    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    network_df['direction'] = 1
    network_df.index = list(range(len(network_df)))
    network_df["link_id"] = network_df.reset_index().index + 1
    network_df = network_df.astype({"a_node": "int64", "b_node": "int64", "direction": "int64", "link_id": "int64"})

    g = Graph()
    g.cost = network_df[cost_field].values
    g.capacity = network_df[capacity_field].values
    g.free_flow = network_df[time_field].values
    g.network = network_df
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(np.arange(1, zones + 1).astype(int))
    g.set_graph(cost_field)
    g.cost = np.array(g.cost, copy=True)
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id

    traffic_class = TrafficClass('car', g, demand)
    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf('BPR')
    assignment.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})
    assignment.set_capacity_field(capacity_field)
    assignment.set_time_field(cost_field)
    assignment.set_algorithm(algorithm)
    assignment.max_iter = max_iter
    assignment.rgap_target = tolerance

    results_df = network_df.copy()
    start_time = time.time()

    try:
        assignment.execute()
        computation_time = time.time() - start_time

        results = assignment.results()
        flows = results['matrix_ab'].values
        times = results['Congested_Time_AB'].values
        total_travel_time = np.sum(flows * times)
        rgap_list = list(assignment.report()['rgap'])
        rgap = rgap_list[-1] if rgap_list else np.inf

        results_df['flow'] = flows
        results_df['time'] = times
        results_df['ratio'] = flows / results_df[capacity_field].replace(0, np.inf)
        convergence = pd.DataFrame(assignment.assignment.convergence_report)

        if verbose:
            print(f"DUE completed in {computation_time:.2f}s | RGAP: {rgap:.6f} | TTT: {total_travel_time:.0f}")

        return {
            'network': results_df,
            'total_travel_time': total_travel_time,
            'computation_time': computation_time,
            'rgap': rgap,
            'flow': flows,
            'time': times,
            'convergence': convergence
        }

    except Exception as e:
        if verbose:
            print(f"Error in DUE assignment: {e}")
            import traceback
            traceback.print_exc()
        return _empty_result(results_df, len(network_df))


def ta_stochastic(edges_gdf, od_gdf, mode='car',
                  time_field='free_flow_time', cost_field='free_flow_time', capacity_field='capacity',
                  algorithm='bfsle', max_routes=5, max_depth=100, max_misses=100,
                  beta=1.0, cutoff_prob=0.0, penalty=1.0, seed=0, cores=1, verbose=False):
    """
    Stochastic (uncongested) traffic assignment using Route Choice.
    """
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df[capacity_field] = network_df[capacity_field].fillna(99999)
    network_df[time_field] = network_df[time_field].fillna(100.01)
    if cost_field != time_field:
        network_df[cost_field] = network_df[cost_field].fillna(100.01)

    zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))
    demand = _build_demand_matrix(od_gdf)

    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    graph = _build_graph_for_stochastic(network_df, zones, time_field, cost_field, capacity_field)

    route_choice = RouteChoice(graph)
    route_choice.set_cores(cores)
    route_choice.set_choice_set_generation(
        algorithm=algorithm,
        max_routes=max_routes,
        max_depth=max_depth,
        max_misses=max_misses,
        beta=beta,
        cutoff_prob=cutoff_prob,
        penalty=penalty,
        seed=seed
    )
    route_choice.add_demand(demand)
    route_choice.prepare(nodes=None)

    results_df = network_df.copy()
    start_time = time.time()

    try:
        route_choice.execute(perform_assignment=True)
        computation_time = time.time() - start_time

        load_results = route_choice.get_load_results()
        flows = load_results['matrix_ab'].values
        times = results_df[time_field].values
        costs = results_df[cost_field].values

        total_travel_time = np.sum(flows * times)
        total_travel_cost = np.sum(flows * costs)

        results_df['flow'] = flows
        results_df['time'] = times
        results_df['cost'] = costs

        if verbose:
            print(f"Stochastic ({mode}) completed in {computation_time:.2f}s | TTT: {total_travel_time:.0f}")

        return {
            'network': results_df,
            'total_travel_time': total_travel_time,
            'total_travel_cost': total_travel_cost,
            'computation_time': computation_time,
            'flow': flows,
            'time': times,
            'cost': costs
        }

    except Exception as e:
        if verbose:
            print(f"Error in stochastic assignment: {e}")
            import traceback
            traceback.print_exc()
        return _empty_result(results_df, len(network_df))


def ta_stochastic_bike(edges_gdf, od_gdf, **kwargs):
    """Stochastic assignment for bike network."""
    kwargs.setdefault('mode', 'bike')
    kwargs.setdefault('time_field', 'free_flow_time')
    kwargs.setdefault('cost_field', 'free_flow_time')
    return ta_stochastic(edges_gdf, od_gdf, **kwargs)


def ta_stochastic_walk(edges_gdf, od_gdf, **kwargs):
    """Stochastic assignment for walk network."""
    kwargs.setdefault('mode', 'walk')
    kwargs.setdefault('time_field', 'free_flow_time')
    kwargs.setdefault('cost_field', 'free_flow_time')
    return ta_stochastic(edges_gdf, od_gdf, **kwargs)


def ta_stochastic_pt(edges_gdf, od_gdf, **kwargs):
    """Stochastic assignment for PT network."""
    kwargs.setdefault('mode', 'pt')
    kwargs.setdefault('time_field', 'trav_time')
    kwargs.setdefault('cost_field', 'trav_time')
    return ta_stochastic(edges_gdf, od_gdf, **kwargs)


def _empty_result(results_df, n):
    """Return empty result dict for failed assignments."""
    results_df['flow'] = np.zeros(n)
    results_df['time'] = np.zeros(n)
    return {
        'network': results_df,
        'total_travel_time': np.inf,
        'computation_time': 0,
        'rgap': np.inf,
        'flow': np.zeros(n),
        'time': np.zeros(n),
        'convergence': None
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_vc_histogram(results_df, capacity_col='capacity', bins=5):
    """Plot V/C ratio histogram."""
    ratios = results_df['flow'] / results_df[capacity_col].replace(0, np.inf)
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

    if len(ratios) == 0:
        print("No valid V/C ratios to plot")
        return None, None

    counts, edges = np.histogram(ratios, bins=bins)
    cmap = mcolors.LinearSegmentedColormap.from_list('', ['#2ecc71', '#f1c40f', '#e74c3c'])
    colors = [cmap(i / max(1, bins - 1)) for i in range(bins)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(edges[:-1], counts, width=edges[1] - edges[0], color=colors, edgecolor='black', alpha=0.8, align='edge')

    for pos, cnt in zip(edges[:-1], counts):
        if cnt > 0:
            ax.text(pos + (edges[1] - edges[0]) / 2, cnt, str(int(cnt)), ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Volume/Capacity Ratio')
    ax.set_ylabel('Number of Links')
    ax.set_title('Link Congestion Distribution')
    ax.grid(axis='y', alpha=0.6, linestyle='--')
    plt.tight_layout()

    return fig, ax