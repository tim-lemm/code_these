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


def _build_graph(edges_gdf, zones, time_field='free_flow_time', cost_field='free_flow_time'):
    """Build AequilibraE Graph from edges GeoDataFrame."""
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df['direction'] = 1
    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    network_df.index = list(range(len(network_df)))
    network_df["link_id"] = network_df.reset_index().index + 1
    network_df = network_df.astype({"a_node": "int64", "b_node": "int64", "direction": "int64", "link_id": "int64"})

    g = Graph()
    g.cost = network_df[cost_field].values
    g.capacity = network_df['capacity'].values
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


def _build_graph_for_stochastic(network_df, zones, time_field, cost_field):
    """Build graph for stochastic assignment."""
    network_df = network_df.copy()
    network_df['direction'] = 1
    network_df.index = list(range(len(network_df)))
    network_df["link_id"] = network_df.reset_index().index + 1
    network_df = network_df.astype({"a_node": "int64", "b_node": "int64", "direction": "int64", "link_id": "int64"})

    g = Graph()
    g.cost = network_df[cost_field].values
    g.capacity = network_df['capacity'].values
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


def _build_transit_graph(edges_gdf, zones, time_field='trav_time', freq_field='freq'):
    """
    Build AequilibraE TransitGraph from transit edges GeoDataFrame.

    Args:
        edges_gdf: GeoDataFrame with transit edges (a_node, b_node, trav_time, freq)
        zones: Number of zones
        time_field: Column name for travel time
        freq_field: Column name for frequency

    Returns:
        TransitGraph configured for optimal-strategies assignment
    """
    from aequilibrae.paths import TransitGraph

    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df['direction'] = 1
    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    network_df.index = list(range(len(network_df)))
    network_df["link_id"] = network_df.reset_index().index + 1
    network_df = network_df.astype({
        "a_node": "int64",
        "b_node": "int64",
        "direction": "int64",
        "link_id": "int64"
    })
    network_df[time_field] = network_df[time_field].astype(float)
    network_df[freq_field] = network_df[freq_field].astype(float)

    centroids = np.arange(1, zones + 1).astype(int)

    g = TransitGraph()
    g.cost = network_df[time_field].values
    g.capacity = network_df[freq_field].values
    g.free_flow = network_df[time_field].values
    g.network = network_df
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(centroids)
    g.cost = np.array(g.cost, copy=True)
    g.set_graph(cost_field=time_field)
    g.set_skimming([time_field, freq_field])
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id
    g.od_node_mapping = pd.DataFrame({
        "taz_id": centroids,
        "node_id": centroids
    })
    g.centroids = centroids

    return g


def _skim_networkx(edges_gdf, od_gdf, cost_field):
    """NetworkX implementation."""
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df[cost_field] = network_df[cost_field].fillna(0.01)

    G = nx.DiGraph()
    for _, row in network_df.iterrows():
        G.add_edge(row['a_node'], row['b_node'], weight=row[cost_field])

    od_pairs = od_gdf[['origin', 'destination']].drop_duplicates()

    results = []
    for _, row in od_pairs.iterrows():
        o, d = int(row['origin']), int(row['destination'])
        try:
            cost = nx.dijkstra_path_length(G, o, d, weight='weight')
        except nx.NetworkXNoPath:
            cost = np.inf
        results.append({'origin': o, 'destination': d, 'cost': cost})

    return pd.DataFrame(results)


def _skim_aequilibrae(edges_gdf, od_gdf, cost_field):
    """AequilibraE implementation for faster skimming."""
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df[cost_field] = network_df[cost_field].fillna(0.01)
    network_df['capacity'] = network_df.get('capacity', pd.Series([99999] * len(network_df))).fillna(99999)

    zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))
    demand = _build_demand_matrix(od_gdf, name='skim')

    g = _build_graph(network_df, zones, time_field=cost_field, cost_field=cost_field)
    g.set_skimming([cost_field])

    assig = TrafficAssignment()
    assig.set_classes([TrafficClass('skim', g, demand)])
    assig.set_vdf('BPR')
    assig.set_vdf_parameters({'alpha': 'alpha', 'beta': 'beta'})
    assig.set_capacity_field('capacity')
    assig.set_time_field(cost_field)
    assig.set_algorithm('msa')
    assig.max_iter = 1

    assig.execute()

    skim_matrix = assig.classes[0].results.skims.matrix[cost_field]
    zone_ids = assig.classes[0].results.skims.index[:]

    od_pairs = od_gdf[['origin', 'destination']].drop_duplicates()
    results = []
    for _, row in od_pairs.iterrows():
        o, d = int(row['origin']), int(row['destination'])
        try:
            o_idx = np.where(zone_ids == o)[0][0]
            d_idx = np.where(zone_ids == d)[0][0]
            cost = skim_matrix[o_idx, d_idx]
            if cost == 0 and o != d:
                cost = np.inf
        except (IndexError, KeyError):
            cost = np.inf
        results.append({'origin': o, 'destination': d, 'cost': cost})

    return pd.DataFrame(results)


# =============================================================================
# TRAFFIC ASSIGNMENT FUNCTIONS (DUE, STOCHASTIC (uncongested), OPTIMAL STRATEGIES)
# =============================================================================

def ta_due(edges_gdf, od_gdf, algorithm='bfw', max_iter=500, tolerance=1e-4,
           time_field='free_flow_time', cost_field='free_flow_time', verbose=False):
    """
    Deterministic User Equilibrium (DUE) traffic assignment.
    """
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df['alpha'] = network_df['alpha'].fillna(0.15)
    network_df['beta'] = network_df['beta'].fillna(4.0)
    network_df['capacity'] = network_df['capacity'].fillna(99999)
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
    g.capacity = network_df['capacity'].values
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
    assignment.set_capacity_field("capacity")
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
        results_df['ratio'] = flows / results_df['capacity'].replace(0, np.inf)
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


def ta_os_pt(edges_gdf, od_gdf, time_field='trav_time', freq_field='freq',
             algorithm='optimal-strategies'):
    """
    Transit assignment using optimal-strategies algorithm.

    Performs frequency-based transit assignment where passengers choose
    routes based on expected travel time considering service frequencies.

    Args:
        edges_gdf: GeoDataFrame with transit edges containing:
            - a_node, b_node: Node IDs
            - trav_time: Travel time on link
            - freq: Service frequency (trips per time unit)
        od_gdf: GeoDataFrame with OD demand (origin, destination, demand)
        time_field: Column for travel time (default: 'trav_time')
        freq_field: Column for frequency (default: 'freq')
        algorithm: Assignment algorithm (default: 'optimal-strategies')

    Returns:
        dict with:
            - 'network': DataFrame with flow results
            - 'total_travel_time': Total network travel time
            - 'computation_time': Execution time in seconds
            - 'flow': Array of link flows
            - 'time': Array of link travel times
    """
    from aequilibrae.paths import TransitAssignment, TransitClass

    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df[time_field] = network_df[time_field].fillna(100.0)
    network_df[freq_field] = network_df[freq_field].fillna(0.001)

    zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))
    demand = _build_demand_matrix(od_gdf, name='pt')

    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    graph = _build_transit_graph(network_df, zones, time_field=time_field, freq_field=freq_field)

    results_df = graph.network.copy()
    start_time = time.time()

    try:
        assig_class = TransitClass(name='pt', graph=graph, matrix=demand)
        assig_class.set_demand_matrix_core('pt')

        assignment = TransitAssignment()
        assignment.add_class(assig_class)
        assignment.set_time_field(time_field)
        assignment.set_frequency_field(freq_field)
        assignment.set_skimming_fields([time_field])
        assignment.set_algorithm(algorithm)

        assignment.execute()
        computation_time = time.time() - start_time

        results = assignment.results()
        results['link_id'] = results.index
        results = results.sort_values(by='link_id').reset_index(drop=True)

        flows = results['pt_volume'].values if 'pt_volume' in results.columns else np.zeros(len(results_df))
        times = results_df[time_field].values

        total_travel_time = np.sum(flows * times)

        results_df['flow'] = flows
        results_df['time'] = times

        print(f"Transit OS (pt) completed in {computation_time:.2f}s | TTT: {total_travel_time:.0f}")

        return {
            'network': results_df,
            'total_travel_time': total_travel_time,
            'computation_time': computation_time,
            'flow': flows,
            'time': times
        }

    except Exception as e:
        print(f"Error in transit OS assignment: {e}")
        import traceback
        traceback.print_exc()
        return _empty_result(results_df, len(network_df))


def ta_stochastic(edges_gdf, od_gdf, mode='car',
                  time_field='free_flow_time', cost_field='free_flow_time',
                  algorithm='bfsle', max_routes=5, max_depth=100, max_misses=100,
                  beta=1.0, cutoff_prob=0.0, penalty=1.0, seed=0, cores=1, verbose=False):
    """
    Stochastic (uncongested) traffic assignment using Route Choice.
    """
    network_df = edges_gdf.copy()
    if 'geometry' in network_df.columns:
        network_df = pd.DataFrame(network_df.drop(columns=['geometry']))

    network_df['capacity'] = network_df['capacity'].fillna(99999)
    network_df[time_field] = network_df[time_field].fillna(100.01)
    if cost_field != time_field:
        network_df[cost_field] = network_df[cost_field].fillna(100.01)

    zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))
    demand = _build_demand_matrix(od_gdf)

    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    graph = _build_graph_for_stochastic(network_df, zones, time_field, cost_field)

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
# MODE CHOICE UTILITIES
# =============================================================================

def mc_get_skim_matrix(edges_gdf, od_gdf, cost_field='free_flow_time', engine='aeq'):
    """
    Get OD skim matrix (shortest path cost per OD pair).

    Args:
        edges_gdf: GeoDataFrame with network edges
        od_gdf: GeoDataFrame with OD pairs (origin, destination, demand)
        cost_field: Column to use for travel cost
        engine: 'aeq' for AequilibraE (default, faster) or 'netx' for NetworkX

    Returns:
        DataFrame with columns: origin, destination, cost
    """
    if engine == 'aeq':
        return _skim_aequilibrae(edges_gdf, od_gdf, cost_field)
    else:
        return _skim_networkx(edges_gdf, od_gdf, cost_field)


def mc_calc_probabilities(mode_costs, theta=1.0):
    """
    Calculate mode choice probabilities using multinomial logit.

    P(mode) = exp(-theta * cost) / sum(exp(-theta * costs))

    Args:
        mode_costs: dict {mode_name: DataFrame with origin, destination, cost/logsum}
        theta: Scale parameter

    Returns:
        DataFrame with columns: origin, destination, mode, probability, cost
    """
    modes = list(mode_costs.keys())

    # Get common OD pairs
    first_df = mode_costs[modes[0]]
    merged = first_df[['origin', 'destination']].copy()

    for mode in modes:
        df = mode_costs[mode].copy()
        cost_col = 'logsum' if 'logsum' in df.columns else 'cost'
        merged = merged.merge(
            df[['origin', 'destination', cost_col]].rename(columns={cost_col: f'cost_{mode}'}),
            on=['origin', 'destination'], how='outer'
        )

    results = []
    for _, row in merged.iterrows():
        o, d = int(row['origin']), int(row['destination'])
        costs = {m: row.get(f'cost_{m}', np.inf) for m in modes}

        # Calculate utilities and probabilities
        utilities = {m: -theta * c for m, c in costs.items()}
        max_u = max(u for u in utilities.values() if not np.isinf(u) or u > 0)

        if np.isinf(max_u) and max_u < 0:
            probs = {m: 0.0 for m in modes}
        else:
            exp_u = {m: np.exp(u - max_u) if not (np.isinf(u) and u < 0) else 0.0
                     for m, u in utilities.items()}
            sum_exp = sum(exp_u.values())
            probs = {m: exp_u[m] / sum_exp if sum_exp > 0 else 0.0 for m in modes}

        for mode in modes:
            results.append({
                'origin': o, 'destination': d, 'mode': mode,
                'probability': probs[mode], 'cost': costs[mode]
            })

    return pd.DataFrame(results)


def mc_split_demand(od_gdf, mode_probs):
    """
    Split total OD demand by mode based on probabilities.

    Args:
        od_gdf: GeoDataFrame with total demand (origin, destination, demand)
        mode_probs: DataFrame from mc_calc_probabilities

    Returns:
        dict {mode_name: OD GeoDataFrame with mode-specific demand}
    """
    modes = mode_probs['mode'].unique()

    probs_wide = mode_probs.pivot_table(
        index=['origin', 'destination'], columns='mode', values='probability'
    ).reset_index()

    demand_df = od_gdf[['origin', 'destination', 'demand']].copy()
    merged = demand_df.merge(probs_wide, on=['origin', 'destination'], how='left')

    for mode in modes:
        if mode not in merged.columns:
            merged[mode] = 0.0
        merged[mode] = merged[mode].fillna(0.0)

    result = {}
    for mode in modes:
        mode_od = merged[['origin', 'destination']].copy()
        mode_od['demand'] = merged['demand'] * merged[mode]

        if 'geometry' in od_gdf.columns:
            mode_od = mode_od.merge(
                od_gdf[['origin', 'destination', 'geometry']],
                on=['origin', 'destination'], how='left'
            )
            mode_od = gpd.GeoDataFrame(mode_od, geometry='geometry')

        result[mode] = mode_od

    return result


def mc_summary(mode_probs, od_gdf=None):
    """
    Calculate summary statistics for mode choice.

    Args:
        mode_probs: DataFrame from mc_calc_probabilities
        od_gdf: Optional OD demand for weighted stats

    Returns:
        dict with mode_shares, avg_costs, mode_demands
    """
    modes = mode_probs['mode'].unique()

    avg_probs = mode_probs.groupby('mode')['probability'].mean().to_dict()
    avg_costs = mode_probs.groupby('mode')['cost'].apply(
        lambda x: x.replace([np.inf, -np.inf], np.nan).mean()
    ).to_dict()

    if od_gdf is not None:
        merged = mode_probs.merge(
            od_gdf[['origin', 'destination', 'demand']],
            on=['origin', 'destination'], how='left'
        )
        merged['demand'] = merged['demand'].fillna(0)
        merged['weighted'] = merged['probability'] * merged['demand']

        totals = merged.groupby('mode')['weighted'].sum()
        mode_shares = (totals / totals.sum()).to_dict()
        mode_demands = totals.to_dict()
    else:
        mode_shares = avg_probs
        mode_demands = None

    return {
        'mode_shares': mode_shares,
        'avg_probabilities': avg_probs,
        'avg_costs': avg_costs,
        'mode_demands': mode_demands,
        'n_od_pairs': len(mode_probs) // len(modes)
    }


# =============================================================================
# COMBINED MODE CHOICE AND ASSIGNMENT METHODS
# =============================================================================

def mca_sequential(edges_car, edges_bike=None, edges_walk=None, edges_pt=None, od_gdf=None,
                   theta=0.1, car_algorithm='bfw', car_max_iter=100, car_tol=1e-4):
    """
    Sequential Mode Choice and Assignment (Method 1).
    Revised to make non-car modes optional.
    """
    if od_gdf is None:
        raise ValueError("od_gdf must be provided.")

    print("\n" + "=" * 60)
    print("SEQUENTIAL MODE CHOICE AND ASSIGNMENT (Method 1)")
    print("=" * 60)

    # Step 1: Calculate initial skim costs
    print("\nStep 1: Computing initial travel costs (skims)...")

    skims = {}

    # Car (Required)
    skim_car = mc_get_skim_matrix(edges_car, od_gdf, cost_field='free_flow_time')
    skims['car'] = skim_car
    print(f"  Car avg cost:  {skim_car['cost'].mean():.1f} min")

    # Optional Modes
    if edges_bike is not None:
        skim_bike = mc_get_skim_matrix(edges_bike, od_gdf, cost_field='free_flow_time')
        skims['bike'] = skim_bike
        print(f"  Bike avg cost: {skim_bike['cost'].mean():.1f} min")
    else:
        skims['bike'] = None

    if edges_walk is not None:
        skim_walk = mc_get_skim_matrix(edges_walk, od_gdf, cost_field='free_flow_time')
        skims['walk'] = skim_walk
        print(f"  Walk avg cost: {skim_walk['cost'].mean():.1f} min")
    else:
        skims['walk'] = None

    if edges_pt is not None:
        skim_pt = mc_get_skim_matrix(edges_pt, od_gdf, cost_field='trav_time')
        skims['pt'] = skim_pt
        print(f"  PT avg cost:   {skim_pt['cost'].mean():.1f} min")
    else:
        skims['pt'] = None

    # Step 2: Mode choice
    print(f"\nStep 2: Calculating mode choice (theta={theta})...")

    # Filter skims to only include available modes for probability calculation
    available_skims = {k: v for k, v in skims.items() if v is not None}

    mode_probs = mc_calc_probabilities(available_skims, theta=theta)
    stats = mc_summary(mode_probs, od_gdf)

    print("  Mode shares:")
    for mode, share in stats['mode_shares'].items():
        print(f"    {mode}: {share:.1%}")

    # Step 3: Split demand
    print("\nStep 3: Splitting demand by mode...")

    od_by_mode = mc_split_demand(od_gdf, mode_probs)

    for mode, od_mode in od_by_mode.items():
        if od_mode is not None and not od_mode.empty:
            print(f"  {mode}: {od_mode['demand'].sum():,.0f} trips")

    # Step 4: Assign to networks
    print("\nStep 4: Assigning mode-specific demand to networks...")

    assignment_results = {
        'car': None, 'bike': None, 'walk': None, 'pt': None
    }

    # Car (DUE)
    print("\n  [Car - DUE Assignment]")
    car_results = ta_due(
        edges_car, od_by_mode['car'],
        algorithm=car_algorithm, max_iter=car_max_iter, tolerance=car_tol
    )
    assignment_results['car'] = car_results

    # Bike (Stochastic)
    if edges_bike is not None and 'bike' in od_by_mode:
        print("\n  [Bike - Stochastic Assignment]")
        bike_results = ta_stochastic_bike(edges_bike, od_by_mode['bike'], max_routes=5)
        assignment_results['bike'] = bike_results

    # Walk (AON)
    if edges_walk is not None and 'walk' in od_by_mode:
        print("\n  [Walk - AON Assignment]")
        walk_results = ta_stochastic_walk(edges_walk, od_by_mode['walk'], max_routes=1)
        assignment_results['walk'] = walk_results

    # PT (Stochastic)
    if edges_pt is not None and 'pt' in od_by_mode:
        print("\n  [PT - Stochastic Assignment]")
        pt_results = ta_stochastic_pt(edges_pt, od_by_mode['pt'], max_routes=3)
        assignment_results['pt'] = pt_results

    print("\n" + "-" * 60)
    print("Sequential Method Complete")
    print("-" * 60)

    return {
        'mode_shares': stats['mode_shares'],
        'mode_probs': mode_probs,
        'od_by_mode': od_by_mode,
        'assignment_results': assignment_results,
        'skims': skims
    }


def mca_iterative(edges_car, edges_bike=None, edges_walk=None, edges_pt=None, od_gdf=None,
                  theta=0.1, max_iter=3, convergence_tol=0.01,
                  car_algorithm='bfw', car_max_iter=100, car_tol=1e-3, verbose=False):
    """
    Iterative Mode Choice and Assignment (Method 2).
    Revised to make non-car modes optional.
    """
    if od_gdf is None:
        raise ValueError("od_gdf must be provided.")

    if verbose:
        print("\n" + "=" * 60)
        print("ITERATIVE MODE CHOICE AND ASSIGNMENT (Method 2)")
        print("=" * 60)

        # Initialize skims with free-flow travel times
        print("\nInitializing with free-flow travel costs...")

    skims = {}

    # Car (Required)
    skim_car = mc_get_skim_matrix(edges_car, od_gdf, cost_field='free_flow_time')
    skims['car'] = skim_car

    # Optional
    if edges_bike is not None:
        skims['bike'] = mc_get_skim_matrix(edges_bike, od_gdf, cost_field='free_flow_time')
    else:
        skims['bike'] = None

    if edges_walk is not None:
        skims['walk'] = mc_get_skim_matrix(edges_walk, od_gdf, cost_field='free_flow_time')
    else:
        skims['walk'] = None

    if edges_pt is not None:
        skims['pt'] = mc_get_skim_matrix(edges_pt, od_gdf, cost_field='trav_time')
    else:
        skims['pt'] = None

    convergence_history = []
    prev_shares = None
    assignment_results = {'car': None, 'bike': None, 'walk': None, 'pt': None}

    for iteration in range(1, max_iter + 1):
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Mode choice (only using available skims)
        available_skims = {k: v for k, v in skims.items() if v is not None}
        mode_probs = mc_calc_probabilities(available_skims, theta=theta)
        stats = mc_summary(mode_probs, od_gdf)
        current_shares = stats['mode_shares']

        convergence_history.append(current_shares.copy())

        if verbose:
            print("  Mode shares:", end=" ")
            for mode, share in current_shares.items():
                print(f"{mode}={share:.1%}", end=" ")
            print()

        # Check convergence
        if prev_shares is not None:
            # Only check convergence on modes that actually exist
            max_change = max(
                abs(current_shares[m] - prev_shares[m])
                for m in current_shares.keys()
            )
            if verbose:
                print(f"  Max share change: {max_change:.4f}")

            if max_change < convergence_tol:
                if verbose:
                    print(f"\n  Converged at iteration {iteration}")
                break

        prev_shares = current_shares.copy()

        # Split demand
        od_by_mode = mc_split_demand(od_gdf, mode_probs)

        # Assign to networks
        if verbose:
            print("  Running assignments...")

        # Car (DUE) - updates congested times
        car_results = ta_due(
            edges_car, od_by_mode['car'],
            algorithm=car_algorithm, max_iter=car_max_iter, tolerance=car_tol, verbose=verbose
        )
        assignment_results['car'] = car_results

        # Bike (Stochastic)
        if edges_bike is not None and 'bike' in od_by_mode:
            bike_results = ta_stochastic_bike(edges_bike, od_by_mode['bike'], max_routes=5, verbose=verbose)
            assignment_results['bike'] = bike_results

        # Walk (AON)
        if edges_walk is not None and 'walk' in od_by_mode:
            walk_results = ta_stochastic_walk(edges_walk, od_by_mode['walk'], max_routes=1, verbose=verbose)
            assignment_results['walk'] = walk_results

        # PT (Stochastic)
        if edges_pt is not None and 'pt' in od_by_mode:
            pt_results = ta_stochastic_pt(edges_pt, od_by_mode['pt'], max_routes=3, verbose=verbose)
            assignment_results['pt'] = pt_results

        # Update skims with congested travel times
        if verbose:
            print("  Updating travel costs...")

        # Update car skim with congested times
        car_net = car_results['network'].copy()
        G_car = nx.DiGraph()
        for _, row in car_net.iterrows():
            G_car.add_edge(row['a_node'], row['b_node'], weight=row['time'])

        updated_car_costs = []
        for _, row in skims['car'].iterrows():
            o, d = int(row['origin']), int(row['destination'])
            try:
                cost = nx.dijkstra_path_length(G_car, o, d, weight='weight')
            except nx.NetworkXNoPath:
                cost = np.inf
            updated_car_costs.append(cost)

        skims['car']['cost'] = updated_car_costs

        if verbose:
            print(f"  Updated car avg cost: {np.mean([c for c in updated_car_costs if c < np.inf]):.1f} min")

    else:
        if verbose:
            print(f"\n  Maximum iterations ({max_iter}) reached")

    if verbose:
        print("\n" + "-" * 60)
        print("Iterative Method Complete")
        print("-" * 60)

    return {
        'mode_shares': current_shares,
        'mode_probs': mode_probs,
        'od_by_mode': od_by_mode,
        'assignment_results': assignment_results,
        'convergence': convergence_history,
        'iterations': iteration,
        'skims': skims
    }


def mca_iterative_nested(edges_car, edges_bike=None, edges_walk=None, edges_pt=None, od_gdf=None,
                         theta=0.1, mu=1.0, max_routes=5, max_iter=3, convergence_tol=0.01,
                         car_algorithm='bfw', car_max_iter=100, car_tol=1e-4):
    """
    Iterative Mode Choice and Assignment with NESTED LOGIT.
    Revised to make non-car modes optional.
    """
    if od_gdf is None:
        raise ValueError("od_gdf must be provided.")

    print("\n" + "=" * 60)
    print("ITERATIVE NESTED LOGIT MODE CHOICE & ASSIGNMENT")
    print("=" * 60)
    print(f"Parameters: theta={theta}, mu={mu}, max_routes={max_routes}")

    # Work with copies of edges that we can update
    edges_car_work = edges_car.copy()

    # Ensure capacity and free_flow_time are filled
    if 'capacity' in edges_car_work.columns:
        edges_car_work['capacity'] = edges_car_work['capacity'].fillna(99999)

    # Robustly calculate free_flow_time
    if 'speed' in edges_car_work.columns and 'length' in edges_car_work.columns:
        speed = edges_car_work['speed'].replace(0, 0.1)
        edges_car_work['free_flow_time'] = (edges_car_work['length'] / 1000) / speed * 60

    if 'free_flow_time' not in edges_car_work.columns:
        edges_car_work['free_flow_time'] = 100.0

    edges_car_work['free_flow_time'] = edges_car_work['free_flow_time'].fillna(100.0)

    # Ensure BPR parameters
    if 'alpha' not in edges_car_work.columns:
        edges_car_work['alpha'] = 0.15
    if 'beta' not in edges_car_work.columns:
        edges_car_work['beta'] = 4.0

    convergence_history = []
    prev_shares = None
    assignment_results = {'car': None, 'bike': None, 'walk': None, 'pt': None}
    logsums = {}

    for iteration in range(1, max_iter + 1):
        print(f"\n--- Iteration {iteration} ---")

        # 1. Compute Logsums (Lower Nest)
        print("  Computing route choice logsums...")

        # Helper to compute logsum for a mode
        def compute_mode_logsum(edges, cost_col):
            # Use AequilibraE's RouteChoice for logsum calculation
            zones = int(max(od_gdf['origin'].max(), od_gdf['destination'].max()))

            net_df = edges.copy()
            if 'geometry' in net_df.columns:
                net_df = net_df.drop(columns=['geometry'])

            if cost_col not in net_df.columns:
                if 'speed' in net_df.columns and 'length' in net_df.columns:
                    speed = net_df['speed'].replace(0, 0.1)
                    net_df[cost_col] = (net_df['length'] / 1000) / speed * 60
                else:
                    net_df[cost_col] = net_df.get('free_flow_time', 100.0)

            net_df[cost_col] = net_df[cost_col].fillna(100.0)

            graph = _build_graph_for_stochastic(net_df, zones, cost_col, cost_col)

            demand = _build_demand_matrix(od_gdf)

            rc = RouteChoice(graph)
            rc.set_cores(1)
            rc.set_choice_set_generation(
                algorithm='bfsle', max_routes=max_routes, max_depth=100,
                max_misses=100, beta=1.0, cutoff_prob=0.0, penalty=1.0, seed=0
            )
            rc.add_demand(demand)
            rc.prepare(nodes=None)
            rc.execute(perform_assignment=True)

            res = rc.get_results().reset_index()

            rename_map = {}
            for c in res.columns:
                if c in ['origin_id', 'origin id']: rename_map[c] = 'origin'
                if c in ['destination_id', 'destination id']: rename_map[c] = 'destination'
            res = res.rename(columns=rename_map)

            res_cost_col = next((c for c in [cost_col, 'cost', 'time', 'free_flow_time'] if c in res.columns), None)

            logsum_list = []
            if res_cost_col:
                for (o, d), group in res.groupby(['origin', 'destination']):
                    costs = group[res_cost_col].values
                    if len(costs) == 0:
                        ls = -np.inf
                    else:
                        scaled = -mu * costs
                        max_s = np.max(scaled)
                        ls = (1 / mu) * (max_s + np.log(np.sum(np.exp(scaled - max_s))))
                    logsum_list.append({'origin': o, 'destination': d, 'logsum': ls})
            else:
                print(f"    Warning: Cost column not found in results. Available: {res.columns.tolist()}")

            df_result = pd.DataFrame(logsum_list)
            if 'logsum' not in df_result.columns:
                df_result['origin'] = []
                df_result['destination'] = []
                df_result['logsum'] = []

            return df_result

        # Car logsum (Required)
        logsums['car'] = compute_mode_logsum(edges_car_work, 'free_flow_time')

        # Other modes (Optional)
        if edges_bike is not None:
            logsums['bike'] = compute_mode_logsum(edges_bike, 'free_flow_time')
        else:
            logsums['bike'] = None

        if edges_walk is not None:
            logsums['walk'] = compute_mode_logsum(edges_walk, 'free_flow_time')
        else:
            logsums['walk'] = None

        if edges_pt is not None:
            logsums['pt'] = compute_mode_logsum(edges_pt, 'trav_time')
        else:
            logsums['pt'] = None

        # 2. Mode Choice (Upper Nest)
        mode_inputs = {}
        # Only process available logsums
        for m, df in logsums.items():
            if df is not None:
                df_input = df.copy()
                df_input['logsum'] = -df_input['logsum']
                df_input['cost'] = df_input['logsum']
                mode_inputs[m] = df_input

        mode_probs = mc_calc_probabilities(mode_inputs, theta=theta)
        stats = mc_summary(mode_probs, od_gdf)
        current_shares = stats['mode_shares']

        convergence_history.append(current_shares.copy())

        print("  Mode shares:", end=" ")
        for mode, share in current_shares.items():
            print(f"{mode}={share:.1%}", end=" ")
        print()

        # Check convergence
        if prev_shares is not None:
            max_change = max(
                abs(current_shares[m] - prev_shares[m])
                for m in current_shares.keys()
            )
            print(f"  Max share change: {max_change:.4f}")

            if max_change < convergence_tol:
                print(f"\n  Converged at iteration {iteration}")
                break

        prev_shares = current_shares.copy()

        # 3. Split Demand
        od_by_mode = mc_split_demand(od_gdf, mode_probs)

        # 4. Assignment
        print("  Running assignments...")

        # Car (DUE)
        car_results = ta_due(
            edges_car_work, od_by_mode['car'],
            algorithm=car_algorithm, max_iter=car_max_iter, tolerance=car_tol
        )
        assignment_results['car'] = car_results

        # Other modes (Optional)
        if edges_bike is not None and 'bike' in od_by_mode:
            assignment_results['bike'] = ta_stochastic_bike(edges_bike, od_by_mode['bike'], max_routes=max_routes)

        if edges_walk is not None and 'walk' in od_by_mode:
            assignment_results['walk'] = ta_stochastic_walk(edges_walk, od_by_mode['walk'], max_routes=1)

        if edges_pt is not None and 'pt' in od_by_mode:
            assignment_results['pt'] = ta_stochastic_pt(edges_pt, od_by_mode['pt'], max_routes=max_routes)

        # 5. Update Car Costs
        edges_car_work['free_flow_time'] = car_results['network']['time'].values

        print(
            f"  Updated car avg cost: {logsums['car']['logsum'].replace([-np.inf], np.nan).mean():.1f} (utility units)")

    else:
        print(f"\n  Maximum iterations ({max_iter}) reached")

    print("\n" + "-" * 60)
    print("Iterative Nested Logit Method Complete")
    print("-" * 60)

    return {
        'mode_shares': current_shares,
        'mode_probs': mode_probs,
        'od_by_mode': od_by_mode,
        'assignment_results': assignment_results,
        'convergence': convergence_history,
        'iterations': iteration,
        'logsums': logsums,
        'stats': stats
    }


def plot_mca_convergence(convergence_history, title='Mode Share Convergence'):
    """
    Plot mode share convergence across iterations.

    Args:
        convergence_history: List of dicts with mode shares per iteration
        title: Plot title

    Returns:
        fig, ax
    """
    modes = list(convergence_history[0].keys())
    iterations = range(1, len(convergence_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))

    for mode, color in zip(modes, colors):
        shares = [h[mode] for h in convergence_history]
        ax.plot(iterations, shares, 'o-', label=mode, color=color, linewidth=2, markersize=8)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mode Share', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    plt.tight_layout()
    return fig, ax


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_network(edges_gdf, nodes_df, width_col='flow', color_col=None,
                 base_width=0.1, width_scale=1.1, cmap='RdYlGn_r',
                 show_nodes=True, node_size=5, node_color='grey',
                 a_node_col='a_node', b_node_col='b_node',
                 node_id_col='id', node_x_col='x', node_y_col='y',
                 figsize=(14, 10), title=None, vmin=None, vmax=None,
                 colorbar_label=None):
    """
    Plot network with edge widths (bandwidth) based on attribute values.

    Args:
        edges_gdf: GeoDataFrame with edges
        nodes_df: DataFrame with node coordinates
        width_col: Column for edge width (bandwidth), typically 'flow'
        color_col: Column for edge color, e.g., 'ratio' for V/C
        base_width: Base line width
        width_scale: Scale factor for width
        cmap: Colormap name (default 'RdYlGn_r' for green-yellow-red)
        show_nodes: Whether to show nodes
        node_size: Size of nodes (default=5, small)
        node_color: Color of nodes
        figsize: Figure size
        title: Plot title
        vmin, vmax: Min/max values for colormap normalization
        colorbar_label: Custom label for colorbar

    Returns:
        fig, ax tuple
    """
    # Reset index to align with numpy arrays if filtered
    edges_gdf = edges_gdf.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    x_range = nodes_df[node_x_col].max() - nodes_df[node_x_col].min()
    y_range = nodes_df[node_y_col].max() - nodes_df[node_y_col].min()
    scale = min(x_range, y_range) / 100

    scaled_base = scale * base_width
    scaled_scale = scale * width_scale * 0.0001

    node_coords = {r[node_id_col]: (r[node_x_col], r[node_y_col])
                   for _, r in nodes_df.iterrows()}

    width_vals = edges_gdf[width_col].fillna(0).values if width_col in edges_gdf.columns else np.ones(len(edges_gdf))

    if color_col is None:
        color_col = width_col

    if color_col in edges_gdf.columns:
        color_vals = edges_gdf[color_col].fillna(0).values

        # Use provided vmin/vmax or compute from data
        if vmin is None:
            vmin = np.nanmin(color_vals)
        if vmax is None:
            vmax = np.nanmax(color_vals)
        if vmin == vmax:
            vmax = vmin + 1

        norm = plt.Normalize(vmin, vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        use_cmap = True
    else:
        use_cmap = False

    # Use enumerate to get integer index for numpy array access
    for idx, edge in edges_gdf.iterrows():
        a_coords = node_coords.get(edge[a_node_col])
        b_coords = node_coords.get(edge[b_node_col])
        if not a_coords or not b_coords:
            continue

        # Use idx (which is now 0..N-1 due to reset_index) to access arrays
        width = scaled_base + width_vals[idx] * scaled_scale
        polygon = _create_offset_polygon([a_coords, b_coords], width)

        if polygon:
            color = sm.to_rgba(color_vals[idx]) if use_cmap else 'steelblue'
            x, y = polygon.exterior.xy
            ax.fill(x, y, color=color, alpha=0.7, edgecolor='gray', linewidth=0.3)

    if show_nodes:
        ax.scatter(nodes_df[node_x_col], nodes_df[node_y_col], s=node_size, c=node_color, zorder=5)

    if use_cmap:
        # Determine colorbar label
        if colorbar_label is None:
            colorbar_label = color_col.replace('_', ' ').title()
        cbar = plt.colorbar(sm, ax=ax, label=colorbar_label, shrink=0.8)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(title or f'Network (width by {width_col})')
    plt.tight_layout()

    return fig, ax


def _create_offset_polygon(coords, width):
    """Create polygon offset to right of line."""
    if len(coords) < 2:
        return None

    right_pts, left_pts = [], []
    for i, (cx, cy) in enumerate(coords):
        if i == 0:
            dx, dy = coords[1][0] - coords[0][0], coords[1][1] - coords[0][1]
        elif i == len(coords) - 1:
            dx, dy = coords[-1][0] - coords[-2][0], coords[-1][1] - coords[-2][1]
        else:
            dx = (coords[i + 1][0] - coords[i - 1][0]) / 2
            dy = (coords[i + 1][1] - coords[i - 1][1]) / 2

        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            dx, dy = dx / length, dy / length

        perp_x, perp_y = dy, -dx
        right_pts.append((cx + perp_x * width, cy + perp_y * width))
        left_pts.append((cx, cy))

    try:
        polygon = Polygon(right_pts + left_pts[::-1])
        return polygon if polygon.is_valid else polygon.buffer(0)
    except:
        return None


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


def plot_mode_shares(mode_stats, title='Mode Shares'):
    """Plot mode share pie chart."""
    shares = mode_stats.get('mode_shares', mode_stats) if isinstance(mode_stats, dict) else mode_stats

    modes = list(shares.keys())
    values = [shares[m] for m in modes]
    colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values, labels=modes, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=[0.02] * len(modes)
    )

    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight('bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def plot_mode_comparison(mode_probs, figsize=(12, 5)):
    """Plot cost distribution and probabilities by mode."""
    modes = mode_probs['mode'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(modes)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Cost boxplot
    ax1 = axes[0]
    costs = [mode_probs[mode_probs['mode'] == m]['cost'].replace([np.inf, -np.inf], np.nan).dropna() for m in modes]
    bp = ax1.boxplot(costs, labels=modes, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax1.set_ylabel('Travel Cost')
    ax1.set_title('Cost Distribution by Mode')
    ax1.grid(axis='y', alpha=0.3)

    # Probability bars
    ax2 = axes[1]
    avg_probs = mode_probs.groupby('mode')['probability'].mean()
    bars = ax2.bar(modes, [avg_probs[m] for m in modes], color=colors)
    ax2.set_ylabel('Average Probability')
    ax2.set_title('Mode Choice Probability')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h, f'{h:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig, axes

