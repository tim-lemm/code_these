import os
import time
import random
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from shapely.geometry import LineString, Polygon
from aequilibrae.paths import RouteChoice
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import Graph, TrafficAssignment, TrafficClass
from scipy.sparse import coo_matrix


def create_grid_network(rows=5, cols=5, node_spacing=10, seed=42):
    """
    Create a grid network with random edge attributes.
    Inputs: rows, cols, node_spacing, seed.
    Output: NetworkX Graph with node and edge attributes.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create grid graph
    G = nx.grid_2d_graph(rows, cols)

    # Convert node labels from (i,j) to single integer
    mapping = {}
    node_id = 1
    for node in G.nodes():
        mapping[node] = node_id
        node_id += 1

    G = nx.relabel_nodes(G, mapping)

    # Add node positions
    node_id = 1
    for i in range(rows):
        for j in range(cols):
            x = j * node_spacing
            y = i * node_spacing
            G.nodes[node_id]['x'] = x
            G.nodes[node_id]['y'] = y
            G.nodes[node_id]['id'] = node_id
            node_id += 1

    # Add edge attributes with random values
    for u, v in G.edges():
        # Calculate edge length based on node positions
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Generate random attributes
        capacity = random.uniform(5000, 25000)  # Random capacity between 5000-25000
        speed = random.uniform(30, 80)  # Random speed between 30-80
        free_flow_time = length / speed * random.uniform(0.8, 1.5)  # Based on length with some variation

        G.edges[u, v]['length'] = length
        G.edges[u, v]['capacity'] = capacity
        G.edges[u, v]['speed'] = speed
        G.edges[u, v]['free_flow_time'] = free_flow_time

    return G


def nx_to_dataframes(G, bidirectional=True):
    """
    Convert NetworkX graph to node DataFrame and edge GeoDataFrame.
    Inputs: G (networkx.Graph), bidirectional (bool).
    Outputs: edges_gdf (GeoDataFrame), nodes_df (DataFrame).
    """
    # Create nodes DataFrame
    nodes_data = []
    for node, data in G.nodes(data=True):
        nodes_data.append({
            'id': node,
            'x': data.get('x', 0),
            'y': data.get('y', 0)
        })
    nodes_df = pd.DataFrame(nodes_data)

    # Create edges DataFrame
    edges_data = []
    edge_id = 1

    for u, v, data in G.edges(data=True):
        # Get node coordinates
        u_x, u_y = G.nodes[u]['x'], G.nodes[u]['y']
        v_x, v_y = G.nodes[v]['x'], G.nodes[v]['y']

        # Create edge entry
        edge_entry = {
            'id': edge_id,
            'a_node': u,
            'b_node': v,
            'length': data.get('length', 1.0),
            'capacity': data.get('capacity', 10000),
            'speed': data.get('speed', 50),
            'free_flow_time': data.get('free_flow_time', 1.0)
        }
        edges_data.append(edge_entry)
        edge_id += 1

        # If bidirectional, add reverse edge
        if bidirectional and not G.is_directed():
            edge_entry_reverse = {
                'id': edge_id,
                'a_node': v,
                'b_node': u,
                'length': data.get('length', 1.0),
                'capacity': data.get('capacity', 10000),
                'speed': data.get('speed', 50),
                'free_flow_time': data.get('free_flow_time', 1.0)
            }
            edges_data.append(edge_entry_reverse)
            edge_id += 1

    edges_df = pd.DataFrame(edges_data)

    # Create LineString geometries for each edge
    geometries = []
    for _, edge in edges_df.iterrows():
        a_node = edge['a_node']
        b_node = edge['b_node']

        # Get coordinates from nodes
        a_coords = (G.nodes[a_node]['x'], G.nodes[a_node]['y'])
        b_coords = (G.nodes[b_node]['x'], G.nodes[b_node]['y'])

        # Create LineString
        line = LineString([a_coords, b_coords])
        geometries.append(line)

    # Create GeoDataFrame
    edges_gdf = gpd.GeoDataFrame(edges_df, geometry=geometries)

    return edges_gdf, nodes_df


def read_od_data(od_file):
    """
    Read OD matrix from TNTP format file.
    Input: od_file (str).
    Output: DataFrame with OD pairs and demand.
    """
    f = open(od_file, 'r')
    all_rows = f.read()
    blocks = all_rows.split('Origin')[1:]
    matrix = {}
    for k in range(len(blocks)):
        orig = blocks[k].split('\n')
        dests = orig[1:]
        origs = int(orig[0])

        d = [eval('{' + a.replace(';', ',').replace(' ', '') + '}') for a in dests]
        destinations = {}
        for i in d:
            destinations = {**destinations, **i}
        matrix[origs] = destinations

    zones = max(matrix.keys())
    od_data = []
    for i in range(zones):
        for j in range(zones):
            demand = matrix.get(i + 1, {}).get(j + 1, 0)
            if demand > 0:
                od_data.append([i + 1, j + 1, demand])

    return pd.DataFrame(od_data, columns=['origin', 'destination', 'demand'])


def read_node_data(nodes_file):
    """
    Read node coordinates from TNTP format file.
    Input: nodes_file (str).
    Output: DataFrame with node id and coordinates.
    """

    # Load input files
    nodes_df = pd.read_csv(nodes_file, sep='\t')

    nodes_df.rename(columns={
        'Node': 'id',
        'X': 'x',
        'Y': 'y'}, inplace=True)

    return nodes_df


def read_network_data(net_file):
    """
    Reads network data from TNTP format file.
    Input: net_file (str).
    Output: DataFrame with network link attributes.
    """
    net_data = pd.read_csv(net_file, skiprows=8, sep='\t')
    # make sure all headers are lower case and without trailing spaces
    trimmed = [s.strip().lower() for s in net_data.columns]
    net_data.columns = trimmed
    # And drop the silly first and last columns
    net_data.drop(['~', ';'], axis=1, inplace=True)

    # make sure everything makes sense (otherwise some solvers throw errors)
    # Convert to float first to avoid dtype warnings
    net_data['free_flow_time'] = net_data['free_flow_time'].astype(float)
    net_data['capacity'] = net_data['capacity'].astype(float)
    net_data['length'] = net_data['length'].astype(float)
    net_data['power'] = net_data['power'].astype(float)

    net_data.loc[net_data['free_flow_time'] <= 0, 'free_flow_time'] = 1e-6
    net_data.loc[net_data['capacity'] <= 0, 'capacity'] = 1e-6
    net_data.loc[net_data['length'] <= 0, 'length'] = 1e-6
    net_data.loc[net_data['power'] <= 1, 'power'] = 4.0
    net_data['init_node'] = net_data['init_node'].astype(int)
    net_data['term_node'] = net_data['term_node'].astype(int)
    net_data['b'] = net_data['b'].astype(float)

    # Rename for consistency
    net_data = net_data.rename(columns={
        'init_node': 'a_node',
        'term_node': 'b_node',
        'free_flow_time': 'free_flow',
        'b': 'alpha',
        'power': 'beta'
    })

    return net_data


def prepare_network_geodata(nodes_df, links_df):
    """
    Prepare GeoDataFrame for network links using node coordinates.
    Inputs: nodes_df, links_df (DataFrames).
    Outputs: links_gdf (GeoDataFrame), nodes_df (DataFrame).
    """


    # Create a mapping from node id to coordinates
    node_coords = nodes_df.set_index('id')[['x', 'y']]

    # Create LineString geometries for each link
    geometries = [
        LineString([
            (node_coords.loc[link['a_node'], 'x'], node_coords.loc[link['a_node'], 'y']),
            (node_coords.loc[link['b_node'], 'x'], node_coords.loc[link['b_node'], 'y'])
        ])
        for _, link in links_df.iterrows()
    ]

    # Create GeoDataFrame
    links_gdf = gpd.GeoDataFrame(links_df, geometry=geometries)

    return links_gdf, nodes_df


def build_demand_matrix_for_aeq(od_df):
    """
    Convert OD DataFrame to AequilibraeMatrix.
    Input: od_df (DataFrame).
    Output: AequilibraeMatrix object.
    """
    zones = int(max(od_df['origin'].max(), od_df['destination'].max()))
    od_matrix = np.zeros((zones, zones))

    for _, row in od_df.iterrows():
        od_matrix[int(row['origin']) - 1, int(row['destination']) - 1] = row['demand']

    demand = AequilibraeMatrix()
    demand.create_empty(zones=zones, matrix_names=['matrix'], memory_only=True)
    demand.matrix['matrix'][:, :] = od_matrix
    demand.index[:] = np.arange(1, zones + 1).astype(int)
    demand.computational_view(["matrix"])

    return demand


def build_network_graph_for_aeq(network_df, zones, mode='car', cost_field='free_flow', time_filed='free_flow'):
    """
    Build an AequilibraE Graph from network DataFrame.
    Inputs: network_df (DataFrame), zones (int).
    Output: Graph object.
    """
    # Add required columns exactly as in original
    network_df = network_df.copy()
    network_df['direction'] = 1
    network_df.index = list(range(len(network_df)))
    network_df["link_id"] = network_df.reset_index().index + 1
    network_df = network_df.astype({"a_node": "int64", "b_node": "int64", "direction": "int64", "link_id": "int64"})

    # Build graph
    g = Graph()
    g.cost = network_df[cost_field].values
    if mode == 'car':
        g.capacity = network_df['capacity'].values
    if 'free_flow' in network_df.columns:
        g.free_flow = network_df['free_flow'].values
    elif 'free_flow_time' in network_df.columns:
        g.free_flow = network_df['free_flow_time'].values
    else:
        g.free_flow = network_df[time_filed].values

    # prep graph - EXACT order matters
    g.network = network_df
    g.network_ok = True
    g.status = 'OK'
    g.prepare_graph(np.arange(1, zones + 1).astype(int))
    g.set_graph(cost_field)
    g.cost = np.array(g.cost, copy=True)
    # g.set_skimming([cost_field])
    g.set_blocked_centroid_flows(False)
    g.network["id"] = g.network.link_id

    return g


def traffic_assignment_due_aeq(network_df, od_df, algorithm='msa', iterations=500, tolerance=1e-3, time_field='free_flow', cost_field='free_flow', mode='car', select_link_index=None):
    """
    Solve DUE traffic assignment using AequilibraE.
    Inputs: network_df, od_df, algorithm, iterations, tolerance.
    Output: dict with results DataFrame and metrics.
    """

    zones = int(max(od_df['origin'].max(), od_df['destination'].max()))

    # Prepare demand and graph
    demand = build_demand_matrix_for_aeq(od_df)

    # sort network_df by a_node and b_node to ensure consistent ordering
    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    graph = build_network_graph_for_aeq(network_df, zones, mode='car', cost_field=cost_field, time_filed=time_field)

    # Setup assignment
    traffic_class = TrafficClass(mode, graph, demand)
    assignment = TrafficAssignment()
    assignment.set_classes([traffic_class])
    assignment.set_vdf('BPR')
    assignment.set_vdf_parameters({"alpha": "alpha", "beta": "beta"})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field(cost_field)
    assignment.set_algorithm(algorithm)
    assignment.max_iter = iterations
    assignment.rgap_target = tolerance

    if select_link_index is not None:
        print(f"Configuring Select Link Analysis for link index: {select_link_index}")

        links_to_analyze = {'selected_link': [select_link_index]}

        traffic_class._selected_links = links_to_analyze

    results_df = network_df.copy()
    select_link_flows_df = None
    select_link_od = None
    start_time = time.time()

    # Execute
    try:
        assignment.execute()
        computation_time = time.time() - start_time

        # Extract results
        results = assignment.results()
        flows = results['matrix_ab'].values
        times = results['Congested_Time_AB'].values
        total_travel_time = np.sum(flows * times)
        rgap_list = list(assignment.report()['rgap'])
        rgap = rgap_list[-1] if rgap_list else np.inf
        results_df['flow'] = flows
        results_df['time'] = times

        # If you want, it is possible to access the convergence report
        convergence_report = pd.DataFrame(assignment.assignment.convergence_report)

        # success
        print(f"Assignment completed successfully in {computation_time:.2f} seconds.")
        # print algorithm, iterations, tolerance, total_travel_time, rgap
        print("Assignment Results:")
        print(f"Algorithm: {algorithm} \nIterations: {iterations}\nTolerance: {tolerance} "
              f"\nTotal Travel Time: {total_travel_time:.2f} \nRGAP: {rgap:.6f}\n")

        # Initialize select link results
        select_link_flows_df = None
        select_link_od = None

        # Process select link results if analysis was performed
        if select_link_index is not None:
            print("Processing select link results...")

            # Get select link flows (link-level results)
            select_link_flows_df = assignment.select_link_flows()

            # Get select link OD matrix
            sl_od_matrix = traffic_class.results.select_link_od
            select_link_od = sl_od_matrix.get_matrix("selected_link")[:, :, 0]  # Get 2D array

    except Exception as e:
        print(f"Error in assignment: {e}")
        total_travel_time = np.inf
        rgap = np.inf
        flows = np.zeros(len(network_df))
        times = np.zeros(len(network_df))
        computation_time = 0
        results_df['flow'] = flows
        results_df['time'] = times
        convergence_report = None

    return {
        'network': results_df,
        'total_travel_time': total_travel_time,
        'computation_time': computation_time,
        'rgap': rgap,
        'flow': flows,
        'time': times,
        'convergence': convergence_report,
        'select_link_flows': select_link_flows_df,
        'select_link_od': select_link_od
    }


def traffic_assignment_stochastic_aeq(
        network_df,
        od_df,
        time_field='free_flow',
        cost_field='free_flow',
        mode='bike',
        algorithm='bfsle',
        max_routes=5,
        max_depth=100,
        max_misses=100,
        beta=1.0,
        cutoff_prob=0.0,
        penalty=1.0,
        seed=0,
        cores=1,
        select_link_index=None
):
    """
    Solve stochastic (uncongested) traffic assignment using AequilibraE RouteChoice.

    This performs path-based stochastic assignment using route choice sets generated
    by either BFSLE (Breadth-First Search with Link Elimination) or Link Penalization
    algorithms, followed by Path Size Logit (PSL) route choice modeling.

    Args:
        network_df (pd.DataFrame): Network links with columns: a_node, b_node,
            free_flow (travel time), capacity, and geometry
        od_df (pd.DataFrame): Origin-destination demand with columns: origin,
            destination, demand
        algorithm (str): Route choice algorithm - 'bfsle' or 'link-penalisation'/'lp'
            Defaults to 'bfsle'
        max_routes (int): Maximum number of routes to generate per OD pair.
            Defaults to 5
        max_depth (int): For BFSLE - max graph height; For LP - max iterations.
            Defaults to 10
        max_misses (int): Maximum duplicate routes before stopping. Defaults to 100
        beta (float): PSL beta parameter for route choice sensitivity. Defaults to 1.0
        cutoff_prob (float): Probability cutoff for excluding routes (0-1).
            Defaults to 0.0 (include all)
        penalty (float): Link penalty factor for BFSLE. Set to 1.0 to disable.
            Defaults to 1.0
        seed (int): Random seed for BFSLE. Defaults to 0
        cores (int): Number of CPU cores to use. Defaults to 1
        select_link_index (int, optional): Link index for select link analysis.
            Defaults to None

    Returns:
        dict: Dictionary containing:
            - 'network': DataFrame with flow results
            - 'total_travel_time': Total network travel time
            - 'computation_time': Time taken for assignment
            - 'flows': Array of link flows
            - 'times': Array of link travel times (free-flow for stochastic)
            - 'route_choice_results': DataFrame with route choice details
            - 'select_link_flows': DataFrame with select link flows (if enabled)
            - 'select_link_od': Sparse matrix with select link OD flows (if enabled)
    """

    zones = int(max(od_df['origin'].max(), od_df['destination'].max()))

    # Prepare demand and graph
    demand = build_demand_matrix_for_aeq(od_df)
    # sort network_df by a_node and b_node to ensure consistent ordering
    network_df = network_df.sort_values(by=['a_node', 'b_node']).reset_index(drop=True)
    graph = build_network_graph_for_aeq(network_df, zones, mode=mode, cost_field=cost_field, time_filed=time_field)

    # Initialize RouteChoice
    route_choice = RouteChoice(graph)

    # Set number of cores
    route_choice.set_cores(cores)

    # Set choice set generation algorithm and parameters
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

    # Add demand matrix
    route_choice.add_demand(demand)

    # Prepare for execution (None means use all OD pairs with non-zero demand)
    route_choice.prepare(nodes=None)

    if select_link_index is not None:
        print(f"Configuring Select Link Analysis for link index: {select_link_index}")

        # RouteChoice requires links as tuples of (link_id, direction)
        # Since all links have direction 1, we specify it explicitly
        links_to_analyze = {'selected_link': [[(int(select_link_index), 1)]]}

        route_choice.set_select_links(links_to_analyze)

    results_df = network_df.copy()
    start_time = time.time()

    # Execute route choice and assignment
    try:
        print(f"Starting stochastic assignment with {algorithm} algorithm...")
        route_choice.execute(perform_assignment=True)
        computation_time = time.time() - start_time

        # Get link loading results
        load_results = route_choice.get_load_results()

        flows = load_results['matrix_ab'].values
        times = network_df[time_field].values
        costs = network_df[cost_field].values

        # Calculate total travel time
        total_travel_time = np.sum(flows * times)
        total_travel_cost = np.sum(flows * costs)

        # Get route choice results (paths and probabilities)
        route_choice_results = route_choice.get_results()

        # Add results to network dataframe
        results_df['travel_time'] = times
        results_df['travel_cost'] = costs
        results_df['flow'] = flows

        # Success message
        print(f"Assignment completed successfully in {computation_time:.2f} seconds.")
        print("Assignment Results:")
        print(f"Algorithm: {algorithm}")
        print(f"Max Routes: {max_routes}")
        print(f"Total Travel Time: {total_travel_time:.2f}\n")
        print(f"Total Travel Cost: {total_travel_cost:.2f}\n")


        # Initialize select link results
        select_link_flows_df = None
        select_link_od = None

        # Process select link results if analysis was performed
        if select_link_index is not None:
            print("Processing select link results...")

            try:
                # Get select link flows (link-level results)
                select_link_flows_df = route_choice.get_select_link_loading_results()

                # Get select link OD matrix (returns dict of sparse matrices)
                sl_od_dict = route_choice.get_select_link_od_matrix_results()

                # Extract the matrix for our selected link set
                if 'selected_link' in sl_od_dict:
                    demand_name = list(sl_od_dict['selected_link'].keys())[0]
                    # Convert Aequilibrae COO to SciPy COO using to_scipy()
                    aeq_coo = sl_od_dict['selected_link'][demand_name]
                    scipy_coo = aeq_coo.to_scipy()
                    select_link_od = scipy_coo.toarray()

                    # Summary
                    sl_total_flow = select_link_od.sum()
                    actual_link_flow = flows[select_link_index]
                    od_pairs_count = np.count_nonzero(select_link_od)

                    print(f"\nSelect Link Summary:")
                    print(f"  Link Index: {select_link_index}")
                    print(f"  OD pairs using link: {od_pairs_count}")
                    print(f"  Total flow from OD matrix: {sl_total_flow:.2f}")
                    print(f"  Actual link flow: {actual_link_flow:.2f}")
                else:
                    print("Warning: Select link results not found in output")

            except Exception as e:
                print(f"Warning: Could not extract select link results: {e}")

    except Exception as e:
        print(f"Error in assignment: {e}")
        import traceback
        traceback.print_exc()

        total_travel_time = np.inf
        total_travel_cost = np.inf
        computation_time = 0
        flows = np.zeros(len(network_df))
        times = network_df[time_field].values
        costs = network_df[cost_field].values
        route_choice_results = None
        results_df['flow'] = flows
        results_df['travel_time'] = times
        results_df['travel_cost'] = costs

    return {
        'network': results_df,
        'total_travel_time': total_travel_time,
        'total_travel_cost': total_travel_cost,
        'computation_time': computation_time,
        'flow': flows,
        'time': times,
        'cost': costs,
        'route_choice_results': route_choice_results,
        'select_link_flows': select_link_flows_df,
        'select_link_od': select_link_od,
        'load_results': load_results if 'load_results' in locals() else None  # For debugging
    }


def plot_network(edges_gdf, nodes_df,
                 width_column='capacity',
                 edge_color=None,
                 color_numerator=None,
                 color_denominator=None,
                 base_width=0.5,
                 width_scale=0.1,
                 cmap='RdYlGn_r',
                 show_nodes=True,
                 node_size=20,
                 node_color='grey',
                 a_node_col='a_node', b_node_col='b_node',
                 node_id_col='id', node_x_col='x', node_y_col='y',
                 figsize=(16, 10)):
    """
    Plot road network with edge widths and colors based on attributes.
    Inputs: edges_gdf, nodes_df, width/color options.
    Outputs: fig, ax (matplotlib objects).
    """

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Calculate coordinate scale automatically
    x_range = nodes_df[node_x_col].max() - nodes_df[node_x_col].min()
    y_range = nodes_df[node_y_col].max() - nodes_df[node_y_col].min()
    coord_scale = min(x_range, y_range) / 100  # Base scale on 1/100th of the coordinate range

    # Apply scaling to width parameters
    scaled_base_width = coord_scale * base_width
    scaled_width_scale = coord_scale * width_scale * 0.0001  # Additional scaling for typical data ranges

    # Convert nodes to dictionary for quick lookup
    node_coords = {row[node_id_col]: (row[node_x_col], row[node_y_col])
                   for _, row in nodes_df.iterrows()}

    # Get width values
    if width_column not in edges_gdf.columns:
        print(f"Warning: Column '{width_column}' not found. Using uniform width.")
        width_values = np.ones(len(edges_gdf))
    else:
        width_values = edges_gdf[width_column].values

    # Determine color values
    color_by_data = False
    color_label = ""

    # If edge_color is None, use width_column for coloring
    if edge_color is None:
        edge_color = width_column

    if edge_color == 'ratio' and color_numerator and color_denominator:
        # Calculate ratio for coloring
        if color_numerator in edges_gdf.columns and color_denominator in edges_gdf.columns:
            numerator_values = edges_gdf[color_numerator].values.astype(float)
            denominator_values = edges_gdf[color_denominator].values.astype(float)
            # Avoid division by zero
            color_values = np.divide(numerator_values, denominator_values,
                                     out=np.zeros_like(numerator_values, dtype=float),
                                     where=denominator_values != 0)
            color_by_data = True
            color_label = f"{color_numerator}/{color_denominator}"
        else:
            print(f"Warning: Columns '{color_numerator}' or '{color_denominator}' not found. Using uniform color.")
            color_values = None
            color_label = ""
    elif edge_color in edges_gdf.columns:
        color_values = edges_gdf[edge_color].values
        color_by_data = True
        color_label = edge_color
    else:
        # edge_color is a color string
        color_values = None
        color_label = ""

    # Prepare colormap if coloring by data
    if color_by_data and color_values is not None:
        min_val = np.min(color_values) if len(color_values) > 0 else 0
        max_val = np.max(color_values) if len(color_values) > 0 else 1
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

    # Process each edge
    edge_polygons = []
    edge_colors = []

    for idx, edge in edges_gdf.iterrows():
        # Get node coordinates
        a_coords = node_coords.get(edge[a_node_col])
        b_coords = node_coords.get(edge[b_node_col])

        if a_coords is None or b_coords is None:
            continue

        # Calculate edge width based on width_column value
        edge_width_value = width_values[idx] if idx < len(width_values) else 1
        edge_width = scaled_base_width + (edge_width_value * scaled_width_scale)

        # Determine the line coordinates
        if hasattr(edge, 'geometry') and edge.geometry is not None and not edge.geometry.is_empty:
            # Use the actual geometry if available
            line = edge.geometry
            coords = list(line.coords)

            # Ensure the geometry starts at a_node and ends at b_node
            first_point = coords[0]
            last_point = coords[-1]

            dist_to_a_first = np.sqrt((first_point[0] - a_coords[0]) ** 2 + (first_point[1] - a_coords[1]) ** 2)
            dist_to_b_first = np.sqrt((first_point[0] - b_coords[0]) ** 2 + (first_point[1] - b_coords[1]) ** 2)

            if dist_to_b_first < dist_to_a_first:
                # Geometry is reversed, flip it
                coords = coords[::-1]
        else:
            # Use straight line between nodes if no geometry
            coords = [a_coords, b_coords]

        # Create offset polygon for the edge
        edge_polygon = create_offset_polygon(coords, edge_width)

        if edge_polygon:
            edge_polygons.append(edge_polygon)

            # Determine color
            if color_by_data and color_values is not None:
                edge_colors.append(sm.to_rgba(color_values[idx]))
            else:
                edge_colors.append(edge_color)

    # Plot all edge polygons
    for polygon, color in zip(edge_polygons, edge_colors):
        x, y = polygon.exterior.xy
        ax.fill(x, y, color=color, alpha=0.7, edgecolor='gray', linewidth=0.5)

    # Plot nodes if requested
    if show_nodes:
        node_x = nodes_df[node_x_col].values
        node_y = nodes_df[node_y_col].values
        ax.scatter(node_x, node_y, s=node_size, c=node_color, zorder=5)

    # Add colorbar if coloring by data
    if color_by_data and color_values is not None:
        cbar = plt.colorbar(sm, ax=ax, label=color_label.replace('_', ' ').title())

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Road Network Visualization (Width Based on {width_column})')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    if len(nodes_df) > 0:
        x_min, x_max = nodes_df[node_x_col].min(), nodes_df[node_x_col].max()
        y_min, y_max = nodes_df[node_y_col].min(), nodes_df[node_y_col].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    plt.title(f'Width by {width_column}')
    plt.axis('off')  # Hide the axis for a cleaner look
    plt.tight_layout()
    plt.show()

    return fig, ax


def create_offset_polygon(coords, width):
    """
    Create a polygon offset to the right of a line.
    Inputs: coords (list of tuples), width (float).
    Output: Polygon object or None.
    """
    if len(coords) < 2:
        return None

    # Create offset points
    offset_points_right = []
    offset_points_left = []

    for i in range(len(coords)):
        if i == 0:
            # First point - use direction to next point
            dx = coords[1][0] - coords[0][0]
            dy = coords[1][1] - coords[0][1]
        elif i == len(coords) - 1:
            # Last point - use direction from previous point
            dx = coords[-1][0] - coords[-2][0]
            dy = coords[-1][1] - coords[-2][1]
        else:
            # Middle points - use average direction
            dx1 = coords[i][0] - coords[i - 1][0]
            dy1 = coords[i][1] - coords[i - 1][1]
            dx2 = coords[i + 1][0] - coords[i][0]
            dy2 = coords[i + 1][1] - coords[i][1]

            # Normalize each segment direction
            len1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
            len2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

            if len1 > 0:
                dx1 /= len1
                dy1 /= len1
            if len2 > 0:
                dx2 /= len2
                dy2 /= len2

            # Average the normalized directions
            dx = (dx1 + dx2) / 2
            dy = (dy1 + dy2) / 2

        # Normalize direction vector
        length = np.sqrt(dx ** 2 + dy ** 2)
        if length > 0:
            dx /= length
            dy /= length
        else:
            dx, dy = 1, 0

        # Calculate perpendicular vector (rotated 90 degrees to the right)
        perp_x = dy
        perp_y = -dx

        # Create offset points
        # The edge extends only to the right side
        center_x, center_y = coords[i]
        right_x = center_x + perp_x * width
        right_y = center_y + perp_y * width

        offset_points_right.append((right_x, right_y))
        offset_points_left.append((center_x, center_y))

    # Create polygon by combining the offset points
    # Go along the right side, then back along the center line
    polygon_coords = offset_points_right + offset_points_left[::-1]

    try:
        polygon = Polygon(polygon_coords)
        if polygon.is_valid:
            return polygon
        else:
            # Try to fix invalid polygons
            polygon = polygon.buffer(0)
            return polygon if polygon.is_valid else None
    except:
        return None


def plot_vc_ratio(links_results, bins):
    """Plot V/C ratio histogram with auto-generated green-yellow-red colors"""
    counts, edges = np.histogram(links_results['ratio'], bins=bins)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#2ecc71", "#f1c40f", "#e74c3c"])
    colors = [cmap(i / max(1, bins - 1)) for i in range(bins)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(edges[:-1], counts, width=edges[1]-edges[0], color=colors,
           edgecolor='black', alpha=0.8, align='edge')

    for pos, cnt in zip(edges[:-1], counts):
        if cnt > 0:
            ax.text(pos + (edges[1]-edges[0])/2, cnt, str(int(cnt)),
                    ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Volume/Capacity Ratio', fontsize=12)
    ax.set_ylabel('Number of Links', fontsize=12)
    ax.set_title('Link Congestion Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.6, linestyle='--')
    plt.tight_layout()
    plt.show()



