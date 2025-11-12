"""Utils function for bike package.

This module avoids importing heavy optional dependencies at top-level where possible
(e.g., osmnx) so the package can be imported in lightweight environments.
"""
from typing import Dict, List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def draw_graph(G: nx.Graph, title: str, ax=None):
    """Draw a graph with bike-friendly edges highlighted.

    Parameters
    - G: networkx Graph or MultiGraph with edge attribute 'bikes' (bool)
    - title: title for the plot
    - ax: optional matplotlib Axes
    """
    pos = nx.kamada_kawai_layout(G, weight="length")
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        created_fig = True

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color="lightblue", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="black", ax=ax)

    # Color edges based on the 'bikes' attribute
    edata = list(G.edges(data=True))
    edgelist = [(u, v) for u, v, _ in edata]
    edge_colors = ["tab:green" if d.get("bikes", d.get("bike", False)) else "lightgray" for _, _, d in edata]
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=2, alpha=0.6, edge_color=edge_colors, arrows=True, arrowsize=10, ax=ax)

    # Edge labels for lengths
    edge_labels = nx.get_edge_attributes(G, 'length')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)

    ax.set_title(title)
    ax.axis("off")
    if created_fig:
        plt.show()


def city_graph_generator(capacity: int, graph_number: int) -> nx.Graph:
    """Generate a small example city graph (used in the notebook)."""
    G = nx.Graph()
    if graph_number == 1:
        G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"])
        edges = [
            ("A", "B", 10, capacity, True),
            ("A", "C", 5, capacity, True),
            ("B", "D", 2, capacity, True),
            ("C", "D", 2, capacity, True),
            ("D", "E", 3, capacity, True),
            ("E", "F", 4, capacity, True),
            ("F", "G", 6, capacity, True),
            ("E", "G", 7, capacity, False),
            ("C", "E", 5, capacity, False),
            ("A", "F", 15, capacity, True),
            ("B", "H", 4, capacity, False),
            ("K", "A", 4, capacity, False),
            ("H", "I", 3, capacity, True),
            ("I", "J", 4, capacity, True),
            ("J", "K", 5, capacity, True),
            ("I", "K", 5, capacity, True),
        ]
    elif graph_number == 2:
        G.add_nodes_from(list("ABCDEFGHIJKLMNOP")[:11])
        edges = [
            ("A", "B", 5, capacity, True),
            ("A", "C", 9, capacity, True),
            ("B", "C", 3, capacity, False),
            ("C", "D", 2, capacity, True),
            ("B", "E", 6, capacity, True),
            ("E", "F", 5, capacity, False),
            ("D", "F", 7, capacity, True),
            ("F", "G", 4, capacity, True),
            ("G", "H", 6, capacity, True),
            ("H", "I", 5, capacity, True),
            ("I", "J", 4, capacity, False),
            ("J", "K", 5, capacity, True),
            ("G", "K", 7, capacity, False),
            ("D", "E", 14, capacity, False),
        ]
    elif graph_number == 3:
        G.add_nodes_from(list("ABCDEFGHIJK"))
        edges = [
            ("A", "C", 7, capacity, True),
            ("A", "B", 4, capacity, True),
            ("B", "D", 3, capacity, True),
            ("C", "D", 6, capacity, False),
            ("B", "E", 8, capacity, True),
            ("D", "F", 5, capacity, True),
            ("E", "G", 9, capacity, False),
            ("F", "G", 3, capacity, True),
            ("C", "H", 6, capacity, True),
            ("H", "I", 4, capacity, False),
            ("I", "J", 5, capacity, True),
            ("J", "K", 6, capacity, True),
            ("G", "K", 7, capacity, False),
        ]
    elif graph_number == 4:
        G.add_nodes_from(list("ABCDEFGHIJK"))
        edges = [
            ("A", "B", 6, capacity, True),
            ("B", "C", 4, capacity, True),
            ("A", "D", 7, capacity, False),
            ("C", "D", 5, capacity, False),
            ("C", "E", 8, capacity, False),
            ("D", "F", 6, capacity, True),
            ("E", "F", 7, capacity, True),
            ("E", "G", 4, capacity, True),
            ("G", "H", 6, capacity, True),
            ("H", "I", 3, capacity, False),
            ("I", "J", 5, capacity, True),
            ("J", "K", 6, capacity, False),
            ("G", "K", 8, capacity, True),
        ]
    elif graph_number == 5:
        G.add_nodes_from(list("ABCDEFGHIJK"))
        edges = [
            ("A", "C", 5, capacity, True),
            ("A", "B", 3, capacity, True),
            ("B", "D", 4, capacity, False),
            ("C", "D", 6, capacity, True),
            ("B", "E", 7, capacity, True),
            ("D", "F", 5, capacity, True),
            ("E", "G", 8, capacity, False),
            ("F", "G", 4, capacity, True),
            ("C", "H", 6, capacity, False),
            ("H", "I", 5, capacity, False),
            ("I", "J", 4, capacity, True),
            ("J", "K", 6, capacity, True),
            ("G", "K", 7, capacity, False),
            ("D", "J", 10, capacity, True),
            ("E", "F", 3, capacity, True),
            ("F", "J", 6, capacity, True),
        ]
    elif graph_number == 6:
        G.add_nodes_from(list("ABCDEFGHI"))
        edges = [
            ("A", "B", 4, capacity, False),
            ("B", "C", 4, capacity, False),
            ("A", "D", 6, capacity, True),
            ("B", "E", 6, capacity, False),
            ("C", "F", 6, capacity, True),
            ("E", "D", 4, capacity, False),
            ("E", "F", 4, capacity, True),
            ("D", "G", 6, capacity, True),
            ("D", "H", 7, capacity, False),
            ("H", "E", 6, capacity, False),
            ("G", "H", 4, capacity, False),
            ("F", "E", 4, capacity, True),
            ("F", "I", 6, capacity, True),
            ("I", "H", 4, capacity, True),
        ]
    else:
        edges = []

    G.add_edges_from([(u, v, {"length": w, "capacity": c, "flow": 0, "bikes": b}) for u, v, w, c, b in edges])
    return G


def import_osm_graph(place_name: str, network_type: str = 'bike', simplify: bool = True) -> nx.Graph:
    """Import OSM graph using osmnx.

    Note: osmnx is imported lazily here so importing the package doesn't require osmnx.
    """
    try:
        import osmnx as ox
    except Exception as e:
        raise ImportError("osmnx is required for import_osm_graph") from e

    G = ox.graph_from_place(place_name, network_type=network_type, simplify=simplify)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    # Mark cycleways as bike-friendly
    gdf_edges['bikes'] = False
    if 'highway' in gdf_edges.columns:
        gdf_edges.loc[gdf_edges['highway'] == 'cycleway', 'bikes'] = True
    # Rebuild graph and return undirected
    G2 = ox.graph_from_gdfs(gdf_nodes, gdf_edges).to_undirected()
    return G2