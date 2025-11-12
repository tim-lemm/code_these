"""Core implementation ported from `notebooks/indicators.ipynb`.

This module avoids importing heavy optional dependencies at top-level where possible
(e.g., osmnx) so the package can be imported in lightweight environments.
"""
from typing import Dict, List
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def extract_bike_friendly_subgraph(G: nx.Graph) -> nx.Graph:
    """Return an edge subgraph containing only bike-friendly edges.

    Works for Graph and MultiGraph.
    """
    if G.is_multigraph():
        bike_edges = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("bikes", d.get("bike", False))]
        return G.edge_subgraph(bike_edges).copy()
    else:
        bike_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("bikes", d.get("bike", False))]
        return G.edge_subgraph(bike_edges).copy()


def network_size(G: nx.Graph) -> float:
    """Total length of all edges in the graph."""
    if G.is_multigraph():
        return sum(data.get('length', 0) for _, _, _, data in G.edges(data=True, keys=True))
    else:
        return sum(data.get('length', 0) for _, _, data in G.edges(data=True))


def network_granularity(G: nx.Graph, bikes_only: bool = False) -> float:
    if bikes_only:
        sub = extract_bike_friendly_subgraph(G)
        lengths = [data.get('length', 0) for _, _, data in sub.edges(data=True)]
    else:
        lengths = [data.get('length', 0) for _, _, data in G.edges(data=True)]
    return float(np.mean(lengths)) if len(lengths) > 0 else 0.0


def network_coverage(G: nx.Graph) -> float:
    total_length = network_size(G)
    bike_length = network_size(extract_bike_friendly_subgraph(G))
    if total_length == 0:
        return 0.0
    return float(bike_length) / float(total_length)


def network_centrality_degree(G: nx.Graph) -> float:
    bike_sub = extract_bike_friendly_subgraph(G)
    if bike_sub.number_of_nodes() == 0:
        return 0.0
    centrality = nx.degree_centrality(bike_sub)
    vals = list(centrality.values())
    if np.mean(vals) == 0:
        return 0.0
    return float(max(vals) / np.mean(vals))


def network_centrality_betweeness(G: nx.Graph, weight: str = "length") -> float:
    bike_sub = extract_bike_friendly_subgraph(G)
    if bike_sub.number_of_nodes() == 0:
        return 0.0
    centrality = nx.betweenness_centrality(bike_sub, weight=weight, normalized=True)
    vals = list(centrality.values())
    if np.mean(vals) == 0:
        return 0.0
    return float(max(vals) / np.mean(vals))


def network_fragmentation_num(G: nx.Graph) -> int:
    bike_sub = extract_bike_friendly_subgraph(G)
    if bike_sub.number_of_nodes() == 0:
        return 0
    return nx.number_connected_components(bike_sub)


def network_fragmentation_sizes(G: nx.Graph) -> List[float]:
    bike_sub = extract_bike_friendly_subgraph(G)
    total = network_size(G)
    if total == 0:
        return []
    components = nx.connected_components(bike_sub)
    sizes = sorted([round(network_size(bike_sub.subgraph(c)) / total, 3) for c in components], reverse=True)
    return sizes


def network_connectivity(G: nx.Graph) -> float:
    bike_sub = extract_bike_friendly_subgraph(G)
    if bike_sub.number_of_nodes() == 0:
        return 0.0
    connectivity = []
    for u in bike_sub.nodes():
        gdeg = G.degree(u) if u in G else 0
        if gdeg > 0:
            connectivity.append(bike_sub.degree(u) / float(gdeg))
        else:
            connectivity.append(0.0)
    return float(np.mean(connectivity)) if connectivity else 0.0


def network_density_1(G: nx.Graph) -> float:
    bike_sub = extract_bike_friendly_subgraph(G)
    if bike_sub.number_of_edges() == 0:
        return 0.0
    mean_length = np.mean([data.get('length', 0) for _, _, data in bike_sub.edges(data=True)])
    total = network_size(G)
    return float(mean_length / total) if total > 0 else 0.0


def compute_all_indicators(G: nx.Graph) -> Dict[str, object]:
    return {
        "Size": network_size(G),
        "Granularity": network_granularity(G),
        "Coverage": network_coverage(G),
        "Degree Centrality": network_centrality_degree(G),
        "Betweenness Centrality": network_centrality_betweeness(G),
        "Number of Components": network_fragmentation_num(G),
        "Component Sizes": network_fragmentation_sizes(G),
        "Connectivity": network_connectivity(G),
        "Density 1": network_density_1(G),
    }
