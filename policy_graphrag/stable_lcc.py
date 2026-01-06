# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module for producing a stable largest connected component, i.e. same input graph == same output lcc."""

from collections import defaultdict
from typing import Any, cast

import networkx as nx
import logging
from graspologic.utils import largest_connected_component
from graspologic.partition import hierarchical_leiden


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    graph = graph.copy()
    graph = cast("nx.Graph", largest_connected_component(graph))
    return _stabilize_graph(graph)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    sorted_nodes = graph.nodes(data=True)
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    fixed_graph.add_nodes_from(sorted_nodes)
    edges = list(graph.edges(data=True))

    # If the graph is undirected, we create the edges in a stable way, so we get the same results
    # for example:
    # A -> B
    # in graph theory is the same as
    # B -> A
    # in an undirected graph
    # however, this can lead to downstream issues because sometimes
    # consumers read graph.nodes() which ends up being [A, B] and sometimes it's [B, A]
    # but they base some of their logic on the order of the nodes, so the order ends up being important
    # so we sort the nodes in the edge in a stable way, so that we always get the same order
    if not graph.is_directed():

        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        edges = [_sort_source_target(edge) for edge in edges]

    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    fixed_graph.add_edges_from(edges)
    return fixed_graph


async def leiden_clustering(
    graph, max_graph_cluster_size, graph_cluster_seed, resolution=1.0
):
    graph = stable_largest_connected_component(graph)
    community_mapping = hierarchical_leiden(
        graph,
        max_cluster_size=max_graph_cluster_size,
        random_seed=graph_cluster_seed,
        resolution=resolution
    )

    node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
    __levels = defaultdict(set)
    for partition in community_mapping:
        level_key = partition.level
        cluster_id = partition.cluster
        node_communities[partition.node].append(
            {"level": level_key, "cluster": cluster_id}
        )
        __levels[level_key].add(cluster_id)
    node_communities = dict(node_communities)
    __levels = {k: len(v) for k, v in __levels.items()}
    logging.info(f"Each level has communities: {dict(__levels)}")
    return node_communities
