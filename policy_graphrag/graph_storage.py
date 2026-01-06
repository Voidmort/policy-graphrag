import json
import os
import logging
import networkx as nx
import pandas as pd

from collections import defaultdict
from typing import Union

from .stable_lcc import leiden_clustering


class GraphStorage:
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            if "gexf" in file_name:
                return nx.read_gexf(file_name)
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logging.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        for node, attr in graph.nodes(data=True):
            for key, value in attr.items():
                if isinstance(value, list):
                    attr[key] = json.dumps(value)
        nx.write_graphml(graph, file_name)

    @staticmethod
    def write_nx_graph_to_gexf(graph: nx.Graph, file_name):
        logging.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        for node, attr in graph.nodes(data=True):
            for key, value in attr.items():
                if isinstance(value, list):
                    attr[key] = json.dumps(value)
                if value is None:
                    if key == "clusters":
                        attr[key] = json.dumps([])
        nx.write_gexf(graph, file_name)

    @staticmethod
    def nx_to_parquet(graph: nx.Graph, name: str) -> None:
        nodes = []
        for id, node in graph.nodes(data=True):
            node["node_id"] = id
            nodes.append(node)
        edges = []
        for source_node_id, target_node_id, edge in graph.edges(data=True):
            edge["source_node_id"] = source_node_id
            edge["target_node_id"] = target_node_id
            edges.append(edge)

        df_nodes = pd.DataFrame(nodes)
        df_edges = pd.DataFrame(edges)

        df_nodes["clusters"] = df_nodes["clusters"].fillna(value="[]")
        df_nodes["text_unit_ids"] = df_nodes["text_unit_ids"].fillna(value="[]")
        df_nodes["name"] = df_nodes["name"].astype("string")
        df_nodes["entity_type"] = df_nodes["entity_type"].astype("string")
        df_nodes["description"] = df_nodes["description"].astype("string")
        df_nodes["node_id"] = df_nodes["node_id"].astype("string")
        df_nodes["text_unit_ids"] = df_nodes["text_unit_ids"].astype("string")
        df_nodes["clusters"] = df_nodes["clusters"].astype("string")

        df_edges["text_unit_ids"] = df_edges["text_unit_ids"].fillna(value="[]")
        df_edges["source"] = df_edges["source"].astype("string")
        df_edges["target"] = df_edges["target"].astype("string")
        df_edges["description"] = df_edges["description"].astype("string")
        df_edges["text_unit_ids"] = df_edges["text_unit_ids"].astype("string")
        df_edges["source_node_id"] = df_edges["source_node_id"].astype("string")
        df_edges["target_node_id"] = df_edges["target_node_id"].astype("string")

        df_nodes.to_parquet(f"{name}_nodes.parquet")
        df_edges.to_parquet(f"{name}_edges.parquet")

        return df_nodes, df_edges

    @staticmethod
    def nx_from_parquet(name: str) -> nx.Graph:
        if not (
            os.path.exists(f"{name}_nodes.parquet")
            and os.path.exists(f"{name}_edges.parquet")
        ):
            return None
        df_nodes = pd.read_parquet(f"{name}_nodes.parquet")
        df_edges = pd.read_parquet(f"{name}_edges.parquet")
        graph = nx.Graph()
        for index, row in df_nodes.iterrows():
            graph.add_node(row["node_id"], **row)
        for index, row in df_edges.iterrows():
            graph.add_edge(row["source_node_id"], row["target_node_id"], **row)
        return graph

    @staticmethod
    def filter_and_save_gexf(G, filename):
        """
        遍历 NetworkX 图，只保留指定的节点和边属性，然后保存为 GEXF 文件。

        参数:
        - G: 待处理的 NetworkX 图对象。
        - filename: 保存 GEXF 文件的路径。
        """
        # 创建一个新图来存储筛选后的数据
        ng = nx.DiGraph()

        # 遍历原始图的节点，并复制你需要的属性
        for node, data in G.nodes(data=True):
            new_data = {}
            if "name" in data:
                new_data["name"] = data["name"]
            if "entity_type" in data:
                new_data["entity_type"] = data["entity_type"]
            ng.add_node(node, **new_data)

        # 遍历原始图的边，并复制你需要的属性
        for u, v, data in G.edges(data=True):
            new_data = {}
            if "description" in data:
                new_data["description"] = data["description"]
            ng.add_edge(u, v, **new_data)

        # 将新图保存为 GEXF 格式
        nx.write_gexf(ng, filename)

    def __init__(
        self,
        working_dir: str,
        max_graph_cluster_size=100,
        use_lcc=True,
        graph_cluster_seed=1,
    ):

        self.max_graph_cluster_size = max_graph_cluster_size
        self.use_lcc = use_lcc
        self.graph_cluster_seed = graph_cluster_seed
        self._graphml_xml_file = os.path.join(working_dir, f"graph_storage.graphml")
        self._graphml_gexf_file = os.path.join(working_dir, f"graph_storage.gexf")
        self._graph_parquet_file = os.path.join(working_dir, f"graph_storage")

        # preloaded_graph = GraphStorage.load_nx_graph(self._graphml_gexf_file)
        preloaded_graph = GraphStorage.nx_from_parquet(self._graph_parquet_file)

        if preloaded_graph is not None:
            logging.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()

    async def index_done_callback(self):
        # GraphStorage.write_nx_graph(self._graph, self._graphml_xml_file)
        # GraphStorage.write_nx_graph_to_gexf(self._graph, self._graphml_gexf_file)
        GraphStorage.nx_to_parquet(self._graph, self._graph_parquet_file)
        GraphStorage.filter_and_save_gexf(self._graph, self._graphml_gexf_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self):
        node_communities = await leiden_clustering(
            self._graph, self.max_graph_cluster_size, self.graph_cluster_seed
        )
        self._cluster_data_to_subgraphs(node_communities)

    async def community_schema(self) -> dict:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                text_unit_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(data=True):
            if not node_data.get("clusters", None):
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                if isinstance(node_data["text_unit_ids"], str):
                    text_unit_ids = json.loads(
                        node_data["text_unit_ids"].replace("'", '"')
                    )
                else:
                    text_unit_ids = node_data["text_unit_ids"]
                results[cluster_key]["text_unit_ids"].update(text_unit_ids)
                max_num_ids = max(
                    max_num_ids, len(results[cluster_key]["text_unit_ids"])
                )

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(results[comm]["nodes"])
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["text_unit_ids"] = list(v["text_unit_ids"])
            v["occurrence"] = (
                len(v["text_unit_ids"]) / max_num_ids if max_num_ids else 0
            )
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)
