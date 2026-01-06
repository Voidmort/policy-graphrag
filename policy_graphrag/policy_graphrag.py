import json
import logging
import pandas as pd
import re
import os

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm

from .graph_storage import GraphStorage
from .data_model.community_report import CommunityReport
from .data_model.entity import Entity
from .data_model.query import QueryParam
from .graph_community import generate_community_report
from .utils import (
    compute_mdhash_id,
    handle_single_entity_extraction,
    handle_single_relationship_extraction,
    list_of_list_to_csv,
    normalize_text,
    policy_docs_parser_by_content,
    split_string_by_multi_markers,
    split_text_by_file,
    split_text_by_sentence,
    truncate_list_by_token_size,
)
from .embeddings import EmbeddingBase
from .llms import LLMProviderBase
from .vector_storage import VectorStorage
from .prompts import *
from .data_model.node import Node
from .data_model.edge import Edge
from .data_model.chunk import Chunk


logging.basicConfig(level=logging.INFO)


class PolicyGraphRAG:
    chunk_text: pd.DataFrame
    communities_report: pd.DataFrame
    summary_max_tokens: int = 200
    entity_extract_max_gleaning: int = 0

    llm_provider: LLMProviderBase
    embed_provider: EmbeddingBase

    def __init__(
        self,
        working_dir: str,
        llm_provider: LLMProviderBase,
        embed_provider: EmbeddingBase,
    ):
        self.working_dir = working_dir
        self.naive_working_dir = f"{working_dir}/naive"
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.naive_working_dir, exist_ok=True)
        self.llm_provider = llm_provider
        self.embed_provider = embed_provider
        self.graph_storage = GraphStorage(working_dir)
        self.vector_storage = VectorStorage(self.embed_provider, self.working_dir)
        self.naive_vector_storage = VectorStorage(
            self.embed_provider, self.naive_working_dir
        )
        self.load_full_docs()
        self.segment_length = 1000  # qwq-plus max：98,304
        self.overlap_length = 50
        self.llm_max_token_size = 8192

    async def query(self, query: str, query_param: QueryParam):
        if query_param.mode == "graph":
            async for item in self.query_graph(query, query_param):
                yield item
        elif query_param.mode == "naive":
            async for item in self.query_naive(query, query_param):
                yield item
        else:
            raise ValueError(f"Unsupported mode: {query_param.mode}")

    async def query_graph(self, query: str, query_param: QueryParam):
        context, context_report = await self._build_graph_rag_query_context(
            query, query_param
        )
        yield {"context": context}
        if query_param.only_need_context:
            yield {"context_report": context_report}
            return
        if context_report is None:
            yield {"llm_response": FAIL_RESPONSE_PROMPT}
            return

        yield {"context_report": context_report}

        sys_prompt = GRAPH_RAG_RESPONSE_PROMPT.format(context_data=context_report)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ]

        async for item in self.llm_provider.async_generate_response_stream(
            messages=messages
        ):
            yield {"llm_response": item}

    def load_full_docs(self):
        # 加载政策文档
        policy_doc_path = os.path.join(self.working_dir, "policy_docs.parquet")
        self.policy_docs = (
            pd.read_parquet(policy_doc_path)
            if os.path.exists(policy_doc_path)
            else pd.DataFrame(columns=["document_id", "name"])
        )

        chunk_text_path = os.path.join(self.working_dir, "chunk_text.parquet")
        self.chunk_text = (
            pd.read_parquet(chunk_text_path)
            if os.path.exists(chunk_text_path)
            else pd.DataFrame(columns=["id", "content", "document_id"])
        )

        logging.info(
            f"load chunk {len(self.chunk_text)} text: \n\n {self.chunk_text.head()}"
        )

        communities_report_path = os.path.join(
            self.working_dir, "communities_report.parquet"
        )
        if os.path.exists(communities_report_path):
            self.communities_report = pd.read_parquet(communities_report_path)
        else:
            self.communities_report = pd.DataFrame(
                columns=[
                    "community_id",
                    "title",
                    "level",
                    "edges",
                    "nodes",
                    "text_unit_ids",
                    "occurrence",
                    "summary",
                    "rating",
                    "rating_explanation",
                    "findings",
                    "sub_communities",
                ]
            )
        self.communities_report["community_id"] = self.communities_report[
            "community_id"
        ].astype(str)
        logging.info(
            f"load {len(self.communities_report)} communities_report: \n\n {self.communities_report.head()}"
        )

    async def save_full_docs(
        self, chunks_data: List[Dict], new_policy_docs_df: pd.DataFrame
    ):
        # 合并切片
        new_chunks = pd.DataFrame(chunks_data)
        self.chunk_text = (
            pd.concat([self.chunk_text, new_chunks])
            .drop_duplicates(subset="id", keep="last")
            .reset_index(drop=True)
        )
        logging.info(f"Update chunks: \n\n {self.chunk_text.head()}")

        # 保存
        self.chunk_text.to_parquet(os.path.join(self.working_dir, "chunk_text.parquet"))

        self.policy_docs = pd.concat([self.policy_docs, new_policy_docs_df])
        self.policy_docs.to_parquet(
            os.path.join(self.working_dir, "policy_docs.parquet")
        )

        logging.info(f"Update graph storage start.")
        await self.graph_storage.index_done_callback()
        logging.info("Update graph done.")

    async def llm(self, messages, **kwargs) -> str:
        return await self.llm_provider.async_generate_response(
            messages=messages, **kwargs
        )

    async def embed(self, text: str) -> List[float]:
        return self.embed_provider.embed(text)

    async def index(
        self,
        file_paths: List[str] = [],
        policy_name: str = "",
        content: str = "",
        source_nodes: List[Node] = [],
        source_edges: List[Edge] = [],
        is_update_community: bool = False,
    ):
        if file_paths:
            for file_path in file_paths:
                await self._index_doc(file_path)

        if content:
            await self._index_doc_by_content(
                policy_name, content, source_nodes, source_edges
            )

        if is_update_community:
            await self.update_graph_community_report()

    async def _index_doc(self, doc_path: str):
        file_name = os.path.basename(doc_path).split(".")[0]

        with open(doc_path, "r", encoding="utf-8") as file:
            content = file.read()  # 读取文件内容

        await self._index_doc_by_content(file_name, content)

    async def _index_doc_by_content(
        self,
        policy_name,
        content: str,
        source_nodes: List[Node] = [],
        source_edges: List[Edge] = [],
    ):
        # 切片
        chunks_data = await split_text_by_sentence(
            policy_name,
            content,
            segment_length=self.segment_length,
            overlap_length=self.overlap_length,
        )

        chunks = [
            Chunk(**chunk)
            for chunk in chunks_data
            if not self.chunk_text["id"].isin([chunk.get("id")]).any()
        ]
        if not chunks:
            logging.info("No new chunks to index")
            return

        # 实体提取
        logging.info(f"Extracting entities from {len(chunks)} chunks")
        maybe_nodes, maybe_edges = await self.extract_entities(chunks)
        logging.info(f"Extracted {len(maybe_nodes)} nodes and {len(maybe_edges)} edges")

        logging.info("Merging nodes and edges")
        add_nodes = [
            await self.merge_nodes(key, value) for key, value in maybe_nodes.items()
        ]
        add_edges = [
            await self.merge_edges(key[0], key[1], value)
            for key, value in maybe_edges.items()
        ]
        logging.info(f"Merged into {len(add_nodes)} nodes and {len(add_edges)} edges")

        # 实体消歧
        logging.info("Disambiguating entities")
        add_nodes += source_nodes
        add_edges += source_edges
        add_nodes, add_edges = await self.disambiguate_entities(
            chunks, add_nodes, add_edges
        )

        logging.info("Updating graph")
        await self.update_graph_and_entities(add_nodes, add_edges)

        logging.info("Saving full documents and chunks")
        new_policy_docs_df = policy_docs_parser_by_content(policy_name, content)
        await self.save_full_docs(chunks_data, new_policy_docs_df)
        logging.info("Indexing completed")

    async def update_graph_community_report(self):
        logging.info("Generating community report")
        community_data = await generate_community_report(self.llm, self.graph_storage)
        logging.info(f"Generated {len(community_data)} community reports")

        # 合并社区报告
        new_communities = pd.DataFrame(map(lambda x: x.model_dump(), community_data))
        self.communities_report = (
            pd.concat([self.communities_report, new_communities])
            .drop_duplicates(subset="community_id", keep="last")
            .sort_values("community_id")
            .reset_index(drop=True)
        )
        self.communities_report.to_parquet(
            os.path.join(self.working_dir, "communities_report.parquet")
        )
        logging.info(
            f"Update communities report: \n\n {self.communities_report.head()}"
        )

    async def get_node(
        self, entity_name: Optional[str] = None, node_id: Optional[str] = None
    ) -> Optional[Node]:
        if not node_id:
            node_id = compute_mdhash_id(entity_name)
        node_data = await self.graph_storage.get_node(node_id)
        if node_data:
            if isinstance(node_data.get("text_unit_ids"), str):
                node_data["text_unit_ids"] = json.loads(
                    node_data["text_unit_ids"].replace("'", '"')
                )
            clusters = node_data.get("clusters", None)

            if isinstance(clusters, str):
                node_data["clusters"] = json.loads(node_data["clusters"])
            if not clusters:
                node_data["clusters"] = []
            return Node(**node_data)
        return None

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Optional[Edge]:
        edge_data = await self.graph_storage.get_edge(source_node_id, target_node_id)
        if edge_data:
            if isinstance(edge_data.get("text_unit_ids"), str):
                edge_data["text_unit_ids"] = json.loads(edge_data["text_unit_ids"])
            edge_data["rank"] = await self.graph_storage.edge_degree(
                source_node_id, target_node_id
            )
            return Edge(**edge_data)
        return None

    async def upsert_node(self, node: Node):
        exist_node = await self.get_node(node_id=node.id)
        if exist_node:
            to_update = False
            # 优先处理 UNKNOWN 类型覆盖逻辑
            if exist_node.entity_type == "UNKNOWN":
                exist_node.entity_type = node.entity_type
                to_update = True

            # 使用集合合并 text_unit_ids 并去重
            existing_ids = set(exist_node.text_unit_ids)
            new_ids = set(node.text_unit_ids)
            merged_ids = list(existing_ids.union(new_ids))

            # 检查是否需要更新
            if not to_update and merged_ids == exist_node.text_unit_ids:
                return

            exist_node.text_unit_ids = merged_ids
            update_data = {
                "name": exist_node.name,
                "entity_type": exist_node.entity_type,
                "description": exist_node.description,
                "text_unit_ids": json.dumps(exist_node.text_unit_ids),
            }
        else:
            update_data = {
                "name": node.name,
                "entity_type": node.entity_type,
                "description": node.description,
                "text_unit_ids": json.dumps(node.text_unit_ids),
            }

        await self.graph_storage.upsert_node(node.id, update_data)

    async def upsert_entity(self, nodes: List[Node]):
        self.vector_storage.upsert_entities(
            [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.entity_type,
                    "description": node.description,
                }
                for node in nodes
            ]
        )

    async def search_node(
        self, query: str, top_k: int = 10, threshold=0.8
    ) -> List[Node]:
        result = self.vector_storage.query_entities(
            query, top_k=top_k, threshold=threshold
        )
        nodes = []
        for entity in map(lambda x: Entity(**x), result):
            node = await self.get_node(entity_name=entity.name)
            if node:
                node.rank = await self.graph_storage.node_degree(node.id)
                nodes.append(node)
        return nodes

    async def upsert_edge(self, edge: Edge) -> int:
        # 确保节点存在
        for node_id, node_name in [
            (edge.source, edge.source),
            (edge.target, edge.target),
        ]:
            await self.upsert_node(
                Node(
                    name=node_name,
                    entity_type="UNKNOWN",
                    description=edge.description,
                    text_unit_ids=edge.text_unit_ids,
                )
            )
        edge_data = {
            "source": edge.source,
            "target": edge.target,
            "description": edge.description,
            "weight": edge.weight,
            "text_unit_ids": json.dumps(edge.text_unit_ids),
        }
        await self.graph_storage.upsert_edge(edge.source_id, edge.target_id, edge_data)

    async def extract_entities(self, chunks: List[Chunk]):
        context_base = dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
        )
        entity_extract_system_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
            **context_base
        )

        maybe_nodes: Dict[str, List[Node]] = defaultdict(list)
        maybe_edges: Dict[tuple, List[Edge]] = defaultdict(list)

        for chunk in tqdm(chunks, desc="Extracting entities from chunks"):
            entity_extract_user_prompt = ENTITY_EXTRACT_USER_PROMPT.format(
                policy_name=chunk.file_name, policy_text=chunk.content
            )
            continue_prompt = ENTITIES_CONTINUE_EXTRACTION_PROMPT.format(
                policy_name=chunk.file_name
            )

            final_result = await self.llm(
                [
                    {"role": "system", "content": entity_extract_system_prompt},
                    {"role": "user", "content": entity_extract_user_prompt},
                ]
            )

            for _ in tqdm(
                range(self.entity_extract_max_gleaning),
                desc="Gleaning entities from chunks",
            ):
                token_count = (
                    len(entity_extract_system_prompt)
                    + len(entity_extract_user_prompt)
                    + len(final_result)
                    + len(continue_prompt)
                )
                if token_count > self.llm_max_token_size:
                    print(f"Token count exceeded, stopping gleaning: {token_count}")
                    break
                glean_result = await self.llm(
                    [
                        {"role": "system", "content": entity_extract_system_prompt},
                        {"role": "user", "content": entity_extract_user_prompt},
                        {"role": "assistant", "content": final_result},
                        {"role": "user", "content": continue_prompt},
                    ]
                )

                if glean_result.strip() == "无":
                    break

                final_result += glean_result

            records = split_string_by_multi_markers(
                final_result,
                [
                    context_base["record_delimiter"],
                    context_base["completion_delimiter"],
                ],
            )
            for record in records:
                m = re.search(r"\((.*)\)", record)
                if not m:
                    continue
                record_attributes = split_string_by_multi_markers(
                    m.group(1), [context_base["tuple_delimiter"]]
                )
                if_entities = await handle_single_entity_extraction(record_attributes)
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(
                        Node(
                            name=if_entities["entity_name"],
                            description=if_entities["description"],
                            entity_type=if_entities["entity_type"],
                            text_unit_ids=[chunk.id],
                        )
                    )
                    continue
                if_relation = await handle_single_relationship_extraction(
                    record_attributes
                )
                if if_relation is not None:
                    key = tuple(sorted([if_relation["src_id"], if_relation["tgt_id"]]))
                    maybe_edges[key].append(
                        Edge(
                            source=if_relation["src_id"],
                            target=if_relation["tgt_id"],
                            description=if_relation["description"],
                            weight=if_relation["weight"],
                            text_unit_ids=[chunk.id],
                        )
                    )
        return maybe_nodes, maybe_edges

    async def disambiguate_entities(
        self,
        chunks: List[Chunk],
        nodes: List[Node],
        edges: List[Edge],
    ) -> Tuple[List[Node], List[Edge]]:
        """
        整合实体信息，补全图，并合并重复的节点和边。

        Args:
            chunks: 与图相关的文本块列表。
            nodes: 从文本中提取的节点列表。
            edges: 从文本中提取的边列表。

        Returns:
            一个元组，包含处理后的新节点列表和新边列表。
        """
        if not chunks:
            return [], []

        # 创建输入列表的副本，以避免副作用
        edges_copy = list(edges)

        # --- 预处理: 归一化所有实体名称，确保一致性 ---
        normalized_nodes = [
            Node(
                name=normalize_text(node.name),
                entity_type=normalize_text(node.entity_type),
                description=node.description,
                text_unit_ids=node.text_unit_ids,
            )
            for node in nodes
        ]
        normalized_edges = [
            Edge(
                source=normalize_text(edge.source),
                target=normalize_text(edge.target),
                description=edge.description,
                weight=edge.weight,
                rank=edge.rank,
                text_unit_ids=edge.text_unit_ids,
            )
            for edge in edges_copy
        ]

        # --- 步骤 1: 整合所有实体信息 ---
        consolidated_entities: Dict[str, Dict[str, Set[str]]] = defaultdict(
            lambda: {"types": set(), "descriptions": set()}
        )

        policy_name_entity = normalize_text(chunks[0].file_name)
        consolidated_entities[policy_name_entity]["types"].add("政策名称")
        consolidated_entities[policy_name_entity]["descriptions"].add(
            policy_name_entity
        )

        for node in normalized_nodes:
            consolidated_entities[node.name]["types"].add(node.entity_type)
            consolidated_entities[node.name]["descriptions"].add(node.description)

        # --- 步骤 2: 识别并补全图中缺失的实体和边 ---
        all_known_entity_names = set(consolidated_entities.keys())
        # 计算一次所有出现在边中的实体，并复用
        all_entities_in_edges = set(edge.source for edge in normalized_edges) | set(
            edge.target for edge in normalized_edges
        )

        # 2.1 找出在边中存在，但在节点列表中缺失的实体
        missing_entities_in_nodes = all_entities_in_edges - all_known_entity_names
        for missing_name in missing_entities_in_nodes:
            consolidated_entities[missing_name]["types"].add("UNKNOWN")
            consolidated_entities[missing_name]["descriptions"].add("")

        # 2.2 找出孤立节点并将其连接到主策略实体
        isolated_entities = all_known_entity_names - all_entities_in_edges
        for entity_name in isolated_entities:
            if entity_name != policy_name_entity:
                # 向副本中添加边
                normalized_edges.append(
                    Edge(
                        source=entity_name,
                        target=policy_name_entity,
                        description="",
                        weight=1.0,
                        text_unit_ids=[chunk.id for chunk in chunks],
                    )
                )

        # --- 步骤 3: 合并重复的边 ---
        edge_mapping: Dict[Tuple[str, str], Edge] = {}
        for edge in normalized_edges:
            key = (edge.source, edge.target)
            if key in edge_mapping:
                existing_edge = edge_mapping[key]
                existing_edge.weight += edge.weight
                existing_edge.text_unit_ids = list(
                    set(existing_edge.text_unit_ids) | set(edge.text_unit_ids)
                )
            else:
                edge_mapping[key] = edge

        # --- 步骤 4: 生成最终的节点和边列表 ---
        new_nodes = [
            Node(
                name=name,
                entity_type="|".join(sorted([x for x in list(data["types"]) if x])),
                description="|".join(
                    sorted([x for x in list(data["descriptions"]) if x])
                ),
                text_unit_ids=[chunk.id for chunk in chunks],
            )
            for name, data in consolidated_entities.items()
        ]

        new_edges = list(edge_mapping.values())

        return new_nodes, new_edges

    async def merge_nodes(self, entity_name: str, nodes: List[Node]) -> Node:
        already_node = await self.get_node(entity_name)
        already_entitiy_types = [already_node.entity_type] if already_node else []
        already_text_unit_ids = already_node.text_unit_ids if already_node else []
        already_description = [already_node.description] if already_node else []

        entity_type = sorted(
            Counter([dp.entity_type for dp in nodes] + already_entitiy_types).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp.description for dp in nodes] + already_description))
        )
        text_unit_ids = list(
            set(
                already_text_unit_ids
                + [id for node in nodes for id in node.text_unit_ids]
            )
        )
        if GRAPH_FIELD_SEP in description:
            description = await self.handle_entity_relation_summary(
                entity_name, description
            )
        return Node(
            name=entity_name,
            entity_type=entity_type,
            description=description,
            text_unit_ids=text_unit_ids,
        )

    async def merge_edges(
        self, src_id: str, tgt_id: str, edges_data: List[Edge]
    ) -> Edge:
        already_edge = (
            await self.get_edge(src_id, tgt_id)
            if await self.graph_storage.has_edge(src_id, tgt_id)
            else None
        )
        already_weights = [already_edge.weight] if already_edge else []
        already_text_unit_ids = already_edge.text_unit_ids if already_edge else []
        already_description = [already_edge.description] if already_edge else []

        weight = sum([edge.weight for edge in edges_data] + already_weights)
        description = GRAPH_FIELD_SEP.join(
            sorted(set([edge.description for edge in edges_data] + already_description))
        )
        text_unit_ids = list(
            set(
                already_text_unit_ids
                + [id for edge in edges_data for id in edge.text_unit_ids]
            )
        )
        if GRAPH_FIELD_SEP in description:
            description = await self.handle_entity_relation_summary(
                (src_id, tgt_id), description
            )
        return Edge(
            source=src_id,
            target=tgt_id,
            weight=weight,
            description=description,
            text_unit_ids=text_unit_ids,
        )

    async def update_graph_and_entities(self, nodes: List[Node], edges: List[Edge]):
        if not nodes and not edges:
            logging.warning("No nodes or edges to update")
            return

        for node in tqdm(nodes, desc="Updating nodes"):
            await self.upsert_node(node)
        await self.upsert_entity(nodes)
        for edge in tqdm(edges, desc="Updating edges"):
            await self.upsert_edge(edge)

        logging.info("Starting graph clustering...")
        await self.graph_storage.clustering()
        logging.info("Graph clustering completed.")

    async def handle_entity_relation_summary(
        self, entity_or_relation_name, description: str
    ) -> str:
        if len(description) < self.summary_max_tokens:
            return description
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=description.split(GRAPH_FIELD_SEP),
        )
        use_prompt = SUMMARIZE_ENTITY_DESCRIPTIONS_PROMPT.format(**context_base)
        logging.debug(f"Trigger summary: {entity_or_relation_name}")
        summary = await self.llm([{"role": "user", "content": use_prompt}])
        return summary

    async def _find_most_related_community_from_entities(
        self, nodes: List[Node], query_param: QueryParam
    ) -> List[CommunityReport]:
        related_communities = []
        for node in nodes:
            if getattr(node, "clusters", None):
                related_communities.extend(node.clusters)
        related_community_dup_keys = [
            str(dp.cluster)
            for dp in related_communities
            if dp.level <= query_param.level
        ]
        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        keys = list(related_community_keys_counts.keys())
        related_community_datas = self.communities_report[
            self.communities_report["community_id"].isin(keys)
        ]
        sorted_community_datas = related_community_datas.sort_values(
            by="rating", ascending=False
        )
        sorted_community_reports = [
            CommunityReport(**x) for x in sorted_community_datas.to_dict("records")
        ]
        use_community_reports = truncate_list_by_token_size(
            sorted_community_reports,
            key=lambda x: x.report_string,
            max_token_size=query_param.local_max_token_for_community_report,
        )
        return use_community_reports

    async def _find_most_related_text_unit_from_entities(
        self, nodes: List[Node], query_param: QueryParam
    ) -> List[Chunk]:
        text_units_ids = []
        all_one_hop_nodes = []
        all_one_hop_nodes_text_units_ids = []
        for node in nodes:
            text_units_ids.extend(node.text_unit_ids)
            edges = await self.graph_storage.get_node_edges(node.id)
            for edge in edges:
                one_hop_node = await self.get_node(node_id=edge[1])
                if one_hop_node:
                    all_one_hop_nodes.append(one_hop_node)
                    all_one_hop_nodes_text_units_ids.extend(one_hop_node.text_unit_ids)
        all_one_hop_nodes_text_units_relation_counter = Counter(
            all_one_hop_nodes_text_units_ids
        )
        chunk_text_df = self.chunk_text[
            self.chunk_text["id"].isin(text_units_ids)
        ].copy()
        chunk_text_df["count_relation"] = chunk_text_df["id"].map(
            all_one_hop_nodes_text_units_relation_counter
        )
        chunk_text_df = chunk_text_df.sort_values(
            by=["count_relation"], ascending=[False]
        )
        chunks = [Chunk(**x) for x in chunk_text_df.to_dict("records")]
        text_units_data = truncate_list_by_token_size(
            chunks,
            key=lambda x: x.content,
            max_token_size=query_param.local_max_token_for_text_unit,
        )
        return text_units_data

    async def _find_most_related_edges_from_entities(
        self, nodes: List[Node], query_param: QueryParam
    ) -> List[Edge]:
        all_edges = []
        seen = set()
        for node in nodes:
            edges = await self.graph_storage.get_node_edges(node.id)
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    edge_obj = await self.get_edge(sorted_edge[0], sorted_edge[1])
                    if edge_obj:
                        all_edges.append(edge_obj)
        all_edges = sorted(all_edges, key=lambda x: (x.rank, x.weight), reverse=True)
        all_edges = truncate_list_by_token_size(
            all_edges,
            key=lambda x: x.edge_string,
            max_token_size=query_param.local_max_token_for_local_context,
        )
        return all_edges

    async def _build_graph_rag_query_context(
        self, query: str, query_param: QueryParam
    ) -> str:
        nodes = await self.search_node(
            query, top_k=query_param.top_k, threshold=query_param.threshold
        )
        related_unities = await self._find_most_related_text_unit_from_entities(
            nodes, query_param=query_param
        )
        related_edges = await self._find_most_related_edges_from_entities(
            nodes, query_param=query_param
        )
        related_communities = await self._find_most_related_community_from_entities(
            nodes, query_param=query_param
        )

        if not (nodes or related_unities or related_edges or related_communities):
            return None, None

        entities_context = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(nodes):
            entities_context.append([i, n.name, n.entity_type, n.description, n.rank])
        entities_context = list_of_list_to_csv(entities_context)

        unities_context = [["id", "content"]]
        for i, unit in enumerate(related_unities):
            unities_context.append([i, unit.content])
        unities_context = list_of_list_to_csv(unities_context)

        edges_context = [["id", "source", "target", "description", "weight", "rank"]]
        for i, edge in enumerate(related_edges):
            edges_context.append(
                [i, edge.source, edge.target, edge.description, edge.weight, edge.rank]
            )
        edges_context = list_of_list_to_csv(edges_context)

        communities_context = [["id", "content"]]
        for i, community in enumerate(related_communities):
            communities_context.append([i, community.report_string])
        communities_context = list_of_list_to_csv(communities_context)

        context = {
            "nodes": nodes,
            "edges": related_edges,
            "unities": related_unities,
            "communities": related_communities,
        }

        return (
            context,
            f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{edges_context}
```
-----Sources-----
```csv
{unities_context}
```
""",
        )

    async def index_naive(
        self,
        file_paths: List[str] = [],
        policy_name: str = "",
        content: str = "",
        segment_length: int = 300,
        overlap_length: int = 30,
    ):
        all_chunks = []
        if file_paths:
            for doc_path in file_paths:
                chunks = await split_text_by_file(
                    doc_path, segment_length, overlap_length
                )
                all_chunks.extend(chunks)

        if content:
            chunks = await split_text_by_sentence(
                policy_name, content, segment_length, overlap_length
            )
            all_chunks.extend(chunks)

        await self.upsert_text_unit(chunks)

    async def upsert_text_unit(self, chunks: List[Dict[str, Any]]):
        self.naive_vector_storage.upsert_text_units(chunks)

    async def _build_naive_rag_query_context(
        self, query: str, query_param: QueryParam
    ) -> str:
        text_units = self.naive_vector_storage.query_units(
            query, top_k=query_param.top_k, threshold=query_param.threshold
        )
        text_units = [context["entity"]["content"] for context in text_units]
        return text_units

    async def query_naive(self, query: str, query_param: QueryParam):
        context = await self._build_naive_rag_query_context(query, query_param)
        yield {"context": context}
        if query_param.only_need_context:
            return
        if not context:
            yield {"llm_response": FAIL_RESPONSE_PROMPT}
            return
        sys_prompt = NAIVE_RAG_RESPONSE_PROMPT.format(context_data=context)
        async for item in self.llm_provider.async_generate_response_stream(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query},
            ]
        ):
            yield {"llm_response": item}

    def get_cites_policy(self, context):
        chunks_ids = set()
        for edge in context.get("edges", []):
            chunks_ids |= set(edge.text_unit_ids)
        for node in context.get("nodes", []):
            chunks_ids |= set(node.text_unit_ids)
        return list(
            set(self.chunk_text[self.chunk_text["id"].isin(chunks_ids)].file_name)
        )
