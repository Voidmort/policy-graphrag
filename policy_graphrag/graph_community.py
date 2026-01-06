import asyncio
import logging

from typing import Dict, List
from tqdm import tqdm

from .data_model.community_report import CommunityReport, Finding
from .utils import extract_first_complete_json, list_of_list_to_csv
from .graph_storage import GraphStorage
from .prompts import *


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(key(data))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def pack_single_community_by_sub_communities(
    community,
    max_token_size: int,
    already_reports: dict[str, str],
) -> tuple[str, int]:
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])
        already_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_communities_describe,
        len(sub_communities_describe),
        set(already_nodes),
        set(already_edges),
    )


async def pack_single_community_describe(
    graph_storage: GraphStorage,
    community: CommunityReport,
    max_token_size: int = 12000,
    already_reports: dict[str] = {},
    force_to_use_sub_communities: bool = False,
) -> str:
    nodes_in_order = sorted(community.nodes)
    edges_in_order = sorted(community.edges, key=lambda x: x[0] + x[1])

    nodes_data = await asyncio.gather(
        *[graph_storage.get_node(n) for n in nodes_in_order]
    )
    edges_data = await asyncio.gather(
        *[graph_storage.get_edge(src, tgt) for src, tgt in edges_in_order]
    )
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            node_data.get("name", "UNKNOWN"),
            node_data.get("entity_type", "UNKNOWN"),
            node_data.get("description", "UNKNOWN"),
            await graph_storage.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            edge_data.get("source", ""),
            edge_data.get("target", ""),
            edge_data.get("description", "UNKNOWN"),
            await graph_storage.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # If context is exceed the limit and have sub-communities:
    report_describe = ""
    need_to_use_sub_communities = (
        truncated and len(community.sub_communities) and len(already_reports)
    )

    if need_to_use_sub_communities or force_to_use_sub_communities:
        logging.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        report_describe, report_size, contain_nodes, contain_edges = (
            pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # if report size is bigger than max_token_size, nodes and edges are []
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


async def generate_community_report(llm, graph_storage) -> List[CommunityReport]:
    communities_schema = await graph_storage.community_schema()
    community_keys, community_values = list(communities_schema.keys()), list(
        map(lambda x: CommunityReport(**x), communities_schema.values())
    )
    already_processed = 0

    async def _form_single_community_report(
        community: CommunityReport, already_reports: Dict[str, CommunityReport]
    ) -> CommunityReport:
        nonlocal already_processed

        describe = await pack_single_community_describe(
            graph_storage, community, already_reports=already_reports
        )

        response = await llm(
            [
                {"role": "system", "content": COMMUNITY_REPORT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMMUNITY_REPORT_USER_PROMPT.format(input_text=describe),
                },
            ]
        )
        data = extract_first_complete_json(response)
        already_processed += 1

        print(
            f"Generate community report processed 【{already_processed}】\r",
            end="",
            flush=True,
        )
        return data

    levels = sorted(set([c.level for c in community_values]), reverse=True)
    logging.info(f"Generating by levels: {levels}")
    community_data_dict = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v.level == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _form_single_community_report(c, community_data_dict)
                for c in this_level_community_values
            ]
        )
        for k, r, v in zip(
            this_level_community_keys,
            this_level_communities_reports,
            this_level_community_values,
        ):
            v.community_id = k
            v.title = r["title"]
            v.summary = r["summary"]
            v.rating = r["rating"]
            v.rating_explanation = r["rating_explanation"]
            v.findings = [Finding(**f) for f in r["findings"]]

            community_data_dict[k] = v
    return community_data_dict.values()


async def generate_community_report_by_df(llm, graph_storage) -> List[CommunityReport]:
    communities_schema = await graph_storage.community_schema()
    community_values = []
    already_processed = 0

    async def _form_single_community_report(
        community: CommunityReport, already_reports: Dict[str, CommunityReport]
    ) -> CommunityReport:
        nonlocal already_processed
        describe = await pack_single_community_describe(
            graph_storage, community, already_reports=already_reports
        )

        response = await llm(
            [
                {"role": "system", "content": COMMUNITY_REPORT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": COMMUNITY_REPORT_USER_PROMPT.format(input_text=describe),
                },
            ]
        )
        data = extract_first_complete_json(response)
        already_processed += 1

        print(
            f"Generate community report processed 【{already_processed}】\r",
            end="",
            flush=True,
        )
        return data

    for key, value in tqdm(communities_schema.items()):
        community = CommunityReport(**value)
        community.community_id = key

        cluster_data = _form_single_community_report(community, {}).result()
        if not cluster_data:
            continue

        community.title = cluster_data.get("title", "")
        community.summary = cluster_data.get("summary", "")
        community.rating = cluster_data.get("rating", 0)
        community.rating_explanation = cluster_data.get("rating_explanation", "")
        findings_dict = cluster_data.get("findings", [])
        try:
            community.findings = [Finding(**f) for f in findings_dict]
        except Exception as e:
            print(f"Error in community {key}, findings: {e}")
        community_values.append(community)
    return community_values
