# 简单的graph可视化

import asyncio
import os
import streamlit as st

from stqdm import stqdm
from streamlit_agraph import agraph, Node, Edge, Config


from policy_graphrag.data_model.query import QueryParam
from policy_graphrag.embeddings import OpenaiEmbedding
from policy_graphrag.llms import OpenAIProvider
from policy_graphrag import PolicyGraphRAG

@st.cache_resource
def load_policy_graph_rag() -> PolicyGraphRAG:
    working_dir = os.getenv("working_dir")
    llm_api_key = os.getenv("llm_api_key")
    llm_base_url = os.getenv("llm_base_url")
    llm_model_name = os.getenv("llm_model_name")

    llm = OpenAIProvider(
        config={
            "api_key": llm_api_key,
            "base_url": llm_base_url,
            "model_name": llm_model_name,
            "temperature": 0.3,
        }
    )
    embedding_type = os.getenv("embedding_type")
    if embedding_type == "huggingface":
        from policy_graphrag.embeddings.hugging_face import HuggingFaceEmbedding

        embedding_model = os.getenv("embedding_model")
        device = os.getenv("device")
        embed = HuggingFaceEmbedding(
            config={
                "device": device,
                "embedding_model": embedding_model,
            }
        )
    else:
        embedding_api_key = os.getenv("embedding_api_key")
        embedding_base_url = os.getenv("embedding_base_url")
        embedding_model_name = os.getenv("embedding_model_name")
        embed = OpenaiEmbedding(
            config={
                "api_key": embedding_api_key,
                "base_url": embedding_base_url,
                "model_name": embedding_model_name,
            }
        )

    pgr = PolicyGraphRAG(
        working_dir=working_dir, llm_provider=llm, embed_provider=embed
    )
    return pgr


pgr = load_policy_graph_rag()


import hashlib
from typing import List, Tuple

pgr = load_policy_graph_rag()
st.set_page_config(layout="wide")

# 设置侧边栏选项
st.sidebar.title("设置")
# tod 设置 config
# 预定义的美观颜色调色板（可自行扩展）
COLOR_PALETTE = [
    "#FDD2B5",
    "#F48B94",
    "#F7A7A6",
    "#DBEBC2",
    "#A8DADC",
    "#457B9D",
    "#E63946",
    "#F4A261",
    "#2A9D8F",
    "#E9C46A",
    "#B5838D",
    "#6D6875",
    "#FFB4A2",
    "#6A4C93",
    "#FFCDB2",
    "#B8B8FF",
    "#C8E6C9",
    "#D4A5A5",
    "#99C1B9",
    "#8E7DBE",
    "#FFADAD",
    "#A0C4FF",
    "#CAFFBF",
    "#9BF6FF",
    "#FFC6FF",
    "#FFFFFC",
]


class ColorAssigner:
    _instance = None
    _color_map = {}  # 类型到颜色的映射缓存

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_color(cls, node_type: str) -> str:
        """获取节点类型对应的颜色"""
        if node_type in cls._color_map:
            return cls._color_map[node_type]

        # 使用稳定哈希分配颜色
        hash_hex = hashlib.sha256(node_type.encode()).hexdigest()
        hash_int = int(hash_hex, 16)
        color = COLOR_PALETTE[hash_int % len(COLOR_PALETTE)]

        cls._color_map[node_type] = color
        return color

    @classmethod
    def get_mapping(cls) -> List[Tuple[str, str]]:
        """获取所有已分配的节点类型-颜色对应列表"""
        return list(cls._color_map.items())


async def main():
    config = Config(
        width=1500,
        height=1000,
        directed=True,
        nodeHighlightBehavior=False,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={"labelProperty": "label"},
    )

    nodes = []
    edges = []
    # with st.spinner("加载中"):
    for id in stqdm(pgr.graph_storage._graph.nodes, desc="加载节点"):
        data = await pgr.get_node(node_id=id)
        nodes.append(
            Node(
                id=data.id,
                title=f"{data.name}\n{data.description}\n{data.entity_type}",
                label=data.name,
                color=ColorAssigner.get_color(data.entity_type),
                size=10,
            )
        )

    for i, j in stqdm(pgr.graph_storage._graph.edges, desc="加载边"):
        data = await pgr.get_edge(source_node_id=i, target_node_id=j)
        edges.append(
            Edge(
                source=data.source_id,
                target=data.target_id,
                title=data.description,
                weight=data.weight,
            )
        )

    legend_items = ColorAssigner.get_mapping()
    # 创建图例
    legend_html = """
    <style>
    .legend {
        list-style-type: none;
        padding: 0;
    }
    .legend li {
        margin-bottom: 10px;
    }
    .legend span {
        display: inline-block;
        width: 10px;
        height: 10px;
        margin-right: 10px;
        border-radius: 50%;
    }
    </style>
    <ul class="legend">
    """

    for name, color in legend_items:
        legend_html += (
            f'<li><span style="background-color: {color};"></span>{name}</li>'
        )

    legend_html += "</ul>"

    # 在 Streamlit 中渲染图例
    st.markdown(legend_html, unsafe_allow_html=True)
    return agraph(nodes=nodes, edges=edges, config=config)


asyncio.run(main())
