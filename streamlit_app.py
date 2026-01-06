import asyncio
import hashlib
import random
import streamlit as st
import torch
import os

from torch import classes  # 显式导入类注册模块

torch.classes.__path__ = []

from traceback import print_exception
from streamlit_agraph import agraph, Node, Edge, Config
from typing import Literal
from collections import defaultdict
from dotenv import load_dotenv

# 加载 .env 文件（默认从当前目录加载）
load_dotenv(".env")

from policy_graphrag.data_model.query import QueryParam
from policy_graphrag.embeddings import OpenaiEmbedding
from policy_graphrag.llms import OpenAIProvider
from policy_graphrag import PolicyGraphRAG


# --- Streamlit 配置 ---
st.set_page_config(page_title="政策问答助手", layout="wide", menu_items=None)

st.markdown(
    '<h1 style="text-align: center; font-size: 42px;">📘 政策问答助手</h1>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_question_list():
    return [
        "哪些政策文件由教育信息化部门负责技术支持与系统运行保障？",
        "哪些政策文件涉及高校辅导员培训和研修基地的备案评估及辅导员专业技术职务聘任情况？",
        "哪些政策文件涉及非折叠纸盒类包装装潢印刷品印制质量的判定标准？",
        "哪些政策文件涉及职业教育课程思政建设并包含课程思政示范项目？",
        "哪些政策文件涉及考生考试报名时间的安排以及相关的考试通知？",
        "哪些政策文件涉及第八届全国青少年民族器乐教育教学成果展示活动的截止时间及相关管理规定？",
        "哪些政策文件涉及知识产权保护并要求学员尊重授课讲师相关资料的知识产权？",
        "哪些政策文件涉及校园安全卫生和生命安全与健康教育进中小学课程教材指南的关系？",
        "哪些政策文件涉及教育部职业院校教学指导委员会委员的组织推荐及主任委员单位的确定？",
        "哪些政策文件涉及教育部关于举办2015年全国职业院校技能大赛的通知以及宴席设计的相关要求？",
        '"哪些政策文件涉及政府收支分类科目""1100245 教育共同财政事权转移支付收入""并用于支持特殊教育事业发展？"的意见在教育资源配置方面有何具体衔接与实施举措？',
        '"第十三届""桃李杯""全国青少年舞蹈教育教学成果现场展示活动的领队会及抽签会时间与地点分别是什么？"',
        "第五届全国大学生网络文化节和全国高校网络教育优秀作品推选展示活动入选名单的通知是由谁制定的？",
        "2023年4月1日推荐截止时间与文化和旅游部办公厅关于实施2023年全国文化艺术职业教育和旅游职业教育提质培优行动计划的通知之间存在何种关系？",
        "第二届中华经典诵写讲大赛的成果展示安排与教育部办公厅关于举办第二届中华经典诵写讲大赛的通知之间存在何种关系？",
        "疫情防控常态化背景下，2021年同等学力人员申请硕士学位全国统一考试安全和防疫工作的通知如何确保考试安全与疫情防控的双重目标？",
        "公安部和教育部联合制定的中小学幼儿园安全防范工作规范(试行)对校园安全防范工作有何具体影响？",
        '"直属高校基本建设信息网是教育部办公厅关于直属高校开展""十三五""基本建设规划编制工作的通知指定平台配套支持的信息化平台。"',
        "哪些院校可以参与2023年全国文化艺术职业教育和旅游职业教育提质培优行动计划的申报？",
        '"教育部关于印发2015年全国硕士研究生招生工作管理规定的通知与省(区,市)高等学校招生委员会在招生工作管理中存在何种关系？"',
        "每名选手限报1名指导教师的规则是否适用于同一学校报名人数不超过2人的情况？",
        "教育部关于同意中国地质大学江城学院转设为武汉工程科技学院的函中提到的涉及部门是否包括中国地质大学(武汉)？",
        "人力资源社会保障部办公厅关于启用新版技工院校毕业证书的通知与人社厅函[2022]76号之间存在何种关系？",
        "专业技术人才知识更新工程2025年高级研修项目计划的通知中提到的高级研修项目如何通过专业技术人才知识更新工程公共服务平台进行申报",
    ]


question_list = load_question_list()


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


# --- 配色工具 ---
COLOR_PALETTE = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEEAD",
    "#FF9F76",
    "#A3C9A8",
    "#84A59D",
    "#F28482",
    "#679436",
    "#F7B267",
    "#2F4858",
]


class EnhancedColorAssigner:
    _color_map = {}

    @classmethod
    def get_color(cls, node_id: str) -> str:
        if node_id not in cls._color_map:
            hash_hex = hashlib.sha256(node_id.encode()).hexdigest()
            hash_int = int(hash_hex, 16)
            cls._color_map[node_id] = COLOR_PALETTE[hash_int % len(COLOR_PALETTE)]
        return cls._color_map[node_id]

    @classmethod
    def get_node_size(cls, degree: int, base_size: int = 20, scale: float = 2.0) -> int:
        return base_size + int(degree * scale)


# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_result" not in st.session_state:
    st.session_state.search_result = {}


# --- 三栏布局 ---
col1, col2, col3 = st.columns([0.5, 3, 2])


# ------------------- 左栏：配置 -------------------
with col1:
    st.header("⚙️ 配置")

    # 查询参数
    mode: Literal["graph", "naive"] = st.radio("模式选择", ["graph", "naive"], index=0)
    only_need_context: bool = st.checkbox("仅返回知识库", False)
    level: int = st.slider("搜索层级", 1, 5, 2)
    top_k: int = st.slider("Top-K", 1, 50, 10)
    threshold: float = st.slider("相似度阈值", 0.0, 1.0, 0.7, 0.01)

    st.markdown("---")
    st.subheader("📥 添加政策文件")
    uploaded_files = st.file_uploader(
        "添加政策文件",
        label_visibility="hidden",
        accept_multiple_files=True,
        type=["txt", "md"],
        help="支持上传多个文本文件（.txt, .md）",
    )

    if st.button("添加到知识库"):

        async def upload_file():
            if uploaded_files:
                success_count = 0
                error_files = []
                with st.spinner(f"正在处理 {len(uploaded_files)} 个文件..."):
                    for uploaded_file in uploaded_files:
                        try:
                            # 从 UploadedFile 对象中获取文件名和内容
                            filename = uploaded_file.name
                            # .read() 返回 bytes, 需要解码为 string
                            file_content = uploaded_file.read().decode("utf-8")

                            # 为每个文件调用 index
                            await pgr.index(
                                policy_name=filename,
                                content=file_content,
                                is_update_community=False,
                            )
                            await pgr.index_naive(
                                policy_name=filename,
                                content=file_content,
                            )
                            success_count += 1
                        except Exception as e:
                            print_exception(e)
                            st.error(f"处理文件 {uploaded_file.name} 时出错: {e}")
                            error_files.append(uploaded_file.name)

                if success_count > 0:
                    st.success(f"{success_count} 个文件已成功添加到知识库！")
                if error_files:
                    st.warning(
                        f"{len(error_files)} 个文件处理失败: {', '.join(error_files)}"
                    )
            else:
                st.warning("请先上传至少一个文件。")

        asyncio.run(upload_file())

    st.markdown("---")
    st.subheader("🧩 图谱显示配置")
    graph_config = {
        "width": st.slider("画布宽度", 500, 1500, 800),
        "height": st.slider("画布高度", 400, 1200, 600),
        "directed": st.checkbox("显示方向箭头", True),
        "physics": st.checkbox("启用物理引擎", True),
    }


# ------------------- 中栏：聊天 -------------------
with col2:
    st.header("💬 对话")
    query_param = QueryParam(
        mode=mode,
        only_need_context=only_need_context,
        level=level,
        top_k=top_k,
        threshold=threshold,
    )

    # 创建独立的消息显示容器
    chat_container = st.container(height=650)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 将输入框放在容器外部
    if prompt := st.chat_input("输入消息..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 定义 get_stream 函数
        async def get_stream(result_msgs):
            async for item in pgr.query(prompt, query_param):
                msg = ""
                if "context" in item:
                    st.session_state.search_result["context"] = item["context"]
                if "context_report" in item:
                    st.session_state.search_result["context_report"] = item[
                        "context_report"
                    ]
                if "llm_response" in item:
                    msg = item["llm_response"]
                    result_msgs.append(msg)
                yield msg

        # 显示加载状态
        result_msgs = []
        with chat_container:
            with st.chat_message("assistant"):
                # write_stream 需要在 chat_message 上下文中调用
                st.write_stream(get_stream(result_msgs))

        # 更新会话消息
        st.session_state.messages.append(
            {"role": "assistant", "content": "".join(result_msgs)}
        )

    candidate_questions = random.sample(question_list, 5)
    st.markdown("🙋推荐问题：")
    for i in range(len(candidate_questions)):
        st.markdown(f"{i+1}. {candidate_questions[i]}")


# ------------------- 右栏：上下文 -------------------
with col3:
    st.header("🔍 知识库")
    st.markdown("---")
    context = st.session_state.search_result.get("context")
    if context and mode == "naive" and isinstance(context, list):
        with st.expander("📝 文本片段", expanded=True):
            for doc in context:
                st.markdown(f"- {doc}  \n")

    if context and mode == "graph" and isinstance(context, dict):
        st.subheader("🌐 知识图谱")

        # --- 优化开始 ---
        async def _create_node_object(node_name, entity_type, description, degree):
            """辅助函数，用于创建 Node 对象"""
            return Node(
                id=node_name,
                title=f"{node_name}\n类型：{entity_type}\n描述：{description}",
                label=node_name,
                size=EnhancedColorAssigner.get_node_size(degree),
                color=EnhancedColorAssigner.get_color(entity_type),
                shape="dot",
            )

        async def build_graph():
            node_degrees, node_ids_from_edges = defaultdict(int), set()
            nodes, edges = [], []

            # 1. 构建 edges 并计算度数
            for edge in context.get("edges", []):
                source, target = edge.source, edge.target
                if source and target:
                    node_ids_from_edges.update([source, target])
                    node_degrees[source] += 1
                    node_degrees[target] += 1
                    edges.append(
                        Edge(
                            source=source,
                            target=target,
                            label=edge.description,
                            color="#A0AEC0",
                        )
                    )

            # 2. 处理上下文中已有的节点
            nodes_in_context = {}
            for node in context.get("nodes", []):
                nodes_in_context[node.name] = node
                nodes.append(
                    await _create_node_object(
                        node_name=node.name,
                        entity_type=node.entity_type,
                        description=node.description,
                        degree=node_degrees[node.name],
                    )
                )
                # 如果节点已处理，从待抓取集合中移除
                if node.name in node_ids_from_edges:
                    node_ids_from_edges.remove(node.name)

            # 3. 抓取剩余的、仅在边中出现的节点
            for node_id in node_ids_from_edges:
                node = await pgr.get_node(entity_name=node_id)
                if node:
                    nodes.append(
                        await _create_node_object(
                            node_name=node.name,
                            entity_type=node.entity_type,
                            description=node.description,
                            degree=node_degrees[node.name],
                        )
                    )

            # 4. 渲染图谱
            if nodes and edges:
                config = Config(**graph_config)
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.info("未找到可显示的图谱数据。")

        try:
            asyncio.run(build_graph())
        except RuntimeError as e:
            if "cannot run" in str(e):
                st.error("图谱构建异步错误：Streamlit 事件循环冲突。请尝试刷新页面。")
            else:
                st.error(f"图谱构建失败: {e}")
        # --- 优化结束 ---

        st.markdown("---")

        cites = pgr.get_cites_policy(context)
        if cites:
            with st.expander("📜 相关政策", expanded=True):
                for cite in cites:
                    st.markdown(f"- {cite}")
        else:
            st.markdown("无相关政策")

        context_report = st.session_state.search_result.get("context_report")
        if context_report:
            # 使用 st.expander 来节省空间
            with st.expander("📝 上下文报告"):
                st.markdown(context_report)
