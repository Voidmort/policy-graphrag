## Policy-GraphRAG

面向政策文本的 GraphRAG 引擎与 NaiveRAG 检索问答实现。本项目以图谱为核心，将文档切片、实体与关系抽取、图存储、社区报告生成、向量检索与大模型问答串联在一起，提供结构化的“图 + 文本”上下文用于回答复杂查询。

本文档重点介绍 `policy_graphrag.py` 中的核心类 `PolicyGraphRAG`，以及其两个主要方法：`index` 与 `query`。

### 目录
- 安装与环境
- 项目结构
- 快速开始
- 工作目录与数据产物
- 核心类：PolicyGraphRAG
  - 初始化参数
  - 索引方法：index（重点）
  - 查询方法：query（重点）
  - Graph 模式的上下文构建
  - Naive 模式的上下文构建
- 进阶说明与可配置项
- 常见问题

---

## 安装与环境

- Python 版本：建议 3.10+
- 安装依赖：

```bash
pip install -r requirements.txt
```

> 说明：项目需要可用的 LLM 与 Embedding 提供者（见 `llms/` 与 `embeddings/` 目录）。你需要根据自身环境提供实现或适配器，例如 OpenAI、Qwen、Azure OpenAI 等。

## 项目结构

目录 `policy_graphrag/` 内主要文件与子目录结构如下：

```bash
policy_graphrag/
  ├─ __init__.py                # 包初始化
  ├─ README.md                  # 使用与设计文档（本文件）
  ├─ requirements.txt           # 依赖清单
  ├─ policy_graphrag.py         # 核心：PolicyGraphRAG（index/query 主流程）
  ├─ graph_storage.py           # 图存储接口与实现（节点/边/聚类/社区相关操作）
  ├─ vector_storage.py          # 向量存储/检索（实体与文本块）
  ├─ graph_community.py         # 社区报告生成（对聚类结果做语义汇总）
  ├─ stable_lcc.py              # 稳定最大连通子图/聚类相关工具
  ├─ utils.py                   # 文本切片、规范化、解析、工具方法集合
  ├─ prompts.py                 # LLM 提示词模板（抽取/总结/回答 等）
  ├─ llms/                      # LLM Provider 适配（需实现/配置具体厂商）
  ├─ embeddings/                # Embedding Provider 适配
  └─ data_model/                # Pydantic 数据模型（Node/Edge/Chunk/QueryParam 等）
```

说明：
- **policy_graphrag.py**：
  - `PolicyGraphRAG` 是本项目入口类，负责完整的“切片→抽取→合并/消歧→写入图与向量→聚类→查询构建→调用 LLM”流程。
  - 重点方法：
    - `index(...)`：构建/更新索引与图谱、社区报告；
    - `query(...)`：面向 Graph/Naive 两种模式的问答（可流式返回）。
- **graph_storage.py**：抽象并封装了图的持久化与查询操作（如 upsert 节点/边、度计算、聚类调用、节点边检索等）。
- **vector_storage.py**：对接向量化与相似度检索，提供实体与文本块两条索引/查询路径（含 naive 子目录的数据）。
- **graph_community.py**：基于图聚类结果生成社区报告（主题/标题/摘要/评分），并提供与 `PolicyGraphRAG` 的集成函数。
- **stable_lcc.py**：与最大连通子图或更稳定的聚类选取相关的工具逻辑。
- **utils.py**：通用工具函数集合，包含文档解析、按句切片、重叠拼接、去重、token 预算裁剪、ID 生成与规范化等。
- **prompts.py**：抽取/补充/总结/回答等环节的提示词模板，`PolicyGraphRAG` 会在不同阶段引用相应模板。
- **llms/**：大模型提供方的基类与适配层，需在此目录提供可用实现（例如 OpenAI、Qwen、Azure OpenAI 等）。
- **embeddings/**：向量嵌入提供方的基类与适配层，需在此目录提供可用实现。
- **data_model/**：核心数据结构定义，常见模型包括 `Node`、`Edge`、`Chunk`、`Entity`、`CommunityReport`、`QueryParam` 等。
- **requirements.txt**：当前目录下维护的依赖清单，安装时使用 `pip install -r requirements.txt`。

## 快速开始

```python
from policy_graphrag.policy_graphrag import PolicyGraphRAG
from policy_graphrag.llms import LLMProviderBase  # 需提供具体实现
from policy_graphrag.embeddings import EmbeddingBase  # 需提供具体实现
from policy_graphrag.data_model.query import QueryParam


# 1) 初始化：指定工作目录、LLM、Embedding
working_dir = "/path/to/workdir"  # 将在此目录下生成/读取 parquet 数据
llm_provider = YourLLMProvider(...)
embed_provider = YourEmbeddingProvider(...)

rag = PolicyGraphRAG(
    working_dir=working_dir,
    llm_provider=llm_provider,
    embed_provider=embed_provider,
)

# 2) 索引：可传入文件列表或直接传入文本内容
await rag.index(
    file_paths=["/path/to/policy1.txt", "/path/to/policy2.txt"],
    is_update_community=True,  # 可选：是否在索引完成后计算/更新社区报告
)

# 3) 查询：Graph 模式（基于图检索）或 Naive 模式（基于向量检索）
query_param = QueryParam(
    mode="graph",                # "graph" 或 "naive"
    top_k=10,
    threshold=0.8,
    level=2,                      # 社区层级过滤（graph 模式）
    only_need_context=False,      # 仅返回上下文，不调用 LLM
)

async for item in rag.query("请概括关于新能源补贴的主要政策要点", query_param):
    if "context" in item:
        print("Context ->", item["context"])               # 原始上下文对象（graph 或 naive）
    if "context_report" in item:
        print("Context Report ->", item["context_report"]) # 图模式下的 CSV 格式上下文报告字符串
    if "llm_response" in item:
        print("LLM ->", item["llm_response"])              # 流式返回
```

## 工作目录与数据产物

在 `working_dir` 下会生成或读取以下文件（如存在则加载增量更新）：
- `policy_docs.parquet`：政策文档元信息（`document_id`, `name`）。
- `chunk_text.parquet`：切片后的文本（`id`, `content`, `document_id`）。
- `communities_report.parquet`：图社区报告，包括 `community_id`, `title`, `level`, `summary` 等。

此外，图谱与向量索引由以下组件负责：
- 图存储：`graph_storage.GraphStorage`（节点、边、聚类/社区等）
- 向量存储：`vector_storage.VectorStorage`（实体与文本块检索）

## 核心类：PolicyGraphRAG

入口文件：`policy_graphrag/policy_graphrag.py`

### 初始化参数

```python
PolicyGraphRAG(
    working_dir: str,
    llm_provider: LLMProviderBase,
    embed_provider: EmbeddingBase,
)
```

- `working_dir`：用于持久化 parquet 数据与索引的目录。
- `llm_provider`：大模型提供者，需实现：
  - `async_generate_response(messages, **kwargs) -> str`
  - `async_generate_response_stream(messages, **kwargs) -> AsyncIterator[str]`
- `embed_provider`：Embedding 提供者，需实现：
  - `embed(text: str) -> List[float]`

内部还会创建：
- `graph_storage = GraphStorage(working_dir)`
- `vector_storage = VectorStorage(embed_provider, working_dir)`
- `naive_vector_storage = VectorStorage(embed_provider, f"{working_dir}/naive")`

默认配置（可在代码中调整）：
- `segment_length = 1000`，`overlap_length = 50`（切片参数）
- `llm_max_token_size = 8192`
- `summary_max_tokens = 200`
- `entity_extract_max_gleaning = 0`（是否进行多轮“捡漏”抽取）

### 索引方法：index

```python
await PolicyGraphRAG.index(
    file_paths: List[str] = [],
    policy_name: str = "",
    content: str = "",
    source_nodes: List[Node] = [],
    source_edges: List[Edge] = [],
    is_update_community: bool = False,
)
```

功能概述：
- 支持两种索引来源：
  - `file_paths`：从本地文件读取内容。
  - `content` + `policy_name`：直接传入文本。
- 处理流程（简化）：
  1) 文本切片：按句子合并到 `segment_length`、`overlap_length`；生成 `Chunk` 列表。
  2) 实体与关系抽取：调用 LLM 解析实体与三元组，得到候选节点、边。
  3) 合并与消歧：
     - `merge_nodes`/`merge_edges` 合并重复信息并进行长描述摘要；
     - `disambiguate_entities` 进行实体归一化、弥补缺失节点及孤立点；
  4) 写入存储：
     - `update_graph_and_entities` 将节点、边写入 `GraphStorage`，并对实体入向量库；
     - 触发 `graph_storage.clustering()` 计算社区/聚类；
  5) 产物落盘：
     - 更新 `chunk_text.parquet` 与 `policy_docs.parquet`；
     - `is_update_community=True` 时，额外生成/合并 `communities_report.parquet`。

可选输入：
- `source_nodes` / `source_edges`：允许外部注入先验节点和边，与抽取结果一起去重合并。

常见用法：
- 批量文件索引：仅传 `file_paths`。
- 单条文本索引：传 `content` 与 `policy_name`。
- 增量更新社区报告：设置 `is_update_community=True`。

### 查询方法：query

```python
async for item in PolicyGraphRAG.query(query: str, query_param: QueryParam):
    ...
```

`QueryParam` 关键字段（来自 `data_model/query.py`，下为实际使用到的字段）：
- `mode`：`"graph" | "naive"`，查询模式。
- `only_need_context`：仅返回上下文，不进行 LLM 生成。
- `top_k`：向量检索返回数量。
- `threshold`：向量相似度阈值。
- `level`：社区层级上限（Graph 模式用于筛选社区报告）。
- `local_max_token_for_community_report`：社区报告拼接的最大 token 预算。
- `local_max_token_for_text_unit`：文本切片拼接的最大 token 预算。
- `local_max_token_for_local_context`：关系边拼接的最大 token 预算。

返回（流式异步迭代器）：
- `{"context": ...}`：上下文对象。
- `{"context_report": ...}`：Graph 模式下的 CSV 报告字符串（含 Entities/Relationships/Sources/Reports 四段）。
- `{"llm_response": ...}`：语言模型的增量输出片段。

两种模式说明：

1) Graph 模式（`mode="graph"`）
   - 步骤：
     - 召回相关实体（基于实体向量）；
     - 基于实体找到关系边、相关文本切片、相关社区报告；
     - 将它们裁剪到各自的 token 预算；
     - 以 CSV 报告字符串形式组织上下文，传入 LLM 生成答案。
   - 适用：需要结构化因果/约束关系、跨文档关联、政策脉络解释的场景。

2) Naive 模式（`mode="naive"`）
   - 步骤：
     - 直接在 `naive_vector_storage` 中检索相似文本块；
     - 拼接为上下文传入 LLM。
   - 适用：简单的语义匹配问答、无需图结构增强的场景。

示例：

```python
from policy_graphrag.data_model.query import QueryParam

# Graph 模式：需要结构化证据链
params_graph = QueryParam(mode="graph", top_k=10, threshold=0.8, level=2)
async for item in rag.query("本市中小企业税收减免的资格条件是什么？", params_graph):
    if "llm_response" in item:
        print(item["llm_response"], end="")

# Naive 模式：轻量语义检索
params_naive = QueryParam(mode="naive", top_k=6, threshold=0.75)
async for item in rag.query("新能源车购置补贴标准", params_naive):
    if "llm_response" in item:
        print(item["llm_response"], end="")
```

### Graph 模式的上下文构建

对应方法：`_build_graph_rag_query_context`

- 实体召回：`search_node(query)`，返回带 `rank` 的节点集合。
- 文本切片：`_find_most_related_text_unit_from_entities`，基于节点一跳邻居的共现度排序，按 `local_max_token_for_text_unit` 预算截断。
- 关系边：`_find_most_related_edges_from_entities`，对边按 `(rank, weight)` 排序，按 `local_max_token_for_local_context` 截断。
- 社区报告：`_find_most_related_community_from_entities`，按评分与层级筛选，按 `local_max_token_for_community_report` 截断。
- 输出：
  - `context`：原始对象集合（`nodes`, `edges`, `unities`, `communities`）。
  - `context_report`：将上述集合序列化为 4 段 CSV 文本，便于 LLM 阅读与对齐。

### Naive 模式的上下文构建

对应方法：`_build_naive_rag_query_context`

- 直接在 `naive_vector_storage` 上进行文本块检索，返回文本内容列表作为上下文。

## 可配置项

- 实体抽取与关系抽取提示词：见 `prompts.py` 中的相关模板（如 `ENTITY_EXTRACTION_SYSTEM_PROMPT`、`ENTITY_EXTRACT_USER_PROMPT`、`ENTITIES_CONTINUE_EXTRACTION_PROMPT`、`GRAPH_RAG_RESPONSE_PROMPT`、`NAIVE_RAG_RESPONSE_PROMPT` 等）。
- 实体/关系合并时的长描述会通过 `handle_entity_relation_summary` 触发二次摘要，最大长度由 `summary_max_tokens` 控制。
- `entity_extract_max_gleaning` 可开启对单 chunk 的多轮“捡漏”抽取，受 `llm_max_token_size` 限制。
- 图聚类/社区：在每次 `update_graph_and_entities` 后调用 `graph_storage.clustering()`。社区报告由 `graph_community.generate_community_report` 生成并合并落盘。
- NaiveRAG 独立索引：如需仅构建简单向量索引，可使用 `index_naive` + `query_naive`（或 `mode="naive"` 的 `query`）。

## 常见问题（FAQ）

- 为什么查询没有返回答案？
  - 首次运行请确保调用过 `index` 构建索引；
  - 检查 `working_dir` 下 parquet 文件是否生成；
  - 调整 `top_k`、`threshold` 或切换到 `naive` 模式检查召回效果；
  - 确认 LLM 与 Embedding 提供方可用。

- 如何定位引用来源？
  - 图模式可通过 `PolicyGraphRAG.get_cites_policy(context)` 从 `context` 中汇总涉及的文档名。

- 是否支持自定义图存储或向量库？
  - 可以。适配或实现 `GraphStorage`、`VectorStorage` 接口，并在初始化时替换相应实现。

---

如需进一步扩展或接入特定云厂商的模型与向量服务，请参考 `llms/` 与 `embeddings/` 目录下的基类，编写适配器并在初始化时注入。


