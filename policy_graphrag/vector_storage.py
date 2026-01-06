import os

from typing import Dict, List, Any, Optional
from pymilvus import DataType, MilvusClient, Hit

from .embeddings.base import EmbeddingBase


class VectorStorage:
    """
    一个用于与 Milvus 向量数据库交互的封装类。

    该类负责管理两个集合（collection）：'entities' 和 'text_units'，
    并提供插入（upsert）、查询（query）等向量操作。
    它使用一个外部的 embedding 客户端来将文本转换为向量。
    """

    # --- 配置常量 ---
    ENTITY_COLLECTION_NAME = "entities"
    TEXT_UNIT_COLLECTION_NAME = "text_units"

    # 实体集合中的字段名
    ENTITY_PRIMARY_FIELD = "id"
    ENTITY_NAME_FIELD = "name"
    ENTITY_TYPE_FIELD = "type"
    ENTITY_DESC_FIELD = "description"
    ENTITY_CONTENT_EMBEDDING_FIELD = "content_embedding"

    # 文本单元集合中的字段名
    TEXT_UNIT_PRIMARY_FIELD = "id"
    TEXT_UNIT_CONTENT_FIELD = "content"
    TEXT_UNIT_EMBEDDING_FIELD = "content_embedding"

    def __init__(self, embedding_client: EmbeddingBase, working_dir: str):
        """
        初始化 VectorStorage。

        Args:
            embedding_client (EmbeddingBase): 用于生成文本向量的客户端。
            working_dir (str): 存储 Milvus 本地数据库文件的目录。
        """
        self.embedding_client = embedding_client
        self.uri = os.path.join(working_dir, "milvus.db")
        self.milvus_client = MilvusClient(uri=self.uri)

        # 通过嵌入一个示例文本获取向量维度
        self._embedding_dim = len(self.embedding_client.embed("test"))

        # Milvus 搜索参数
        self.search_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 64,  # HNSW 搜索参数，值越大越精确但越慢
            },
        }

        self._initialize_collections()

    def _initialize_collections(self):
        """如果集合不存在，则创建并索引它们。"""
        self._create_collection(
            collection_name=self.ENTITY_COLLECTION_NAME,
            fields=self._get_entity_fields(),
            vector_fields_to_index=[self.ENTITY_CONTENT_EMBEDDING_FIELD],
        )
        self._create_collection(
            collection_name=self.TEXT_UNIT_COLLECTION_NAME,
            fields=self._get_text_unit_fields(),
            vector_fields_to_index=[self.TEXT_UNIT_EMBEDDING_FIELD],
        )

    def _get_entity_fields(self) -> List[Dict[str, Any]]:
        """定义实体集合的 schema 字段。"""
        return [
            {
                "field_name": self.ENTITY_PRIMARY_FIELD,
                "datatype": DataType.VARCHAR,
                "max_length": 65535,
                "is_primary": True,
            },
            {
                "field_name": self.ENTITY_NAME_FIELD,
                "datatype": DataType.VARCHAR,
                "max_length": 1000,
            },
            {
                "field_name": self.ENTITY_TYPE_FIELD,
                "datatype": DataType.VARCHAR,
                "max_length": 255,
            },
            {
                "field_name": self.ENTITY_DESC_FIELD,
                "datatype": DataType.VARCHAR,
                "max_length": 4000,
            },
            {
                "field_name": self.ENTITY_CONTENT_EMBEDDING_FIELD,
                "datatype": DataType.FLOAT_VECTOR,
                "dim": self._embedding_dim,
            },
        ]

    def _get_text_unit_fields(self) -> List[Dict[str, Any]]:
        """定义文本单元集合的 schema 字段。"""
        return [
            {
                "field_name": self.TEXT_UNIT_PRIMARY_FIELD,
                "datatype": DataType.VARCHAR,
                "max_length": 65535,
                "is_primary": True,
            },
            {
                "field_name": self.TEXT_UNIT_CONTENT_FIELD,
                "datatype": DataType.VARCHAR,
                "max_length": 65535,
            },
            {
                "field_name": self.TEXT_UNIT_EMBEDDING_FIELD,
                "datatype": DataType.FLOAT_VECTOR,
                "dim": self._embedding_dim,
            },
        ]

    def _create_collection(
        self,
        collection_name: str,
        fields: List[Dict],
        vector_fields_to_index: List[str],
    ):
        """
        一个通用的创建集合的方法，包含 schema 定义和索引创建。

        Args:
            collection_name (str): 要创建的集合的名称。
            fields (List[Dict]): 集合的字段定义列表。
            vector_fields_to_index (List[str]): 需要创建 HNSW 索引的向量字段名列表。
        """
        if self.milvus_client.has_collection(collection_name):
            print(f"集合 '{collection_name}' 已存在，跳过创建。")
            return

        schema_fields = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )

        for field in fields:
            schema_fields.add_field(**field)

        index_params = self.milvus_client.prepare_index_params()
        for field in vector_fields_to_index:
            index_params.add_index(
                field_name=field,
                index_name=f"{field}_index",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 128},
            )

        self.milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema_fields,
            index_params=index_params,
            consistency_level="Strong",
        )

        print(f"集合 '{collection_name}' 创建并索引成功。")

    def _search(
        self,
        collection_name: str,
        vector_field: str,
        query_text: str,
        top_k: int,
        output_fields: List[str],
    ) -> List[Hit]:
        """
        执行向量搜索的内部方法。

        Args:
            collection_name (str): 要搜索的集合名称。
            vector_field (str): 用于搜索的向量字段。
            query_text (str): 用户输入的查询文本。
            top_k (int): 返回的最相似结果的数量。
            output_fields (List[str]): 希望从结果中返回的字段列表。

        Returns:
            List[Hit]: Milvus 返回的搜索结果列表。
        """
        query_embedding = self.embedding_client.embed(query_text)
        results = self.milvus_client.search(
            collection_name=collection_name,
            data=[query_embedding],
            anns_field=vector_field,
            limit=top_k,
            output_fields=output_fields,
            search_params=self.search_params,
        )
        return results[0]  # 返回第一个查询的命中列表

    def query_entities(
        self, query: str, top_k: int = 10, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        通过实体名称和描述来综合搜索实体。

        此方法会分别搜索名称和描述字段，然后合并结果，
        对相同ID的实体保留最高分，并按分数从高到低排序。

        Args:
            query (str): 查询文本。
            top_k (int): 每个字段（名称/描述）要检索的候选项数量。
            threshold (float): 用于过滤结果的相似度分数阈值。

        Returns:
            List[Dict[str, Any]]: 排序和去重后的实体结果列表，包含 'entity' 和 'score'。
        """
        hits = self._search(
            collection_name=self.ENTITY_COLLECTION_NAME,
            vector_field=self.ENTITY_CONTENT_EMBEDDING_FIELD,
            query_text=query,
            top_k=top_k,
            output_fields=["*"],
        )

        filtered = [hit for hit in hits if hit["distance"] >= threshold]

        sorted_results = sorted(filtered, key=lambda h: h["distance"], reverse=True)
        results = sorted_results[:1]
        return [dict(item["entity"]) for item in results]

    def upsert_entities(self, data: List[Dict[str, Any]]):
        """
        批量插入或更新实体数据。
        此方法会为每条数据的名称和描述生成向量，然后一次性提交给 Milvus。
        Args:
        data (List[Dict[str, Any]]): 实体数据列表。每个字典应包含 id, name, type, description。
        """
        if not data:
            return

        contents_embeddings = []

        for item in data:
            name = item.get(self.ENTITY_NAME_FIELD, "")
            description = item.get(self.ENTITY_DESC_FIELD, "")
            content = name + description
            content_embedding = self.embedding_client.embed(content)
            contents_embeddings.append(content_embedding)

        data_to_upsert = [
            {
                self.ENTITY_PRIMARY_FIELD: entity["id"],
                self.ENTITY_NAME_FIELD: entity["name"],
                self.ENTITY_TYPE_FIELD: entity["type"],
                self.ENTITY_DESC_FIELD: entity["description"],
                self.ENTITY_CONTENT_EMBEDDING_FIELD: contents_embeddings[i],
            }
            for i, entity in enumerate(data)
        ]

        self.milvus_client.upsert(
            collection_name=self.ENTITY_COLLECTION_NAME, data=data_to_upsert
        )

    def upsert_text_units(self, data: List[Dict[str, Any]]):
        """
        批量插入或更新文本单元数据。

        Args:
            data (List[Dict[str, Any]]): 文本单元列表。每个字典应包含 id 和 content。
        """
        if not data:
            return

        contents = [item[self.TEXT_UNIT_CONTENT_FIELD] for item in data]
        content_embeddings = [
            self.embedding_client.embed(content) for content in contents
        ]

        data_to_upsert = [
            {
                self.TEXT_UNIT_PRIMARY_FIELD: item["id"],
                self.TEXT_UNIT_CONTENT_FIELD: item["content"],
                self.TEXT_UNIT_EMBEDDING_FIELD: content_embeddings[i],
            }
            for i, item in enumerate(data)
        ]

        self.milvus_client.upsert(
            collection_name=self.TEXT_UNIT_COLLECTION_NAME, data=data_to_upsert
        )

    def query_units(
        self, query: str, top_k: int = 10, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Args:
            query (str): 查询文本。
            top_k (int): 每个字段（名称/描述）要检索的候选项数量。
            threshold (float): 用于过滤结果的相似度分数阈值。

        Returns:
            List[Dict[str, Any]]
        """

        results = self._search(
            collection_name=self.TEXT_UNIT_COLLECTION_NAME,
            vector_field=self.TEXT_UNIT_EMBEDDING_FIELD,
            query_text=query,
            top_k=top_k,
            output_fields=["*"],
        )

        filtered = [hit for hit in results if hit["distance"] >= threshold]

        # 格式化输出
        return filtered
