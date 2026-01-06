import functools

from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from .base import EmbeddingBase


class HuggingFaceEmbedding(EmbeddingBase):
    def __init__(self, config):
        self.device = config["device"]
        self.embedding_model = config["embedding_model"]
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model, model_kwargs={"device": self.device}
        )

    @functools.lru_cache(maxsize=16)
    def embed(self, text) -> List[float]:
        """
        Get the embedding for the given text using Ollama.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        return self.embeddings.embed_query(text)
