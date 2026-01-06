import functools

from openai import OpenAI
from typing import Dict, List

from .base import EmbeddingBase


class OpenaiEmbedding(EmbeddingBase):
    def __init__(self, config: Dict):
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.model_name = config["model_name"]

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @functools.lru_cache(maxsize=16)
    def embed(self, text) -> List[float]:
        if not text:
            return []
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding
