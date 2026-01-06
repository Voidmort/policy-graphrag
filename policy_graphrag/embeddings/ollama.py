from ollama import Client

from .base import EmbeddingBase


class OllamaEmbedding(EmbeddingBase):
    def __init__(self, config):
        self.embedding_model = config["embedding_model"]
        self.embedding_dims = config["embedding_dims"]
        self.embedding_base_url = config["embedding_base_url"]

        self.client = Client(host=self.embedding_base_url)

    def embed(self, text):
        """
        Get the embedding for the given text using Ollama.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        response = self.client.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]
