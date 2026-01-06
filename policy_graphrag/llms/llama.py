from typing import Dict, List, Optional, Union
from ollama import Client, AsyncClient
from .base import LLMProviderBase


class LlamaProvider(LLMProviderBase):
    def __init__(self, config: Dict):
        self.model = config["model_name"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.top_p = config["top_p"]
        self.client = Client(host=config["base_url"])
        self.async_client = AsyncClient(host=config["base_url"])

    def _parse_response(
        self, response: Dict, tools: Optional[List[Dict]] = None
    ) -> Union[str, Dict]:
        """Process the response based on whether tools are used or not."""
        content = response["message"]["content"]
        if not tools:
            return content

        processed_response = {"content": content, "tool_calls": []}
        for tool_call in response["message"].get("tool_calls", []):
            processed_response["tool_calls"].append(
                {
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                }
            )
        return processed_response

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Dict:
        """Helper to build request parameters."""
        params = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "top_p": self.top_p,
            },
        }
        if response_format:
            params["format"] = "json"
        if tools:
            params["tools"] = tools
        return params

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        """Synchronously generate a response."""
        params = self._build_params(messages, response_format, tools)
        response = self.client.chat(**params)
        return self._parse_response(response, tools)

    async def async_generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        """Asynchronously generate a response."""
        params = self._build_params(messages, response_format, tools)
        response = await self.async_client.chat(**params)
        return self._parse_response(response, tools)

    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """Stream a response."""
        params = self._build_params(messages, response_format, tools)
        params["stream"] = True
        for item in self.client.chat(**params):
            yield self._parse_response(item, tools)

    async def async_generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """Stream a response."""
        params = self._build_params(messages, response_format, tools)
        params["stream"] = True
        async for item in await self.async_client.chat(**params):
            yield self._parse_response(item, tools)
