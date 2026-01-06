import json
from typing import Dict, List, Optional, Union
from openai import AsyncOpenAI, OpenAI
from .base import LLMProviderBase


class OpenAIProvider(LLMProviderBase):
    def __init__(self, config: Dict):
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        self.model_name = config["model_name"]
        self.temperature = config["temperature"]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_tokens = config.get("max_tokens", 4096)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _parse_response(
        self, response, tools: Optional[List[Dict]] = None
    ) -> Union[str, Dict]:
        """Process the response based on whether tools are used or not."""
        content = response.choices[0].message.content
        if not tools:
            return content

        processed_response = {"content": content, "tool_calls": []}
        tool_calls = response.choices[0].message.tool_calls or []
        for tool_call in tool_calls:
            processed_response["tool_calls"].append(
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                }
            )
        return processed_response

    def _parse_response_stream(
        self, response, tools: Optional[List[Dict]] = None
    ) -> Union[str, Dict]:
        """Process the response based on whether tools are used or not."""
        content = response.choices[0].delta.content

        if content is None:
            return ""

        if not tools:
            return content

        processed_response = {"content": content, "tool_calls": []}
        tool_calls = response.choices[0].delta.tool_calls or []
        for tool_call in tool_calls:
            processed_response["tool_calls"].append(
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                }
            )
        return processed_response

    def _build_params(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools=None,
        tool_choice="auto",
    ) -> Dict:
        """Helper to build request parameters."""
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"enable_thinking": False},
        }
        if response_format:
            params["response_format"] = response_format
        if tools:
            params.update({"tools": tools, "tool_choice": tool_choice})
        return params

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        """Synchronously generate a response."""
        params = self._build_params(messages, response_format, tools, tool_choice)

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)

    async def async_generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        """
        Generate a response based on the given messages using OpenAI.
        """
        params = self._build_params(messages, response_format, tools, tool_choice)
        response = await self.async_client.chat.completions.create(**params)
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
        for item in self.client.chat.completions.create(**params):
            yield self._parse_response_stream(item, tools)

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
        async for item in await self.async_client.chat.completions.create(**params):
            yield self._parse_response_stream(item, tools)
