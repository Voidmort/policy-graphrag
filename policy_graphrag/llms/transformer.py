import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Dict, List, Optional, Union

from .base import LLMProviderBase


class TransformerProvider(LLMProviderBase):
    def __init__(self, config: Dict):
        self.model_name = config["model_name"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        self.top_p = config["top_p"]
        self.enable_thinking = config.get("enable_thinking", False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        """Synchronously generate a response."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        if not tools:
            return response
        return {"content": response, "tool_calls": []}

    async def async_generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Union[str, Dict]:
        """Asynchronously generate a response."""
        return self.generate_response(messages, response_format, tools, tool_choice)

    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """Stream a response synchronously (generator)."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            decode_with_accumulated=True,
        )

        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            streamer=streamer,
        )

        # 用线程启动生成，不阻塞主线程
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    async def async_generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """Stream a response asynchronously (async generator)."""
        for chunk in self.generate_response_stream(
            messages, response_format, tools, tool_choice
        ):
            yield chunk
