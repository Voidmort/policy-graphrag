import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Dict, List, Optional, Any, AsyncGenerator, Generator
from threading import Thread

from .base import LLMProviderBase


class QwenLLMProvider(LLMProviderBase):
    """
    一个使用 Hugging Face's transformers 库与 Qwen 模型交互的 LLMProvider。
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", **kwargs: Any):
        """
        初始化模型和分词器。

        Args:
            model_name (str): 要从 Hugging Face Hub 加载的模型名称。
            **kwargs: 传递给 AutoModelForCausalLM.from_pretrained 的其他关键字参数。
                      例如: torch_dtype="auto", device_map="auto"
        """
        print(f"正在加载模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.enable_thinking = False  # Qwen3 的 enable_thinking 默认为 False
        # 为 from_pretrained 设置默认值，并允许用户覆盖
        default_model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        model_kwargs = {**default_model_kwargs, **kwargs}

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("模型加载完成。")

    def _prepare_inputs(self, messages: List[Dict[str, str]]):
        """
        一个辅助方法，用于准备模型的输入。
        """
        # 注意: Qwen3 的 enable_thinking 默认为 False，如果你需要模型思考，可以设为 True
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return self.tokenizer([text], return_tensors="pt").to(self.model.device)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Dict:
        """
        同步、非流式生成响应。
        """
        if tools:
            print("警告: 当前 QwenLLMProvider 实现不支持工具（tools）。")
        if response_format:
            print(f"警告: 当前 QwenLLMProvider 实现不支持 response_format。")

        model_inputs = self._prepare_inputs(messages)

        # 设置默认生成参数，并允许用户通过 kwargs 覆盖
        generation_kwargs = {"max_new_tokens": 32768, **kwargs}

        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        # 从生成结果中移除输入部分
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

        # 解析 "thinking" 内容
        # Qwen3 的 </think> token id 是 151668
        think_end_token_id = 151668
        try:
            # 从后往前找 </think>
            index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip()
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip()

        return {
            "content": content,
            "tool_calls": [],  # 当前实现不支持工具
            "thinking_content": thinking_content,  # 添加额外字段以返回思考内容
        }

    def generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        同步、流式生成响应。
        """
        if tools:
            print("警告: 当前 QwenLLMProvider 实现不支持工具（tools）。")
        if response_format:
            print(f"警告: 当前 QwenLLMProvider 实现不支持 response_format。")

        model_inputs = self._prepare_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            "max_new_tokens": 32768,
            **kwargs,
            **model_inputs,
            "streamer": streamer,
        }

        # 因为 generate 是阻塞的，我们需要在单独的线程中运行它
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 从 streamer 中 yield 每个生成的 token
        for new_text in streamer:
            yield new_text

    async def async_generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Dict:
        """
        异步、非流式生成响应。
        通过在线程池中运行同步方法来实现。
        """
        # 使用 asyncio.to_thread 在独立的线程中运行阻塞的同步代码
        return await asyncio.to_thread(
            self.generate_response,
            messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def async_generate_response_stream(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        异步、流式生成响应。
        这是一个将同步生成器转换为异步生成器的常见模式。
        """
        sync_generator = self.generate_response_stream(
            messages, response_format, tools, tool_choice, **kwargs
        )
        # 遍历同步生成器并将结果放入队列
        for item in sync_generator:
            yield item
