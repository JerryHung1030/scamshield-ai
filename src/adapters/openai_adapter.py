# src/adapters/openai_adapter.py
import logging
import asyncio
from typing import AsyncGenerator

from openai import OpenAI, OpenAIError
from .base_adapter import LLMAdapter

logger = logging.getLogger(__name__)


class OpenAIAdapter(LLMAdapter):
    """
    使用新版 openai Python 套件的方式，並指定 model="gpt-4o"。
    適用於已配置的 'gpt-4o' 模型或代理 API。
    """
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        :param openai_api_key: OpenAI或代理API的金鑰
        :param temperature: 生成溫度
        :param max_tokens: 回應最大 token 數
        """
        # 固定使用 "gpt-4o" 作為 model 名稱
        # super().__init__(model=model_name)

        self.temperature = temperature
        self.max_tokens = max_tokens

        # 使用新版 openai.OpenAI(client) 初始化
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        """
        同步 (blocking) 呼叫 gpt-4o API 取得回覆。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # 假設 response結構與 openai.ChatCompletion 類似
            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            self.handle_error(e)
            return "Error in generate_response"

    def stream_response(self, prompt: str):
        """
        同步(阻塞)streaming呼叫 gpt-4o API，yield 逐段回覆。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            # response為迭代器，每個chunk包含 delta
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except OpenAIError as e:
            self.handle_error(e)

    async def async_generate_response(self, prompt: str) -> str:
        """
        非同步方式執行 generate_response。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_response, prompt)

    async def async_stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        非同步方式streaming回傳回應內容。
        """
        loop = asyncio.get_event_loop()

        def _sync_stream():
            for chunk in self.stream_response(prompt):
                yield chunk

        gen = _sync_stream()
        while True:
            try:
                chunk = await loop.run_in_executor(None, next, gen, None)
                if chunk is None:
                    break
                yield chunk
            except StopIteration:
                break

    def handle_error(self, e: Exception) -> None:
        """
        錯誤處理: 記錄log, 並可自行擴充其他行為(通知/重試等)。
        """
        super().handle_error(e)
        logger.error(f"OpenAIAdapter error: {str(e)}")
