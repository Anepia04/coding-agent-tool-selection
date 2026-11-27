"""大模型调用封装。

基于 OpenAI Python SDK 实现，可对接任何 OpenAI 兼容接口（如 Qwen、DeepSeek）。
其他模块只依赖 ``LLMClient.chat()`` 与 ``LLMResponse``，不直接依赖 SDK。

职责边界：
- 负责：消息发送、工具 schema 透传、文本 / tool_calls / Token 用量解析、异常统一；
- 不负责：实际执行工具（由 ``tools.registry.ToolRegistry`` 完成）。

负责成员：成员 A
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: 对话消息结构，形如 ``{"role": "user", "content": "..."}``
#: role 取值：system / user / assistant / tool
Message = Dict[str, Any]


class LLMClientError(RuntimeError):
    """大模型调用或响应解析失败时抛出的统一异常。"""


@dataclass
class ToolCall:
    """模型发起的一次工具调用请求。

    Attributes:
        id: 工具调用唯一 ID，用于把工具执行结果回填到对话中。
        name: 要调用的工具名称，需与 ToolDefinition.name 一致。
        arguments: 解析后的工具参数字典。
    """

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class LLMResponse:
    """统一的模型调用结果。

    Attributes:
        content: 模型返回的文本内容；纯工具调用时可能为空。
        tool_calls: 模型请求的工具调用列表；无工具调用时为空列表。
        input_tokens: 输入 Token 数（API 未提供时为 0）。
        output_tokens: 输出 Token 数（API 未提供时为 0）。
        total_tokens: 总 Token 数（API 未提供时按输入 + 输出计算）。
        raw: 原始响应字典，便于调试与逐字段核对。
    """

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    raw: Optional[Dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        """本次响应是否包含工具调用请求。"""
        return bool(self.tool_calls)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典（不含 raw 原始响应）。"""
        return {
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


class LLMClient:
    """OpenAI 兼容接口的大模型客户端。

    不绑定具体供应商：通过 ``base_url`` 指向 Qwen / DeepSeek 等兼容网关即可。
    API Key 优先级：显式参数 > ``LLM_API_KEY`` > ``OPENAI_API_KEY`` 环境变量。
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **options: Any,
    ) -> None:
        """
        Args:
            model: 模型名称，例如 "qwen-plus"、"deepseek-chat"。
            api_key: API Key；为空时从环境变量读取，禁止硬编码在代码中。
            base_url: OpenAI 兼容接口地址，例如
                "https://dashscope.aliyuncs.com/compatible-mode/v1"。
            **options: 默认请求参数（temperature、max_tokens、timeout 等），
                可被 ``chat(**kwargs)`` 中的同名参数覆盖。
        """
        if not model:
            raise ValueError("model 不能为空")

        self.model = model
        resolved_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.api_key = resolved_key.strip() or None
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or None
        self.options: Dict[str, Any] = dict(options)
        self._client: Any = None

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送对话消息并获取模型回复。

        Args:
            messages: 对话消息列表（OpenAI 风格）。
            tools: 可选的工具 schema 列表，通常来自 ``ToolRegistry.get_schemas()``。
            **kwargs: 本次调用级别的参数覆盖（temperature、max_tokens、timeout 等）。

        Returns:
            LLMResponse: 统一格式的模型响应（文本 + 工具调用 + Token 统计）。

        Raises:
            ValueError: messages 为空，或缺少 API Key。
            LLMClientError: SDK 未安装、网络 / API 异常、超时或响应结构无效。
        """
        if not messages:
            raise ValueError("messages 不能为空")
        if not self.api_key:
            raise ValueError(
                "缺少 API Key：请通过 api_key 参数或环境变量 LLM_API_KEY / OPENAI_API_KEY 提供"
            )

        payload: Dict[str, Any] = {"model": self.model, "messages": list(messages)}
        if tools is not None:
            payload["tools"] = list(tools)

        parameters = dict(self.options)
        parameters.update(kwargs)
        payload.update(parameters)

        client = self._get_client()
        logger.debug("调用模型 %s，消息数=%d，工具数=%d", self.model, len(messages), len(tools or []))

        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # 网络异常 / 超时 / API 错误统一包装
            raise LLMClientError(f"大模型调用失败（model={self.model}）: {exc}") from exc

        return self._parse_response(response)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """懒加载 OpenAI SDK 客户端（便于测试替换与依赖缺失时的清晰报错）。"""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖已写入 requirements
            raise LLMClientError("未安装 openai SDK，请执行 pip install -r requirements.txt") from exc

        try:
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as exc:
            raise LLMClientError(f"初始化 OpenAI 客户端失败: {exc}") from exc
        return self._client

    @staticmethod
    def _parse_arguments(function: Any) -> Dict[str, Any]:
        """解析工具调用参数（SDK 通常返回 JSON 字符串）。"""
        raw_arguments = getattr(function, "arguments", None)
        if isinstance(raw_arguments, dict):
            return dict(raw_arguments)
        if raw_arguments is None:
            return {}
        if isinstance(raw_arguments, (bytes, bytearray)):
            raw_arguments = raw_arguments.decode("utf-8", errors="replace")
        if not isinstance(raw_arguments, str):
            raise LLMClientError(f"工具调用参数类型无效: {type(raw_arguments).__name__}")

        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"工具调用参数不是合法 JSON: {raw_arguments!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMClientError(f"工具调用参数必须是 JSON 对象，实际为 {type(parsed).__name__}")
        return parsed

    def _parse_response(self, response: Any) -> LLMResponse:
        """把 SDK 响应解析为统一结构，校验关键字段并统计 Token。"""
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMClientError("模型响应缺少 choices 字段")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMClientError("模型响应缺少 message 字段")

        tool_calls: List[ToolCall] = []
        for raw_call in getattr(message, "tool_calls", None) or []:
            function = getattr(raw_call, "function", None)
            if function is None:
                raise LLMClientError("工具调用缺少 function 字段")
            name = getattr(function, "name", None)
            if not name:
                raise LLMClientError("工具调用缺少名称")
            tool_calls.append(
                ToolCall(
                    id=str(getattr(raw_call, "id", "") or ""),
                    name=str(name),
                    arguments=self._parse_arguments(function),
                )
            )

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens

        return LLMResponse(
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw=self._to_dict(response),
        )

    @staticmethod
    def _to_dict(response: Any) -> Optional[Dict[str, Any]]:
        """尽力把 SDK 响应对象转换为字典（失败时返回 None，不影响主流程）。"""
        if isinstance(response, dict):
            return dict(response)
        for method_name in ("model_dump", "to_dict", "dict"):
            method = getattr(response, method_name, None)
            if callable(method):
                try:
                    data = method()
                except Exception:  # noqa: BLE001 - 序列化失败不影响调用结果
                    continue
                if isinstance(data, dict):
                    return data
        return None


__all__ = ["LLMClient", "LLMClientError", "LLMResponse", "ToolCall", "Message"]
