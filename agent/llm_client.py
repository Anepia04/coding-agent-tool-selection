"""大模型调用封装。

本模块隔离具体模型供应商：其他模块只依赖 ``LLMClient.chat()`` 与
``LLMResponse``，不直接依赖 openai / 其他 SDK。

当前阶段只定义接口与统一响应结构，不绑定任何供应商实现。

负责成员：成员 A
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: 对话消息结构，形如 ``{"role": "user", "content": "..."}``
#: role 取值：system / user / assistant / tool
Message = Dict[str, Any]


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
        total_tokens: 总 Token 数（API 未提供时为 0）。
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
    """大模型客户端接口。

    约定：
    - ``messages`` 采用 OpenAI 风格的对话消息列表；
    - ``tools`` 由 ``ToolRegistry.get_schemas()`` 生成；
    - 返回值统一为 ``LLMResponse``。
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
            model: 模型名称，例如 "gpt-4o-mini"。
            api_key: API Key，默认从环境变量读取（不写入代码）。
            base_url: API 服务地址，兼容自建 / 代理服务。
            **options: 其他供应商相关参数（temperature、max_tokens 等）。
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.options: Dict[str, Any] = dict(options)

    def chat(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """发送对话消息并获取模型回复。

        Args:
            messages: 对话消息列表。
            tools: 可选的工具 schema 列表（来自 ToolRegistry.get_schemas）。
            **kwargs: 本次调用级别的参数覆盖（如 temperature）。

        Returns:
            LLMResponse: 统一格式的模型响应（文本 + 工具调用 + Token 统计）。

        Raises:
            NotImplementedError: 当前阶段未绑定具体模型供应商。
        """
        # TODO(成员 A)：接入具体供应商 SDK，解析 content / tool_calls / usage，
        # 组装为 LLMResponse；网络异常时抛出清晰的运行时异常。
        raise NotImplementedError("待成员 A 实现：LLMClient.chat 模型调用")


__all__ = ["LLMClient", "LLMResponse", "ToolCall", "Message"]
