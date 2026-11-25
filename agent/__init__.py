"""Coding Agent 核心框架。

结构：
- llm_client.py   LLM 调用封装与统一响应结构（LLMResponse / ToolCall）
- task_analyzer.py 任务结构化分析（TaskInfo）
- controller.py   Agent 执行流程控制中心（AgentController / AgentRunResult）

负责成员：成员 A
"""

from .controller import AgentController, AgentRunResult, AgentStep
from .llm_client import LLMClient, LLMResponse, ToolCall
from .task_analyzer import TaskAnalyzer, TaskInfo

__all__ = [
    "LLMClient",
    "LLMResponse",
    "ToolCall",
    "TaskAnalyzer",
    "TaskInfo",
    "AgentController",
    "AgentRunResult",
    "AgentStep",
]
