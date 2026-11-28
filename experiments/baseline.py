"""Baseline 实验：向模型提供全部工具。

对照组定义：不做任何工具筛选，每次对话都携带工具库中的完整工具列表，
用于衡量「动态选择」相对「全量提供」在成功率 / Token / 效率上的差异。

负责成员：成员 C
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from agent.controller import AgentController, AgentRunResult


def run_baseline(
    agent: "AgentController",
    task: Mapping[str, Any],
) -> "AgentRunResult":
    """使用全部工具执行任务。

    工具筛选完全由 ``AgentController`` 的 baseline 分支决定（即
    ``ToolRegistry.tool_names``），本入口不额外干预，保证三种模式共用同一套执行逻辑。

    Args:
        agent: 已装配好的 AgentController。
        task: benchmark 任务字典（见 benchmark/tasks.json），也接受 TaskInfo 或字符串。

    Returns:
        AgentRunResult: 执行结果，``selected_tools`` 为工具库全部工具。
    """
    return agent.run(task, mode="baseline")


__all__ = ["run_baseline"]