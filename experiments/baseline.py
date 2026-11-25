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

    Args:
        agent: 已装配好的 AgentController。
        task: benchmark 任务字典（见 benchmark/tasks.json）。

    Returns:
        AgentRunResult: 执行结果。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：调用 agent.run(task, mode="baseline")，
    # 并保证工具选择结果为 ToolRegistry 的全部工具。
    raise NotImplementedError("待成员 C 实现：run_baseline Baseline 实验入口")


__all__ = ["run_baseline"]
