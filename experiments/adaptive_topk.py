"""Adaptive Top-K 实验：按任务复杂度动态确定工具数量。

实验组 2：在检索结果基础上，由 AdaptiveSelector 根据任务复杂度决定最终
提供给模型的工具数量（简单任务少给、复杂任务多给），用于验证「任务感知」
相对「固定 K」的优势。

负责成员：成员 C
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from agent.controller import AgentController, AgentRunResult


def run_adaptive_topk(
    agent: "AgentController",
    task: Mapping[str, Any],
) -> "AgentRunResult":
    """使用自适应 Top-K 工具执行任务。

    Args:
        agent: 已装配好 retriever / selector 的 AgentController。
        task: benchmark 任务字典。

    Returns:
        AgentRunResult: 执行结果，其中 selected_tools 长度即自适应得到的 K。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：调用 agent.run(task, mode="adaptive_topk")，
    # 并记录每次任务实际选择的工具数量，供成员 D 统计。
    raise NotImplementedError("待成员 C 实现：run_adaptive_topk Adaptive Top-K 实验入口")


__all__ = ["run_adaptive_topk"]
