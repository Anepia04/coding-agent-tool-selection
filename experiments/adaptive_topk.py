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

    K 由 ``AdaptiveSelector`` 依据 ``task.complexity`` 决定（simple→3 / medium→5 /
    complex→7，并夹在 ``SelectionConfig`` 的 [min_k, max_k] 内），本入口不覆盖该取值；
    实际 K 即 ``len(result.selected_tools)``，由实验编排层落盘供成员 D 统计。

    Args:
        agent: 已装配好 retriever / selector 的 AgentController。
        task: benchmark 任务字典（complexity 字段决定 K）。

    Returns:
        AgentRunResult: 执行结果，其中 selected_tools 长度即自适应得到的 K。

    Raises:
        ValueError: agent 缺少 retriever / selector，或任务 complexity 非法。
    """
    return agent.run(task, mode="adaptive_topk")


__all__ = ["run_adaptive_topk"]