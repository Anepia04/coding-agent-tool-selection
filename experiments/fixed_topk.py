"""Fixed Top-K 实验：检索后固定选择 K 个工具。

实验组 1：根据任务描述检索工具，并固定选择前 K 个（K 由实验配置决定），
用于观察「一定程度的工具筛选」带来的收益。

负责成员：成员 C
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from retrieval.adaptive_selector import AdaptiveSelector

if TYPE_CHECKING:
    from agent.controller import AgentController, AgentRunResult


def run_fixed_topk(
    agent: "AgentController",
    task: Mapping[str, Any],
    top_k: int = 5,
) -> "AgentRunResult":
    """使用固定 Top-K 工具执行任务。

    K 由本入口的参数决定：临时把 ``agent.selector`` 换成 ``top_k`` 生效的选择器，
    运行结束后恢复原选择器，避免污染同一 agent 上的其他实验组。

    Args:
        agent: 已装配好 retriever / selector 的 AgentController。
        task: benchmark 任务字典。
        top_k: 固定选择的工具数量，必须为正整数。

    Returns:
        AgentRunResult: 执行结果，``selected_tools`` 长度不超过 top_k。

    Raises:
        ValueError: top_k 非正整数，或 agent 缺少 retriever / selector。
    """
    if top_k <= 0:
        raise ValueError(f"top_k 必须为正整数，当前为 {top_k!r}")

    selector = agent.selector
    if selector is None or selector.config.top_k == top_k:
        return agent.run(task, mode="fixed_topk")

    agent.selector = AdaptiveSelector(replace(selector.config, top_k=top_k))
    try:
        return agent.run(task, mode="fixed_topk")
    finally:
        agent.selector = selector


__all__ = ["run_fixed_topk"]