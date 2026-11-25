"""Fixed Top-K 实验：检索后固定选择 K 个工具。

实验组 1：根据任务描述检索工具，并固定选择前 K 个（K 由实验配置决定），
用于观察「一定程度的工具筛选」带来的收益。

负责成员：成员 C
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from agent.controller import AgentController, AgentRunResult


def run_fixed_topk(
    agent: "AgentController",
    task: Mapping[str, Any],
    top_k: int = 5,
) -> "AgentRunResult":
    """使用固定 Top-K 工具执行任务。

    Args:
        agent: 已装配好 retriever / selector 的 AgentController。
        task: benchmark 任务字典。
        top_k: 固定选择的工具数量。

    Returns:
        AgentRunResult: 执行结果。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：调用 agent.run(task, mode="fixed_topk")，
    # top_k 通过 selector.config / ExperimentConfig 传入并记录到结果中。
    raise NotImplementedError("待成员 C 实现：run_fixed_topk Fixed Top-K 实验入口")


__all__ = ["run_fixed_topk"]
