"""实验指标计算接口。

本模块只定义评价函数签名，不实现统计逻辑。
除任务成功率外，工具集合质量同时用 Recall / Precision 衡量；
注意 Expected Tools 仅用于辅助评价工具检索模块，不作为任务成功的唯一依据。

负责成员：成员 C（指标实现）、成员 D（汇总与可视化）
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Union

if TYPE_CHECKING:
    from agent.controller import AgentRunResult

    from .schemas import ExperimentResult


def evaluate_task_success(
    result: Union["ExperimentResult", "AgentRunResult"],
    task: Optional[Mapping[str, Any]] = None,
) -> bool:
    """判断单个任务是否成功。

    Args:
        result: 单次运行结果。
        task: 任务定义，包含 success_criteria 等字段（可选）。

    Returns:
        任务是否成功。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：结合测试结果 / success_criteria 判定任务成功，
    # 而不是仅比对 Expected Tools。
    raise NotImplementedError("待成员 C 实现：evaluate_task_success 成功判定")


def calculate_token_usage(
    result: Union["ExperimentResult", "AgentRunResult"],
) -> Dict[str, int]:
    """统计单次任务的 Token 消耗。

    Returns:
        形如 ``{"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}``。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：从运行结果汇总 Token；如涉及多轮对话需先累加再返回。
    raise NotImplementedError("待成员 C 实现：calculate_token_usage Token 统计")


def calculate_execution_time(
    result: Union["ExperimentResult", "AgentRunResult"],
) -> float:
    """统计单次任务的执行耗时（秒）。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：返回执行耗时；必要时细化到每轮工具调用耗时。
    raise NotImplementedError("待成员 C 实现：calculate_execution_time 耗时统计")


def calculate_tool_call_count(
    result: Union["ExperimentResult", "AgentRunResult"],
) -> int:
    """统计单次任务的工具调用次数。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：统计工具调用次数（可区分成功 / 失败调用）。
    raise NotImplementedError("待成员 C 实现：calculate_tool_call_count 调用次数统计")


def calculate_tool_recall(
    selected_tools: Sequence[str],
    expected_tools: Iterable[str],
) -> float:
    """计算工具召回率：required_tools 中被选中的比例。

    Args:
        selected_tools: 实际提供给模型的工具名。
        expected_tools: 人工标注的 required_tools。

    Returns:
        召回率，取值 [0, 1]；expected_tools 为空时行为由实现约定。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：recall = |selected ∩ required| / |required|。
    raise NotImplementedError("待成员 C 实现：calculate_tool_recall 工具召回率")


def calculate_tool_precision(
    selected_tools: Sequence[str],
    expected_tools: Iterable[str],
) -> float:
    """计算工具精确率：选中工具中属于 required ∪ optional 的比例。

    Args:
        selected_tools: 实际提供给模型的工具名。
        expected_tools: 人工标注的 required_tools ∪ optional_tools。

    Returns:
        精确率，取值 [0, 1]；selected_tools 为空时行为由实现约定。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 C)：precision = |selected ∩ expected| / |selected|。
    raise NotImplementedError("待成员 C 实现：calculate_tool_precision 工具精确率")


__all__ = [
    "evaluate_task_success",
    "calculate_token_usage",
    "calculate_execution_time",
    "calculate_tool_call_count",
    "calculate_tool_recall",
    "calculate_tool_precision",
]
