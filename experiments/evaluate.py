"""实验指标计算。

六项指标分两类：
- **执行代价**：Token 消耗、执行耗时、工具调用次数，直接取自运行结果；
- **工具集合质量**：Recall（required_tools 被选中的比例）与 Precision
  （选中工具中属于 required ∪ optional 的比例），只用于评价检索 / 选择模块。

任务成功判定不使用 Expected Tools：Expected Tools 是「人类会怎么做」的标注，
不是任务正确性的依据。``evaluate_task_success`` 以工作副本的客观状态为准：
流程无错误 + 工作区确有改动 + 全部测试通过。

负责成员：成员 C（指标实现）、成员 D（汇总与可视化）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Union

from .workspace import (
    resolve_path,
    run_workspace_tests,
    task_source_dir,
    workspace_changed,
)

if TYPE_CHECKING:
    from agent.controller import AgentRunResult

    from .schemas import ExperimentResult

logger = logging.getLogger(__name__)

#: Token 统计字段
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def evaluate_task_success(
    result: Union["ExperimentResult", "AgentRunResult"],
    task: Optional[Mapping[str, Any]] = None,
) -> bool:
    """判断单个任务是否成功。

    判定规则（依次短路）：

    1. 流程失败（``error_message`` 非空、``termination_reason == "error"`` 或
       ``task_success`` 为假）直接判失败；
    2. 未提供 ``task`` 时无法核验代码状态，退化为流程级结论；
    3. 提供 ``task`` 时，在待评价工作区（``task["workspace_dir"]``，缺省回退到
       ``project_path``）上核验：相对初始工作区确有改动，且全部测试通过
       （``run_tests`` 的 ``ok`` 为真且确实发现了测试）。

    ``success_criteria`` 中超出测试覆盖范围的语义要求（如「函数复杂度下降」）
    无法自动判定，需人工复核；本函数只给出可自动核验的客观部分。

    Args:
        result: 单次运行结果（AgentRunResult 或 ExperimentResult）。
        task: 任务定义，可含 ``workspace_dir`` / ``project_path`` / ``success_criteria``。

    Returns:
        任务是否成功。
    """
    if _flow_failed(result):
        return False
    if task is None:
        return True

    workspace = _evaluation_workspace(task)
    if workspace is None or not workspace.is_dir():
        logger.warning("任务 %s 缺少可评价的工作区，判定为失败", task.get("task_id"))
        return False

    initial = task_source_dir(task)
    if initial.is_dir() and initial.resolve() != workspace.resolve():
        if not workspace_changed(workspace, initial):
            logger.info("任务 %s 的工作区未产生任何改动，判定为未完成", task.get("task_id"))
            return False

    try:
        report = run_workspace_tests(workspace)
    except Exception as exc:  # 测试无法执行（路径 / 超时 / 环境异常）视为未通过
        logger.warning("任务 %s 的测试执行失败: %s", task.get("task_id"), exc)
        return False

    success = bool(report.get("ok")) and not report.get("no_tests_found")
    logger.info(
        "任务 %s 测试判定: success=%s passed=%s failed=%s",
        task.get("task_id"), success, report.get("passed"), report.get("failed"),
    )
    return success


def calculate_token_usage(
    result: Union["ExperimentResult", "AgentRunResult"],
) -> Dict[str, int]:
    """统计单次任务的 Token 消耗。

    Returns:
        ``{"input_tokens": ..., "output_tokens": ..., "total_tokens": ...}``；
        运行结果已按轮次累加，total 缺失时按 input + output 补齐。
    """
    usage = {name: int(getattr(result, name, 0) or 0) for name in _TOKEN_FIELDS}
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def calculate_execution_time(
    result: Union["ExperimentResult", "AgentRunResult"],
) -> float:
    """统计单次任务的执行耗时（秒）。"""
    return float(getattr(result, "execution_time", 0.0) or 0.0)


def calculate_tool_call_count(
    result: Union["ExperimentResult", "AgentRunResult"],
) -> int:
    """统计单次任务的工具调用次数（含执行失败的调用）。"""
    return int(getattr(result, "tool_call_count", 0) or 0)


def calculate_tool_recall(
    selected_tools: Sequence[str],
    expected_tools: Iterable[str],
) -> float:
    """计算工具召回率：required_tools 中被选中的比例。

    Args:
        selected_tools: 实际提供给模型的工具名。
        expected_tools: 人工标注的 required_tools。

    Returns:
        ``|selected ∩ required| / |required|``，取值 [0, 1]；
        required 为空时约定返回 1.0（没有必需工具，视为全部满足）。
    """
    required = set(expected_tools)
    if not required:
        return 1.0
    return len(set(selected_tools) & required) / len(required)


def calculate_tool_precision(
    selected_tools: Sequence[str],
    expected_tools: Iterable[str],
) -> float:
    """计算工具精确率：选中工具中属于 required ∪ optional 的比例。

    Args:
        selected_tools: 实际提供给模型的工具名。
        expected_tools: 人工标注的 required_tools ∪ optional_tools。

    Returns:
        ``|selected ∩ expected| / |selected|``，取值 [0, 1]；
        selected 为空时约定返回 0.0（没有提供任何工具，精确率无从谈起）。
    """
    selected = set(selected_tools)
    if not selected:
        return 0.0
    return len(selected & set(expected_tools)) / len(selected)


def _flow_failed(result: Any) -> bool:
    """运行流程本身是否失败（与任务最终正确性无关）。"""
    if getattr(result, "error_message", None):
        return True
    if getattr(result, "termination_reason", None) == "error":
        return True
    return not bool(getattr(result, "task_success", False))


def _evaluation_workspace(task: Mapping[str, Any]) -> Optional[Path]:
    """确定待评价的工作区：优先实验副本，其次任务初始工作区。"""
    for key in ("workspace_dir", "project_path"):
        value = task.get(key)
        if value:
            return resolve_path(value)
    task_id = task.get("task_id")
    return task_source_dir(task) if task_id else None


__all__ = [
    "evaluate_task_success",
    "calculate_token_usage",
    "calculate_execution_time",
    "calculate_tool_call_count",
    "calculate_tool_recall",
    "calculate_tool_precision",
]