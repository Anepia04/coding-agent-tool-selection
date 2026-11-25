"""实验统一数据结构。

ExperimentConfig 描述「一次实验如何运行」，ExperimentResult 描述「一次实验
跑完之后要记录的指标」。二者是 Controller 运行结果与结果文件（CSV）之间的
桥梁，字段定义必须与 docs/architecture.md 保持一致。

负责成员：成员 A（结构定义）、成员 C（使用）、成员 D（落盘统计）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import EXPERIMENT_MODES

logger = logging.getLogger(__name__)

#: raw_results.csv 的列顺序（与 ExperimentResult 字段一一对应）
RESULT_CSV_FIELDS: tuple[str, ...] = (
    "task_id",
    "experiment_mode",
    "task_success",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "execution_time",
    "tool_call_count",
    "selected_tools",
    "error_message",
)


@dataclass
class ExperimentConfig:
    """单次实验配置。

    Attributes:
        mode: 实验模式，取值见 ``config.EXPERIMENT_MODES``。
        task_id: 本次实验执行的任务 ID。
        top_k: Fixed Top-K 模式的 K 值；其他模式为 None。
        max_steps: Agent 最大执行轮数。
    """

    mode: str
    task_id: str
    top_k: Optional[int] = None
    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.mode not in EXPERIMENT_MODES:
            raise ValueError(f"mode 必须是 {EXPERIMENT_MODES} 之一，当前为 {self.mode!r}")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k 必须为正整数或 None")
        if self.max_steps <= 0:
            raise ValueError("max_steps 必须为正整数")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "task_id": self.task_id,
            "top_k": self.top_k,
            "max_steps": self.max_steps,
        }


@dataclass
class ExperimentResult:
    """单次实验的可记录结果。

    Attributes:
        task_id: 任务 ID。
        experiment_mode: 实验模式（baseline / fixed_topk / adaptive_topk）。
        task_success: 任务是否成功完成。
        input_tokens: 输入 Token 总数。
        output_tokens: 输出 Token 总数。
        total_tokens: Token 总数。
        execution_time: 执行耗时（秒）。
        tool_call_count: 工具调用次数。
        selected_tools: 提供给模型的工具名称列表。
        error_message: 失败原因，成功时为 None。
    """

    task_id: str
    experiment_mode: str
    task_success: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    execution_time: float = 0.0
    tool_call_count: int = 0
    selected_tools: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "task_id": self.task_id,
            "experiment_mode": self.experiment_mode,
            "task_success": self.task_success,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "execution_time": self.execution_time,
            "tool_call_count": self.tool_call_count,
            "selected_tools": list(self.selected_tools),
            "error_message": self.error_message,
        }


__all__ = ["ExperimentConfig", "ExperimentResult", "RESULT_CSV_FIELDS"]
