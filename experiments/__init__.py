"""实验运行与评价包。

结构：
- baseline.py        Baseline：提供全部工具
- fixed_topk.py      Fixed Top-K：检索后固定取 K 个工具
- adaptive_topk.py   Adaptive Top-K：按任务复杂度动态确定工具数量
- evaluate.py        六项指标计算与任务成功判定
- workspace.py       实验工作副本的复制 / 重置 / 快照 / 测试执行
- runner.py          对照实验编排、结果落盘与命令行入口（按需从 experiments.runner 导入）
- schemas.py         统一实验数据结构（ExperimentConfig / ExperimentResult）

负责成员：成员 C（实验设计）、成员 D（统计与可视化）
"""

from .adaptive_topk import run_adaptive_topk
from .baseline import run_baseline
from .evaluate import (
    calculate_execution_time,
    calculate_token_usage,
    calculate_tool_call_count,
    calculate_tool_precision,
    calculate_tool_recall,
    evaluate_task_success,
)
from .fixed_topk import run_fixed_topk
from .schemas import RESULT_CSV_FIELDS, ExperimentConfig, ExperimentResult

__all__ = [
    "run_baseline",
    "run_fixed_topk",
    "run_adaptive_topk",
    "evaluate_task_success",
    "calculate_token_usage",
    "calculate_execution_time",
    "calculate_tool_call_count",
    "calculate_tool_recall",
    "calculate_tool_precision",
    "ExperimentConfig",
    "ExperimentResult",
    "RESULT_CSV_FIELDS",
]