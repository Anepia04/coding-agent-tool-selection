"""三组对照实验的编排入口。

职责（成员 C）：
1. 读取 benchmark 任务与 Expected Tools 标注；
2. 为「任务 × 模式 × 重复序号」复制独立工作副本，保证各组初始状态一致；
3. 通过三个 ``run_*`` 入口驱动同一个 AgentController，唯一差异是工具提供策略；
4. 用 ``evaluate.py`` 的六项指标评价每次运行，写出 ``results/raw_results.csv``
   与 ``results/summary.csv``（供成员 D 统计与可视化）。

本模块只消费 ``AgentRunResult``，不绕过 AgentController 直接调用工具。

用法::

    python -m experiments.runner                       # 全部任务 × 三种模式
    python -m experiments.runner --task-id task_001    # 只跑指定任务
    python -m experiments.runner --modes baseline fixed_topk --repeats 3

需要真实 API Key（``.env`` 中的 ``LLM_API_KEY``）；离线验证请用 unittest 中的 Mock。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agent.controller import AgentController, AgentRunResult
from agent.llm_client import LLMClient
from agent.task_analyzer import TaskAnalyzer
from config import EXPERIMENT_MODES, Settings, get_settings
from retrieval.adaptive_selector import AdaptiveSelector, SelectionConfig
from retrieval.tfidf_retriever import TFIDFRetriever
from tools.implementations import configure_workspace
from tools.registry import build_default_registry

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
from .workspace import (
    DEFAULT_WORK_ROOT,
    prepare_workspace,
    resolve_path,
    task_source_dir,
    workspace_ready,
)

logger = logging.getLogger(__name__)

#: summary.csv 的列顺序（按实验模式聚合）
SUMMARY_CSV_FIELDS: Tuple[str, ...] = (
    "experiment_mode",
    "task_count",
    "success_count",
    "success_rate",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_total_tokens",
    "avg_execution_time",
    "avg_tool_call_count",
    "avg_selected_tools",
    "avg_tool_recall",
    "avg_tool_precision",
)


@dataclass(frozen=True)
class RunnerConfig:
    """一轮对照实验的运行配置。

    Attributes:
        modes: 参与对照的实验模式，默认三种全跑。
        top_k: Fixed Top-K 的 K，为空时取 ``settings.retrieval.top_k``。
        max_steps: Agent 最大执行轮数，为空时取 ``settings.agent.max_steps``。
        repeats: 每个「任务 × 模式」的重复次数（每次都从干净副本开始）。
        work_root: 工作副本根目录，为空时使用 ``workspace.DEFAULT_WORK_ROOT``。
        keep_workspaces: 运行结束后是否保留副本（便于人工复核改动）。
        tasks_file / expected_tools_file: benchmark 数据文件路径。
        raw_results_file / summary_file: 结果输出路径。
    """

    modes: Tuple[str, ...] = EXPERIMENT_MODES
    top_k: Optional[int] = None
    max_steps: Optional[int] = None
    repeats: int = 1
    work_root: Optional[Path] = None
    keep_workspaces: bool = False
    tasks_file: Optional[Path] = None
    expected_tools_file: Optional[Path] = None
    raw_results_file: Optional[Path] = None
    summary_file: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("modes 不能为空")
        unknown = [mode for mode in self.modes if mode not in EXPERIMENT_MODES]
        if unknown:
            raise ValueError(f"不支持的实验模式: {unknown}，可选值: {EXPERIMENT_MODES}")
        if self.repeats <= 0:
            raise ValueError("repeats 必须为正整数")


@dataclass
class RunRecord:
    """一次运行的完整评价记录。

    ``result`` 是落盘到 raw_results.csv 的部分（字段契约由成员 A 维护）；
    Recall / Precision 与终止原因只进入 summary.csv 与日志，不改动 raw 列。
    """

    result: ExperimentResult
    tool_recall: float = 0.0
    tool_precision: float = 0.0
    termination_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# 数据加载与 Agent 装配
# ---------------------------------------------------------------------------

def load_tasks(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读取 benchmark 任务清单。"""
    source = resolve_path(path) if path is not None else get_settings().benchmark.tasks_file
    data = json.loads(source.read_text(encoding="utf-8"))
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        raise ValueError(f"任务文件结构不合法（应为任务数组）: {source}")
    return [dict(task) for task in tasks]


def load_expected_tools(path: Optional[Path] = None) -> Dict[str, Dict[str, List[str]]]:
    """读取 Expected Tools 标注，返回 ``task_id -> {required_tools, optional_tools}``。"""
    source = (
        resolve_path(path) if path is not None else get_settings().benchmark.expected_tools_file
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    items = data.get("expected_tools", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Expected Tools 文件结构不合法（应为数组）: {source}")
    return {
        str(item["task_id"]): {
            "required_tools": list(item.get("required_tools") or []),
            "optional_tools": list(item.get("optional_tools") or []),
        }
        for item in items
    }


def build_agent(
    settings: Optional[Settings] = None,
    llm_client: Optional[LLMClient] = None,
) -> AgentController:
    """按统一配置装配 AgentController（三种模式共用同一个实例）。"""
    settings = settings or get_settings()
    registry = build_default_registry()
    retriever = TFIDFRetriever().fit(registry.get_all_tools())
    selector = AdaptiveSelector(
        SelectionConfig(
            top_k=settings.retrieval.top_k,
            min_k=settings.retrieval.adaptive_min_k,
            max_k=settings.retrieval.adaptive_max_k,
        )
    )
    client = llm_client or LLMClient(
        model=settings.model.model,
        api_key=settings.model.api_key,
        base_url=settings.model.base_url,
        temperature=settings.model.temperature,
        max_tokens=settings.model.max_tokens,
    )
    return AgentController(
        llm_client=client,
        task_analyzer=TaskAnalyzer(client),
        tool_registry=registry,
        retriever=retriever,
        selector=selector,
        default_max_steps=settings.agent.max_steps,
    )


# ---------------------------------------------------------------------------
# 实验执行
# ---------------------------------------------------------------------------

def run_experiment(
    config: Optional[RunnerConfig] = None,
    agent: Optional[AgentController] = None,
    tasks: Optional[Sequence[Mapping[str, Any]]] = None,
    expected_tools: Optional[Mapping[str, Mapping[str, Sequence[str]]]] = None,
) -> List[RunRecord]:
    """运行一轮对照实验，返回每次运行的评价记录。

    初始工作区尚未就绪的任务（例如只有 ``.gitkeep`` 占位）会被跳过并告警，
    不计入成功率，避免把「数据缺失」混同于「任务失败」。

    Args:
        config: 运行配置，为空时使用默认值。
        agent: 已装配的 AgentController；为空时按配置装配（需要真实 API Key）。
        tasks: 任务列表，为空时从 ``config.tasks_file`` 读取。
        expected_tools: Expected Tools 标注，为空时从 ``config.expected_tools_file`` 读取。

    Returns:
        按「任务 → 模式 → 重复序号」排列的 RunRecord 列表。
    """
    config = config or RunnerConfig()
    settings = get_settings()
    top_k = config.top_k if config.top_k is not None else settings.retrieval.top_k
    max_steps = config.max_steps if config.max_steps is not None else settings.agent.max_steps
    task_list = list(tasks) if tasks is not None else load_tasks(config.tasks_file)
    annotations = (
        dict(expected_tools)
        if expected_tools is not None
        else load_expected_tools(config.expected_tools_file)
    )
    controller = agent if agent is not None else build_agent(settings)

    records: List[RunRecord] = []
    for task in task_list:
        task_id = str(task.get("task_id", ""))
        source = task_source_dir(task)
        if not workspace_ready(source):
            logger.warning("跳过任务 %s：初始工作区尚未就绪（%s）", task_id, source)
            continue

        try:
            frozen = _freeze_task(task, controller)
        except Exception as exc:
            logger.error("任务 %s 无法确定统一任务语义: %s", task_id, exc)
            records.extend(_error_records(task_id, config, top_k, max_steps, exc))
            continue

        annotation = annotations.get(task_id) or {}
        required = list(annotation.get("required_tools") or [])
        acceptable = required + list(annotation.get("optional_tools") or [])
        if not required:
            logger.warning("任务 %s 缺少 required_tools 标注，Recall 按约定记为 1.0", task_id)

        for mode in config.modes:
            for repeat in range(1, config.repeats + 1):
                records.append(
                    _run_once(
                        controller, frozen, mode, repeat,
                        top_k=top_k, max_steps=max_steps, config=config,
                        required=required, acceptable=acceptable,
                    )
                )
    return records


def _run_once(
    agent: AgentController,
    task: Mapping[str, Any],
    mode: str,
    repeat: int,
    *,
    top_k: int,
    max_steps: int,
    config: RunnerConfig,
    required: Sequence[str],
    acceptable: Sequence[str],
) -> RunRecord:
    """在独立工作副本上运行一次「任务 × 模式」，并计算全部指标。"""
    task_id = str(task.get("task_id", ""))
    run_config = ExperimentConfig(
        mode=mode,
        task_id=task_id,
        top_k=top_k if mode == "fixed_topk" else None,
        max_steps=max_steps,
    )
    workspace = prepare_workspace(task, config.work_root, mode=mode, repeat=repeat)
    agent.default_max_steps = run_config.max_steps

    configure_workspace(str(workspace))
    try:
        run_result = _dispatch(agent, task, run_config)
    finally:
        configure_workspace(None)

    success = evaluate_task_success(run_result, dict(task, workspace_dir=str(workspace)))
    usage = calculate_token_usage(run_result)
    result = ExperimentResult(
        task_id=task_id or run_result.task_id,
        experiment_mode=mode,
        task_success=success,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        total_tokens=usage["total_tokens"],
        execution_time=round(calculate_execution_time(run_result), 4),
        tool_call_count=calculate_tool_call_count(run_result),
        selected_tools=list(run_result.selected_tools),
        error_message=run_result.error_message,
    )
    record = RunRecord(
        result=result,
        tool_recall=calculate_tool_recall(result.selected_tools, required),
        tool_precision=calculate_tool_precision(result.selected_tools, acceptable),
        termination_reason=run_result.termination_reason,
    )
    logger.info(
        "%s/%s r%d: success=%s tools=%d calls=%d tokens=%d time=%.2fs recall=%.2f precision=%.2f (%s)",
        task_id, mode, repeat, success, len(result.selected_tools), result.tool_call_count,
        result.total_tokens, result.execution_time, record.tool_recall, record.tool_precision,
        record.termination_reason,
    )

    if not config.keep_workspaces:
        shutil.rmtree(workspace, ignore_errors=True)
    return record


def _dispatch(
    agent: AgentController,
    task: Mapping[str, Any],
    run_config: ExperimentConfig,
) -> AgentRunResult:
    """按模式调用对应实验入口。"""
    if run_config.mode == "baseline":
        return run_baseline(agent, task)
    if run_config.mode == "fixed_topk":
        return run_fixed_topk(agent, task, top_k=run_config.top_k)
    return run_adaptive_topk(agent, task)


def _freeze_task(task: Mapping[str, Any], agent: AgentController) -> Dict[str, Any]:
    """冻结任务语义：三种模式与多次重复共用同一份分析结果。

    预标注字段应由 ``scripts/preanalyze_tasks.py`` 提前写入 tasks.json；
    缺失时在这里只调用一次 TaskAnalyzer 补齐，而不是让每种模式各自分析。
    """
    frozen = dict(task)
    if str(frozen.get("retrieval_query") or "").strip():
        return frozen

    task_id = str(frozen.get("task_id", ""))
    logger.warning(
        "任务 %s 缺少预标注 retrieval_query，本次运行临时分析补齐；"
        "正式实验前请先执行 scripts/preanalyze_tasks.py 并人工复核",
        task_id,
    )
    analyzed = agent.task_analyzer.analyze(str(frozen.get("description") or ""), task_id=task_id)
    frozen["retrieval_query"] = analyzed.retrieval_query
    if not frozen.get("negative_constraints"):
        frozen["negative_constraints"] = list(analyzed.negative_constraints)
    for field in ("task_type", "language", "complexity"):
        if not frozen.get(field):
            frozen[field] = getattr(analyzed, field)
    return frozen


def _error_records(
    task_id: str,
    config: RunnerConfig,
    top_k: int,
    max_steps: int,
    exc: Exception,
) -> List[RunRecord]:
    """任务无法运行时，为每个「模式 × 重复」留下一条失败记录。"""
    message = f"{type(exc).__name__}: {exc}"
    records: List[RunRecord] = []
    for mode in config.modes:
        for _ in range(config.repeats):
            records.append(
                RunRecord(
                    result=ExperimentResult(
                        task_id=task_id,
                        experiment_mode=mode,
                        task_success=False,
                        error_message=message,
                    ),
                    termination_reason="error",
                )
            )
    return records


# ---------------------------------------------------------------------------
# 结果落盘
# ---------------------------------------------------------------------------

def write_raw_results(
    records: Sequence[RunRecord],
    path: Optional[Path] = None,
) -> Path:
    """写出每次运行的原始记录（列顺序遵循 ``RESULT_CSV_FIELDS``）。

    ``selected_tools`` 以 JSON 数组字符串落盘，读取时 ``json.loads`` 即可还原。
    """
    target = resolve_path(path) if path is not None else get_settings().results.raw_results_file
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_CSV_FIELDS))
        writer.writeheader()
        for record in records:
            row = record.result.to_dict()
            row["selected_tools"] = json.dumps(row["selected_tools"], ensure_ascii=False)
            row["execution_time"] = round(float(row["execution_time"]), 4)
            writer.writerow(row)
    logger.info("已写出 %d 条原始结果: %s", len(records), target)
    return target


def write_summary(
    records: Sequence[RunRecord],
    path: Optional[Path] = None,
) -> Path:
    """按实验模式聚合，写出汇总结果（列顺序见 ``SUMMARY_CSV_FIELDS``）。"""
    groups: Dict[str, List[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.result.experiment_mode, []).append(record)

    target = resolve_path(path) if path is not None else get_settings().results.summary_file
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_CSV_FIELDS))
        writer.writeheader()
        for mode in EXPERIMENT_MODES:
            group = groups.get(mode)
            if not group:
                continue
            writer.writerow(_summary_row(mode, group))
    logger.info("已写出 %d 个模式的汇总结果: %s", len(groups), target)
    return target


def _summary_row(mode: str, group: Sequence[RunRecord]) -> Dict[str, Any]:
    """计算单个模式的聚合行。"""
    results = [record.result for record in group]
    count = len(results)
    success_count = sum(1 for result in results if result.task_success)
    return {
        "experiment_mode": mode,
        "task_count": count,
        "success_count": success_count,
        "success_rate": round(success_count / count, 4),
        "avg_input_tokens": _mean(result.input_tokens for result in results),
        "avg_output_tokens": _mean(result.output_tokens for result in results),
        "avg_total_tokens": _mean(result.total_tokens for result in results),
        "avg_execution_time": _mean(result.execution_time for result in results),
        "avg_tool_call_count": _mean(result.tool_call_count for result in results),
        "avg_selected_tools": _mean(len(result.selected_tools) for result in results),
        "avg_tool_recall": _mean(record.tool_recall for record in group),
        "avg_tool_precision": _mean(record.tool_precision for record in group),
    }


def _mean(values: Iterable[float]) -> float:
    """均值（保留 4 位小数）；空序列返回 0.0。"""
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _print_summary(records: Sequence[RunRecord]) -> None:
    """在终端打印按模式聚合的对照表。"""
    groups: Dict[str, List[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.result.experiment_mode, []).append(record)

    print()
    print(
        f"{'mode':<15}{'runs':<6}{'success':<9}{'tokens':<10}"
        f"{'time(s)':<10}{'calls':<8}{'tools':<8}{'recall':<9}{'precision'}"
    )
    for mode in EXPERIMENT_MODES:
        group = groups.get(mode)
        if not group:
            continue
        row = _summary_row(mode, group)
        print(
            f"{mode:<15}{row['task_count']:<6}{row['success_rate']:<9.2%}"
            f"{row['avg_total_tokens']:<10.1f}{row['avg_execution_time']:<10.2f}"
            f"{row['avg_tool_call_count']:<8.2f}{row['avg_selected_tools']:<8.2f}"
            f"{row['avg_tool_recall']:<9.3f}{row['avg_tool_precision']:.3f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：``python -m experiments.runner``。

    退出码：0 成功；2 缺少 API Key；3 没有可运行的任务。
    """
    parser = argparse.ArgumentParser(description="运行三组对照实验并写出结果 CSV")
    parser.add_argument("--modes", nargs="+", choices=list(EXPERIMENT_MODES),
                        default=list(EXPERIMENT_MODES), help="参与对照的实验模式")
    parser.add_argument("--task-id", nargs="+", default=None, help="只运行指定任务 ID")
    parser.add_argument("--tasks", type=Path, default=None, help="任务清单路径")
    parser.add_argument("--expected-tools", type=Path, default=None, help="Expected Tools 路径")
    parser.add_argument("--top-k", type=int, default=None, help="Fixed Top-K 的 K")
    parser.add_argument("--max-steps", type=int, default=None, help="Agent 最大执行轮数")
    parser.add_argument("--repeats", type=int, default=1, help="每个任务×模式的重复次数")
    parser.add_argument("--work-root", type=Path, default=None, help="工作副本根目录")
    parser.add_argument("--keep-workspaces", action="store_true", help="保留工作副本便于复核")
    parser.add_argument("--raw-out", type=Path, default=None, help="raw_results.csv 输出路径")
    parser.add_argument("--summary-out", type=Path, default=None, help="summary.csv 输出路径")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if not settings.model.api_key:
        print("错误：缺少 LLM_API_KEY，请在项目根目录 .env 中配置后重试。", file=sys.stderr)
        return 2

    config = RunnerConfig(
        modes=tuple(args.modes),
        top_k=args.top_k,
        max_steps=args.max_steps,
        repeats=args.repeats,
        work_root=args.work_root,
        keep_workspaces=args.keep_workspaces,
        tasks_file=args.tasks,
        expected_tools_file=args.expected_tools,
        raw_results_file=args.raw_out,
        summary_file=args.summary_out,
    )

    tasks = load_tasks(config.tasks_file)
    if args.task_id:
        wanted = set(args.task_id)
        unknown = wanted - {str(task.get("task_id")) for task in tasks}
        if unknown:
            print(f"错误：任务清单中不存在这些 task_id: {sorted(unknown)}", file=sys.stderr)
            return 3
        tasks = [task for task in tasks if str(task.get("task_id")) in wanted]

    print(f"模型: {settings.model.model}；模式: {', '.join(config.modes)}；"
          f"K={config.top_k or settings.retrieval.top_k}；"
          f"max_steps={config.max_steps or settings.agent.max_steps}；repeats={config.repeats}")
    records = run_experiment(config, tasks=tasks)
    if not records:
        print("没有可运行的任务：初始工作区尚未就绪或任务被过滤。", file=sys.stderr)
        return 3

    raw_path = write_raw_results(records, config.raw_results_file)
    summary_path = write_summary(records, config.summary_file)
    _print_summary(records)
    print()
    print(f"原始结果: {raw_path}")
    print(f"汇总结果: {summary_path}")
    if config.keep_workspaces:
        print(f"工作副本保留在: {resolve_path(config.work_root) if config.work_root else DEFAULT_WORK_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RunnerConfig",
    "RunRecord",
    "SUMMARY_CSV_FIELDS",
    "load_tasks",
    "load_expected_tools",
    "build_agent",
    "run_experiment",
    "write_raw_results",
    "write_summary",
    "main",
]