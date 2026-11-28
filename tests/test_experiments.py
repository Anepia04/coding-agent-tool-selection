"""实验层测试（成员 C）：三个实验入口、六项指标、工作副本与编排落盘。

全部使用 Mock / 脚本化 LLM 与临时工作区，不调用真实 API、不写正式 results 文件。
运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

from agent.controller import AgentController, AgentRunResult
from agent.llm_client import LLMResponse, ToolCall
from agent.task_analyzer import TaskInfo
from experiments import adaptive_topk, baseline, evaluate, fixed_topk
from experiments.runner import (
    SUMMARY_CSV_FIELDS,
    RunnerConfig,
    run_experiment,
    write_raw_results,
    write_summary,
)
from experiments.schemas import RESULT_CSV_FIELDS, ExperimentResult
from experiments.workspace import (
    prepare_workspace,
    run_workspace_tests,
    snapshot_workspace,
    task_source_dir,
    workspace_changed,
    workspace_ready,
)
from retrieval.adaptive_selector import AdaptiveSelector, SelectionConfig
from retrieval.tfidf_retriever import RetrievedTool
from tools.registry import build_default_registry

INITIAL_CALC = '''"""极简计算器。"""


def divide(a, b):
    return a / b
'''

FIXED_CALC = '''"""极简计算器。"""


def divide(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b
'''

INITIAL_TEST = '''import unittest

import calc


class TestDivide(unittest.TestCase):
    def test_exact_division(self):
        self.assertEqual(calc.divide(6, 3), 2)


if __name__ == "__main__":
    unittest.main()
'''

ZERO_TEST = '''import unittest

import calc


class TestDivideByZero(unittest.TestCase):
    def test_zero_divisor_raises_value_error(self):
        with self.assertRaises(ValueError):
            calc.divide(1, 0)


if __name__ == "__main__":
    unittest.main()
'''

#: 入口测试使用的工具名（顺序即注册顺序）
TOOL_NAMES = ["read_file", "write_file", "search_code", "run_tests", "list_files", "git_diff"]


def task_dict(**overrides: Any) -> Dict[str, Any]:
    """构造一条已预标注的 benchmark 任务。"""
    data = {
        "task_id": "task_001",
        "description": "修复 calculator 的除零异常并补充单元测试",
        "task_type": "bug_fix",
        "language": "python",
        "complexity": "simple",
        "retrieval_query": "除零异常 修复 单元测试",
        "success_criteria": "全部单元测试通过",
    }
    data.update(overrides)
    return data


def build_ranking(names: Sequence[str]) -> List[RetrievedTool]:
    """按给定顺序构造检索结果。"""
    return [
        RetrievedTool(tool_name=name, score=1.0 / (index + 1), rank=index + 1)
        for index, name in enumerate(names)
    ]


def build_controller(
    ranking: Optional[Sequence[str]] = None,
    top_k: int = 5,
    min_k: int = 3,
    max_k: int = 8,
    responses: Optional[Sequence[LLMResponse]] = None,
    with_selector: bool = True,
):
    """装配 Mock 依赖 + 真实 AdaptiveSelector 的 AgentController。"""
    llm = MagicMock()
    if responses is not None:
        llm.chat.side_effect = list(responses)
    else:
        llm.chat.return_value = LLMResponse(content="已完成")

    analyzer = MagicMock()
    registry = MagicMock()
    registry.tool_names = list(TOOL_NAMES)
    registry.get_schemas.side_effect = lambda names=None: [
        {"type": "function", "function": {"name": name}}
        for name in (TOOL_NAMES if names is None else names)
    ]

    retriever = MagicMock()
    retriever.retrieve.return_value = build_ranking(ranking or TOOL_NAMES)
    selector = AdaptiveSelector(SelectionConfig(top_k=top_k, min_k=min_k, max_k=max_k))

    controller = AgentController(
        llm_client=llm,
        task_analyzer=analyzer,
        tool_registry=registry,
        retriever=retriever,
        selector=selector if with_selector else None,
    )
    return controller, {"llm": llm, "analyzer": analyzer, "retriever": retriever, "selector": selector}


def write_project(directory: Path, *, fixed: bool = False, with_zero_test: bool = False) -> None:
    """在目录中写入一个最小 Python 项目（源码 + 测试）。"""
    (directory / "tests").mkdir(parents=True, exist_ok=True)
    (directory / "calc.py").write_text(FIXED_CALC if fixed else INITIAL_CALC, encoding="utf-8")
    (directory / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (directory / "tests" / "test_calc.py").write_text(INITIAL_TEST, encoding="utf-8")
    if with_zero_test:
        (directory / "tests" / "test_zero.py").write_text(ZERO_TEST, encoding="utf-8")


class TempDirTestCase(unittest.TestCase):
    """提供临时根目录的基类。"""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)


# ---------------------------------------------------------------------------
# 三个实验入口
# ---------------------------------------------------------------------------

class TestExperimentEntries(unittest.TestCase):
    """run_baseline / run_fixed_topk / run_adaptive_topk。"""

    def test_baseline_exposes_all_tools(self) -> None:
        controller, mocks = build_controller()

        result = baseline.run_baseline(controller, task_dict())

        self.assertIsInstance(result, AgentRunResult)
        self.assertEqual(result.mode, "baseline")
        self.assertEqual(result.selected_tools, TOOL_NAMES)
        mocks["retriever"].retrieve.assert_not_called()
        mocks["analyzer"].analyze.assert_not_called()

    def test_fixed_topk_applies_requested_k(self) -> None:
        controller, mocks = build_controller(top_k=5)
        original_selector = controller.selector

        result = fixed_topk.run_fixed_topk(controller, task_dict(), top_k=2)

        self.assertEqual(result.mode, "fixed_topk")
        self.assertEqual(result.selected_tools, TOOL_NAMES[:2])
        # 运行结束后恢复原选择器，不污染同一 agent 上的其他实验组
        self.assertIs(controller.selector, original_selector)
        self.assertEqual(controller.selector.config.top_k, 5)
        mocks["retriever"].retrieve.assert_called_once()

    def test_fixed_topk_keeps_selector_when_k_matches(self) -> None:
        controller, _ = build_controller(top_k=3)
        original_selector = controller.selector

        result = fixed_topk.run_fixed_topk(controller, task_dict(), top_k=3)

        self.assertEqual(result.selected_tools, TOOL_NAMES[:3])
        self.assertIs(controller.selector, original_selector)

    def test_fixed_topk_restores_selector_after_failure(self) -> None:
        controller, mocks = build_controller(top_k=5)
        original_selector = controller.selector
        mocks["llm"].chat.side_effect = RuntimeError("api down")

        result = fixed_topk.run_fixed_topk(controller, task_dict(), top_k=2)

        self.assertFalse(result.task_success)
        self.assertIs(controller.selector, original_selector)

    def test_fixed_topk_rejects_non_positive_k(self) -> None:
        controller, _ = build_controller()
        for invalid in (0, -1):
            with self.subTest(top_k=invalid):
                with self.assertRaises(ValueError):
                    fixed_topk.run_fixed_topk(controller, task_dict(), top_k=invalid)

    def test_fixed_topk_without_selector_raises(self) -> None:
        controller, _ = build_controller(with_selector=False)
        with self.assertRaises(ValueError):
            fixed_topk.run_fixed_topk(controller, task_dict(), top_k=2)

    def test_adaptive_topk_follows_complexity(self) -> None:
        expected = {"simple": 3, "medium": 5, "complex": 6}  # complex→7，但候选只有 6 个
        for complexity, count in expected.items():
            with self.subTest(complexity=complexity):
                controller, _ = build_controller()

                result = adaptive_topk.run_adaptive_topk(controller, task_dict(complexity=complexity))

                self.assertEqual(result.mode, "adaptive_topk")
                self.assertEqual(len(result.selected_tools), count)
                self.assertEqual(result.selected_tools, TOOL_NAMES[:count])

    def test_adaptive_topk_is_clamped_by_selection_config(self) -> None:
        controller, _ = build_controller(min_k=4, max_k=5)

        result = adaptive_topk.run_adaptive_topk(controller, task_dict(complexity="simple"))

        self.assertEqual(len(result.selected_tools), 4)

    def test_all_modes_share_preannotated_task(self) -> None:
        for entry in (baseline.run_baseline, fixed_topk.run_fixed_topk, adaptive_topk.run_adaptive_topk):
            with self.subTest(entry=entry.__name__):
                controller, mocks = build_controller()
                entry(controller, task_dict())
                mocks["analyzer"].analyze.assert_not_called()


# ---------------------------------------------------------------------------
# 六项指标
# ---------------------------------------------------------------------------

class TestMetrics(unittest.TestCase):
    """Token / 耗时 / 调用次数 / Recall / Precision。"""

    def test_token_usage_from_run_result(self) -> None:
        result = AgentRunResult(
            task_id="task_001", mode="baseline",
            input_tokens=120, output_tokens=30, total_tokens=150,
        )
        self.assertEqual(
            evaluate.calculate_token_usage(result),
            {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
        )

    def test_token_usage_from_experiment_result(self) -> None:
        result = ExperimentResult(
            task_id="task_001", experiment_mode="fixed_topk",
            input_tokens=10, output_tokens=4, total_tokens=14,
        )
        self.assertEqual(evaluate.calculate_token_usage(result)["total_tokens"], 14)

    def test_token_usage_falls_back_to_sum(self) -> None:
        result = ExperimentResult(
            task_id="task_001", experiment_mode="baseline",
            input_tokens=7, output_tokens=3, total_tokens=0,
        )
        self.assertEqual(evaluate.calculate_token_usage(result)["total_tokens"], 10)

    def test_execution_time_and_call_count(self) -> None:
        result = AgentRunResult(
            task_id="task_001", mode="baseline", execution_time=1.25, tool_call_count=4,
        )
        self.assertEqual(evaluate.calculate_execution_time(result), 1.25)
        self.assertEqual(evaluate.calculate_tool_call_count(result), 4)
        self.assertEqual(evaluate.calculate_execution_time(ExperimentResult("t", "baseline")), 0.0)
        self.assertEqual(evaluate.calculate_tool_call_count(ExperimentResult("t", "baseline")), 0)

    def test_tool_recall(self) -> None:
        required = ["read_file", "write_file", "run_tests"]
        self.assertAlmostEqual(
            evaluate.calculate_tool_recall(["read_file", "write_file"], required), 2 / 3
        )
        self.assertEqual(evaluate.calculate_tool_recall(required, required), 1.0)
        self.assertEqual(evaluate.calculate_tool_recall([], required), 0.0)
        # required 为空：没有必需工具，约定为全部满足
        self.assertEqual(evaluate.calculate_tool_recall(["read_file"], []), 1.0)

    def test_tool_recall_deduplicates(self) -> None:
        self.assertEqual(
            evaluate.calculate_tool_recall(["read_file", "read_file"], ["read_file", "read_file"]),
            1.0,
        )

    def test_tool_precision(self) -> None:
        acceptable = ["read_file", "write_file", "run_tests", "list_files"]
        selected = ["read_file", "write_file", "git_status", "run_command"]
        self.assertEqual(evaluate.calculate_tool_precision(selected, acceptable), 0.5)
        self.assertEqual(evaluate.calculate_tool_precision(["read_file"], acceptable), 1.0)
        # selected 为空：没有提供任何工具，约定为 0.0
        self.assertEqual(evaluate.calculate_tool_precision([], acceptable), 0.0)
        self.assertEqual(evaluate.calculate_tool_precision(["read_file"], []), 0.0)


# ---------------------------------------------------------------------------
# 工作副本
# ---------------------------------------------------------------------------

class TestWorkspace(TempDirTestCase):
    """复制 / 重置 / 快照 / 就绪判定。"""

    def setUp(self) -> None:
        super().setUp()
        self.initial = self.root / "initial"
        write_project(self.initial)
        self.task = task_dict(project_path=str(self.initial))
        self.work_root = self.root / "copies"

    def test_task_source_dir_defaults_to_benchmark_layout(self) -> None:
        source = task_source_dir({"task_id": "task_001"})
        self.assertEqual(source.name, "task_001")
        self.assertEqual(source.parent.name, "workspaces")

    def test_workspace_ready(self) -> None:
        placeholder = self.root / "placeholder"
        placeholder.mkdir()
        (placeholder / ".gitkeep").write_text("", encoding="utf-8")
        self.assertFalse(workspace_ready(placeholder))
        self.assertFalse(workspace_ready(self.root / "missing"))
        self.assertTrue(workspace_ready(self.initial))

    def test_copy_is_isolated_and_resettable(self) -> None:
        copy = prepare_workspace(self.task, self.work_root, mode="baseline", repeat=1)
        self.assertTrue((copy / "calc.py").is_file())

        (copy / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
        self.assertEqual((self.initial / "calc.py").read_text(encoding="utf-8"), INITIAL_CALC)

        again = prepare_workspace(self.task, self.work_root, mode="baseline", repeat=1)
        self.assertEqual(again, copy)
        self.assertEqual((again / "calc.py").read_text(encoding="utf-8"), INITIAL_CALC)

    def test_each_mode_and_repeat_gets_own_copy(self) -> None:
        first = prepare_workspace(self.task, self.work_root, mode="baseline", repeat=1)
        second = prepare_workspace(self.task, self.work_root, mode="fixed_topk", repeat=1)
        third = prepare_workspace(self.task, self.work_root, mode="baseline", repeat=2)
        self.assertEqual(len({first, second, third}), 3)

    def test_prepare_rejects_invalid_repeat_and_missing_source(self) -> None:
        with self.assertRaises(ValueError):
            prepare_workspace(self.task, self.work_root, repeat=0)
        with self.assertRaises(FileNotFoundError):
            prepare_workspace(task_dict(project_path=str(self.root / "missing")), self.work_root)

    def test_snapshot_detects_changes_and_ignores_cache(self) -> None:
        copy = prepare_workspace(self.task, self.work_root)
        self.assertFalse(workspace_changed(copy, self.initial))

        (copy / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
        self.assertTrue(workspace_changed(copy, self.initial))

        cache = copy / "__pycache__"
        cache.mkdir()
        (cache / "calc.pyc").write_bytes(b"cache")
        self.assertFalse(any("__pycache__" in key for key in snapshot_workspace(copy)))

    def test_run_workspace_tests_targets_given_copy(self) -> None:
        first = prepare_workspace(self.task, self.work_root, mode="baseline", repeat=1)
        second = prepare_workspace(self.task, self.work_root, mode="fixed_topk", repeat=1)
        (second / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
        (second / "tests" / "test_zero.py").write_text(ZERO_TEST, encoding="utf-8")

        first_report = run_workspace_tests(first)
        second_report = run_workspace_tests(second)

        # 每次评价都只作用于传入的副本，覆盖不会泄漏到下一次调用
        self.assertTrue(first_report["ok"])
        self.assertEqual(first_report["passed"], 1)
        self.assertTrue(second_report["ok"])
        self.assertEqual(second_report["passed"], 2)


# ---------------------------------------------------------------------------
# 任务成功判定
# ---------------------------------------------------------------------------

class TestEvaluateTaskSuccess(TempDirTestCase):
    """成功判定 = 流程无错误 + 工作区确有改动 + 全部测试通过。"""

    def setUp(self) -> None:
        super().setUp()
        self.initial = self.root / "initial"
        write_project(self.initial)
        self.copy = prepare_workspace(
            task_dict(project_path=str(self.initial)), self.root / "copies"
        )
        self.task = task_dict(
            project_path=str(self.initial), workspace_dir=str(self.copy)
        )
        self.run_result = AgentRunResult(
            task_id="task_001", mode="baseline", task_success=True,
            termination_reason="completed",
        )

    def test_flow_error_short_circuits(self) -> None:
        failed = AgentRunResult(
            task_id="task_001", mode="baseline", task_success=False,
            error_message="LLMClientError: api down", termination_reason="error",
        )
        self.assertFalse(evaluate.evaluate_task_success(failed, self.task))

    def test_without_task_falls_back_to_flow_result(self) -> None:
        self.assertTrue(evaluate.evaluate_task_success(self.run_result))
        self.assertFalse(
            evaluate.evaluate_task_success(ExperimentResult("task_001", "baseline"))
        )

    def test_unchanged_workspace_is_not_success(self) -> None:
        # 初始测试本来就能通过：没有改动不能算完成任务
        self.assertFalse(evaluate.evaluate_task_success(self.run_result, self.task))

    def test_fixed_workspace_is_success(self) -> None:
        (self.copy / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
        (self.copy / "tests" / "test_zero.py").write_text(ZERO_TEST, encoding="utf-8")
        self.assertTrue(evaluate.evaluate_task_success(self.run_result, self.task))

    def test_failing_tests_are_not_success(self) -> None:
        (self.copy / "calc.py").write_text(
            INITIAL_CALC.replace("return a / b", "return a - b"), encoding="utf-8"
        )
        self.assertFalse(evaluate.evaluate_task_success(self.run_result, self.task))

    def test_missing_workspace_is_not_success(self) -> None:
        task = dict(self.task, workspace_dir=str(self.root / "missing"))
        self.assertFalse(evaluate.evaluate_task_success(self.run_result, task))

    def test_falls_back_to_project_path(self) -> None:
        task = task_dict(project_path=str(self.copy))
        (self.copy / "calc.py").write_text(FIXED_CALC, encoding="utf-8")
        (self.copy / "tests" / "test_zero.py").write_text(ZERO_TEST, encoding="utf-8")
        # 待评价目录就是初始目录时无法比对改动，只以测试结果为准
        self.assertTrue(evaluate.evaluate_task_success(self.run_result, task))


# ---------------------------------------------------------------------------
# 编排与落盘
# ---------------------------------------------------------------------------

class FixingLLM:
    """脚本化 LLM：首轮写修复代码与新测试，收到工具结果后给出最终回答。"""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.schema_sizes: List[int] = []

    def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        if self.fail:
            raise RuntimeError("api down")
        self.schema_sizes.append(len(tools or []))
        if any(message.get("role") == "tool" for message in messages):
            return LLMResponse(
                content="已修复除零异常并补充测试", input_tokens=5, output_tokens=2, total_tokens=7
            )
        return LLMResponse(
            tool_calls=[
                ToolCall(id="call_1", name="write_file",
                         arguments={"path": "calc.py", "content": FIXED_CALC, "overwrite": True}),
                ToolCall(id="call_2", name="write_file",
                         arguments={"path": "tests/test_zero.py", "content": ZERO_TEST, "overwrite": True}),
            ],
            input_tokens=100, output_tokens=20, total_tokens=120,
        )


class TestRunner(TempDirTestCase):
    """run_experiment：工作副本隔离、指标计算与 CSV 落盘。"""

    def setUp(self) -> None:
        super().setUp()
        self.initial = self.root / "initial"
        write_project(self.initial)
        self.task = task_dict(project_path=str(self.initial))
        self.expected = {
            "task_001": {
                "required_tools": ["read_file", "write_file", "run_tests"],
                "optional_tools": ["list_files", "git_diff"],
            }
        }
        self.work_root = self.root / "copies"
        self.raw_path = self.root / "raw_results.csv"
        self.summary_path = self.root / "summary.csv"

    def build_agent(self, llm=None, ranking=None):
        registry = build_default_registry()
        names = ranking or ["write_file", "read_file", "run_tests", "search_code", "list_files"]
        remaining = [name for name in registry.tool_names if name not in names]
        retriever = MagicMock()
        retriever.retrieve.return_value = build_ranking(names + remaining)
        analyzer = MagicMock()
        analyzer.analyze.return_value = TaskInfo(
            task_id="task_001", description=self.task["description"],
            task_type="bug_fix", language="python",
            retrieval_query="除零异常 修复 单元测试", complexity="simple",
        )
        return AgentController(
            llm_client=llm if llm is not None else FixingLLM(),
            task_analyzer=analyzer,
            tool_registry=registry,
            retriever=retriever,
            selector=AdaptiveSelector(SelectionConfig(top_k=5, min_k=3, max_k=8)),
        ), analyzer

    def config(self, **overrides) -> RunnerConfig:
        defaults = dict(
            top_k=2, work_root=self.work_root,
            raw_results_file=self.raw_path, summary_file=self.summary_path,
        )
        defaults.update(overrides)
        return RunnerConfig(**defaults)

    def test_three_modes_produce_records_and_csv(self) -> None:
        agent, _ = self.build_agent()

        records = run_experiment(
            self.config(), agent=agent, tasks=[self.task], expected_tools=self.expected
        )

        self.assertEqual(len(records), 3)
        self.assertEqual(
            [record.result.experiment_mode for record in records],
            ["baseline", "fixed_topk", "adaptive_topk"],
        )
        for record in records:
            self.assertTrue(record.result.task_success, record.result.error_message)
            self.assertEqual(record.result.total_tokens, 127)
            self.assertEqual(record.result.tool_call_count, 2)
            self.assertEqual(record.termination_reason, "completed")

        total_tools = len(build_default_registry())
        self.assertEqual(len(records[0].result.selected_tools), total_tools)
        self.assertEqual(len(records[1].result.selected_tools), 2)
        self.assertEqual(len(records[2].result.selected_tools), 3)  # simple -> 3
        self.assertAlmostEqual(records[0].tool_recall, 1.0)
        self.assertAlmostEqual(records[1].tool_recall, 2 / 3)
        self.assertAlmostEqual(records[2].tool_recall, 1.0)
        self.assertAlmostEqual(records[0].tool_precision, 5 / total_tools)
        self.assertAlmostEqual(records[1].tool_precision, 1.0)

        raw_path = write_raw_results(records, self.raw_path)
        with open(raw_path, encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0].keys()), RESULT_CSV_FIELDS)
        self.assertEqual(json.loads(rows[1]["selected_tools"]), ["write_file", "read_file"])
        self.assertEqual(rows[0]["task_success"], "True")

        summary_path = write_summary(records, self.summary_path)
        with open(summary_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(tuple(reader.fieldnames), SUMMARY_CSV_FIELDS)
            summary = list(reader)
        self.assertEqual(len(summary), 3)
        for row in summary:
            self.assertEqual(row["success_rate"], "1.0")
            self.assertEqual(row["task_count"], "1")

    def test_workspaces_are_removed_unless_kept(self) -> None:
        agent, _ = self.build_agent()
        run_experiment(self.config(), agent=agent, tasks=[self.task], expected_tools=self.expected)
        self.assertFalse(any(self.work_root.rglob("calc.py")))

        agent, _ = self.build_agent()
        run_experiment(
            self.config(keep_workspaces=True), agent=agent,
            tasks=[self.task], expected_tools=self.expected,
        )
        self.assertEqual(len(list(self.work_root.rglob("calc.py"))), 3)

    def test_repeats_start_from_clean_copy(self) -> None:
        agent, _ = self.build_agent()

        records = run_experiment(
            self.config(repeats=2, modes=("baseline",)), agent=agent,
            tasks=[self.task], expected_tools=self.expected,
        )

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.result.task_success for record in records))

    def test_unready_workspace_is_skipped(self) -> None:
        placeholder = self.root / "placeholder"
        placeholder.mkdir()
        (placeholder / ".gitkeep").write_text("", encoding="utf-8")
        pending = task_dict(task_id="task_002", project_path=str(placeholder))
        agent, _ = self.build_agent()

        records = run_experiment(
            self.config(), agent=agent, tasks=[pending, self.task], expected_tools=self.expected
        )

        self.assertEqual(len(records), 3)
        self.assertEqual({record.result.task_id for record in records}, {"task_001"})

    def test_llm_failure_is_recorded_not_raised(self) -> None:
        agent, _ = self.build_agent(llm=FixingLLM(fail=True))

        records = run_experiment(
            self.config(), agent=agent, tasks=[self.task], expected_tools=self.expected
        )

        self.assertEqual(len(records), 3)
        for record in records:
            self.assertFalse(record.result.task_success)
            self.assertIn("api down", record.result.error_message)
            self.assertEqual(record.termination_reason, "error")

    def test_missing_preannotation_is_analyzed_once(self) -> None:
        agent, analyzer = self.build_agent()
        task = {key: value for key, value in self.task.items() if key != "retrieval_query"}

        records = run_experiment(
            self.config(), agent=agent, tasks=[task], expected_tools=self.expected
        )

        self.assertEqual(len(records), 3)
        analyzer.analyze.assert_called_once()
        self.assertTrue(all(record.result.task_success for record in records))

    def test_missing_expected_annotation_defaults(self) -> None:
        agent, _ = self.build_agent()

        records = run_experiment(
            self.config(modes=("baseline",)), agent=agent, tasks=[self.task], expected_tools={}
        )

        self.assertEqual(records[0].tool_recall, 1.0)
        self.assertEqual(records[0].tool_precision, 0.0)

    def test_runner_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            RunnerConfig(modes=())
        with self.assertRaises(ValueError):
            RunnerConfig(modes=("unknown",))
        with self.assertRaises(ValueError):
            RunnerConfig(repeats=0)

    def test_empty_records_write_header_only(self) -> None:
        write_raw_results([], self.raw_path)
        write_summary([], self.summary_path)
        with open(self.raw_path, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.readline().strip(), ",".join(RESULT_CSV_FIELDS))
        with open(self.summary_path, encoding="utf-8", newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)