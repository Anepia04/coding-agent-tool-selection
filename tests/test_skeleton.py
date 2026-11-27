"""框架自检测试（仅使用标准库 unittest）。

验证内容：
1. 所有包与模块可以正常导入；
2. 统一数据结构的字段、默认值与序列化可用；
3. ToolRegistry 的注册 / 查询 / schema 转换可以正常工作；
4. 所有未实现接口会抛出 NotImplementedError，而不是静默通过；
5. benchmark 数据文件结构符合约定。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import config
from agent.controller import AgentController, AgentRunResult, AgentStep
from agent.llm_client import LLMClient, LLMResponse, ToolCall
from agent.task_analyzer import TaskAnalyzer, TaskInfo
from experiments import adaptive_topk, baseline, evaluate, fixed_topk
from experiments.schemas import RESULT_CSV_FIELDS, ExperimentConfig, ExperimentResult
from retrieval.adaptive_selector import AdaptiveSelector
from retrieval.tfidf_retriever import RetrievedTool
from tools.definitions import EXAMPLE_TOOL_DEFINITIONS, ToolDefinition
from tools.registry import ToolRegistry, build_default_registry
from tools.implementations import set_workspace_root

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestFrameworkSkeleton(unittest.TestCase):
    """框架骨架自检。"""

    # ------------------------------------------------------------------
    # 导入与配置
    # ------------------------------------------------------------------

    def test_settings_loaded(self) -> None:
        settings = config.get_settings()
        self.assertEqual(settings.agent.max_steps, 8)
        self.assertEqual(settings.retrieval.top_k, 5)
        self.assertIn(settings.default_mode, config.EXPERIMENT_MODES)
        # API Key 必须来自环境变量，代码中不外泄
        self.assertTrue(settings.model.api_key is None or isinstance(settings.model.api_key, str))

    def test_benchmark_files_exist_and_valid(self) -> None:
        tasks_path = PROJECT_ROOT / "benchmark" / "tasks.json"
        expected_path = PROJECT_ROOT / "benchmark" / "expected_tools.json"
        self.assertTrue(tasks_path.is_file())
        self.assertTrue(expected_path.is_file())

        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        task_ids = {task["task_id"] for task in tasks}
        expected_ids = {item["task_id"] for item in expected}
        self.assertEqual(task_ids, expected_ids)

        for task in tasks:
            for field in ("task_id", "description", "project_path",
                          "task_type", "complexity", "success_criteria"):
                self.assertIn(field, task)
            self.assertIn(task["complexity"], config.COMPLEXITY_LEVELS)

        for item in expected:
            self.assertIn("required_tools", item)
            self.assertIn("optional_tools", item)

        # Expected Tools 中的工具名必须都能在工具注册表中找到
        registered = set(build_default_registry().tool_names)
        for item in expected:
            for tool_name in item["required_tools"] + item["optional_tools"]:
                self.assertIn(tool_name, registered)

    # ------------------------------------------------------------------
    # 数据结构
    # ------------------------------------------------------------------

    def test_task_info_roundtrip(self) -> None:
        data = {
            "task_id": "task_001",
            "description": "修复除零异常",
            "task_type": "bug_fix",
            "language": "python",
            "intents": ["bug_fix", "test"],
            "complexity": "simple",
        }
        info = TaskInfo.from_dict(data)
        self.assertEqual(info.to_dict(), data)

        with self.assertRaises(ValueError):
            TaskInfo(task_id="x", description="y", complexity="unknown")

    def test_llm_response_defaults(self) -> None:
        response = LLMResponse()
        self.assertFalse(response.has_tool_calls)
        self.assertEqual(response.total_tokens, 0)

        response = LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})],
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )
        self.assertTrue(response.has_tool_calls)
        self.assertEqual(response.to_dict()["tool_calls"][0]["name"], "read_file")

    def test_agent_result_and_experiment_schemas(self) -> None:
        run_result = AgentRunResult(
            task_id="task_001",
            mode="baseline",
            selected_tools=["read_file"],
            steps=[AgentStep(step_index=0, step_type="assistant", content="ok")],
        )
        payload = run_result.to_dict()
        self.assertEqual(payload["task_id"], "task_001")
        self.assertEqual(payload["steps"][0]["step_type"], "assistant")

        exp_config = ExperimentConfig(mode="fixed_topk", task_id="task_001", top_k=5)
        self.assertEqual(exp_config.to_dict()["top_k"], 5)
        with self.assertRaises(ValueError):
            ExperimentConfig(mode="unknown", task_id="task_001")

        exp_result = ExperimentResult(task_id="task_001", experiment_mode="baseline")
        self.assertEqual(set(exp_result.to_dict().keys()), set(RESULT_CSV_FIELDS))

        retrieved = RetrievedTool(tool_name="read_file", score=0.9, rank=1)
        self.assertEqual(retrieved.to_dict(), {"tool_name": "read_file", "score": 0.9, "rank": 1})

    # ------------------------------------------------------------------
    # 工具注册表（框架基础设施，本阶段已可用）
    # ------------------------------------------------------------------

    def test_tool_registry(self) -> None:
        registry = build_default_registry()
        self.assertEqual(len(registry), len(EXAMPLE_TOOL_DEFINITIONS))

        definition, implementation = registry.get_tool("read_file")
        self.assertIsInstance(definition, ToolDefinition)
        self.assertTrue(callable(implementation))

        names = ["read_file", "run_tests"]
        self.assertEqual([d.name for d in registry.get_tools_by_names(names)], names)

        schemas = registry.get_schemas(names)
        self.assertEqual(len(schemas), 2)
        self.assertEqual(schemas[0]["type"], "function")
        self.assertEqual(schemas[0]["function"]["name"], "read_file")
        self.assertIn("parameters", schemas[0]["function"])
        self.assertEqual(len(registry.get_schemas()), len(EXAMPLE_TOOL_DEFINITIONS))

        with self.assertRaises(KeyError):
            registry.get_tool("not_exists")
        with self.assertRaises(KeyError):
            registry.get_tools_by_names(["read_file", "not_exists"])
        with self.assertRaises(ValueError):
            registry.register_tool(ToolDefinition(name="read_file", description="dup"),
                                   lambda **kwargs: None)

    def test_tool_registry_execute_dispatch(self) -> None:
        registry = ToolRegistry()
        registry.register_tool(
            ToolDefinition(name="echo", description="回显参数"),
            lambda text="": f"echo:{text}",
        )
        self.assertEqual(registry.execute_tool("echo", {"text": "hi"}), "echo:hi")

        # 示例工具实现尚未完成，必须明确抛出 NotImplementedError
        default_registry = build_default_registry()
        workspace = PROJECT_ROOT / "workspace"
        workspace.mkdir(exist_ok=True)
        (workspace / "a.py").write_text("print('ok')\n", encoding="utf-8")
        set_workspace_root(workspace)
        try:
            self.assertEqual(
                default_registry.execute_tool("read_file", {"path": "a.py"}),
                "print('ok')",
            )
        finally:
            set_workspace_root(None)

    # ------------------------------------------------------------------
    # 未实现接口必须明确报错
    # ------------------------------------------------------------------

    def test_unimplemented_interfaces(self) -> None:
        llm_client = LLMClient(model="dummy-model")
        task_analyzer = TaskAnalyzer(llm_client=None)
        registry = build_default_registry()
        controller = AgentController(llm_client, task_analyzer, registry)
        exp_result = ExperimentResult(task_id="task_001", experiment_mode="baseline")

        cases = {
            "run_baseline": lambda: baseline.run_baseline(controller, {"task_id": "task_001"}),
            "run_fixed_topk": lambda: fixed_topk.run_fixed_topk(
                controller, {"task_id": "task_001"}, top_k=3
            ),
            "run_adaptive_topk": lambda: adaptive_topk.run_adaptive_topk(
                controller, {"task_id": "task_001"}
            ),
            "evaluate_task_success": lambda: evaluate.evaluate_task_success(exp_result),
            "calculate_token_usage": lambda: evaluate.calculate_token_usage(exp_result),
            "calculate_execution_time": lambda: evaluate.calculate_execution_time(exp_result),
            "calculate_tool_call_count": lambda: evaluate.calculate_tool_call_count(exp_result),
            "calculate_tool_recall": lambda: evaluate.calculate_tool_recall(["a"], ["a"]),
            "calculate_tool_precision": lambda: evaluate.calculate_tool_precision(["a"], ["a"]),
        }
        for name, call in cases.items():
            with self.subTest(interface=name):
                with self.assertRaises(NotImplementedError):
                    call()

    def test_controller_rejects_unknown_mode(self) -> None:
        controller = AgentController(LLMClient("dummy"), TaskAnalyzer(), build_default_registry())
        with self.assertRaises(ValueError):
            controller.run({"task_id": "task_001"}, mode="unknown_mode")

    def test_selector_rejects_unknown_strategy(self) -> None:
        selector = AdaptiveSelector()
        with self.assertRaises(ValueError):
            selector.select([], strategy="unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
