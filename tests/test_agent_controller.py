"""AgentController 测试：任务分析、工具选择与多轮 Agent Loop。

全部依赖使用 Mock，不调用真实 API、不依赖真实工具实现。
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from agent.controller import AgentController, AgentRunResult, AgentStep, SYSTEM_PROMPT
from agent.llm_client import LLMResponse, ToolCall
from agent.task_analyzer import TaskInfo
from retrieval.tfidf_retriever import RetrievedTool


def build_task_info(**overrides) -> TaskInfo:
    """构造一个结构化的任务信息。"""
    data = {
        "task_id": "task_001",
        "description": "修复计算器的除零异常并补充测试",
        "task_type": "bug_fix",
        "language": "python",
        "intents": ["修改代码", "运行测试"],
        "complexity": "simple",
    }
    data.update(overrides)
    return TaskInfo(**data)


def build_controller(response=None, responses=None, with_deps: bool = True):
    """构造装配了 Mock 依赖的 AgentController。

    Args:
        response: 单次响应的 return_value（适用于无工具调用的场景）。
        responses: 按轮次依次返回的响应列表（多轮场景必须使用，避免无限重复）。
    """
    llm_client = MagicMock()
    if responses is not None:
        llm_client.chat.side_effect = list(responses)
    else:
        llm_client.chat.return_value = response or LLMResponse(content="任务已完成")

    task_analyzer = MagicMock()
    task_analyzer.analyze.return_value = build_task_info()

    tool_registry = MagicMock()
    tool_registry.tool_names = ["read_file", "write_file", "run_tests"]
    tool_registry.get_schemas.return_value = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": {"name": "run_tests"}},
    ]
    tool_registry.execute_tool.return_value = "file content"

    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedTool(tool_name="read_file", score=0.9, rank=1),
        RetrievedTool(tool_name="run_tests", score=0.6, rank=2),
    ]
    selector = MagicMock()
    selector.select.return_value = ["read_file", "run_tests"]

    controller = AgentController(
        llm_client=llm_client,
        task_analyzer=task_analyzer,
        tool_registry=tool_registry,
        retriever=retriever if with_deps else None,
        selector=selector if with_deps else None,
    )
    mocks = {
        "llm": llm_client,
        "analyzer": task_analyzer,
        "registry": tool_registry,
        "retriever": retriever,
        "selector": selector,
    }
    return controller, mocks


def tool_response(*calls: ToolCall, content=None, **tokens) -> LLMResponse:
    """构造一次请求工具调用的模型响应。"""
    return LLMResponse(content=content, tool_calls=list(calls), **tokens)


def text_response(content="已完成", **tokens) -> LLMResponse:
    """构造一次最终文本响应。"""
    return LLMResponse(content=content, **tokens)


class TestBaselineMode(unittest.TestCase):
    """baseline：提供全部工具。"""

    def test_baseline_uses_all_tools(self) -> None:
        controller, mocks = build_controller()

        result = controller.run("修复除零异常", mode="baseline")

        self.assertIsInstance(result, AgentRunResult)
        self.assertEqual(result.mode, "baseline")
        self.assertEqual(result.task_id, "task_001")
        self.assertEqual(result.selected_tools, ["read_file", "write_file", "run_tests"])
        mocks["registry"].get_schemas.assert_called_once_with(
            ["read_file", "write_file", "run_tests"]
        )
        mocks["retriever"].retrieve.assert_not_called()
        mocks["selector"].select.assert_not_called()

    def test_baseline_without_retriever_and_selector(self) -> None:
        controller, _ = build_controller(with_deps=False)
        result = controller.run("修复除零异常", mode="baseline")
        self.assertEqual(result.selected_tools, ["read_file", "write_file", "run_tests"])


class TestFilteredModes(unittest.TestCase):
    """fixed_topk / adaptive_topk：仅工具选择策略不同。"""

    def test_fixed_topk_selection(self) -> None:
        controller, mocks = build_controller()
        task_info = mocks["analyzer"].analyze.return_value

        result = controller.run("修复除零异常", mode="fixed_topk")

        mocks["retriever"].retrieve.assert_called_once_with(task_info)
        mocks["selector"].select.assert_called_once_with(
            mocks["retriever"].retrieve.return_value, strategy="fixed"
        )
        self.assertEqual(result.selected_tools, ["read_file", "run_tests"])
        mocks["registry"].get_schemas.assert_called_once_with(["read_file", "run_tests"])

    def test_adaptive_topk_selection(self) -> None:
        controller, mocks = build_controller()
        task_info = mocks["analyzer"].analyze.return_value

        result = controller.run("修复除零异常", mode="adaptive_topk")

        mocks["retriever"].retrieve.assert_called_once_with(task_info)
        mocks["selector"].select.assert_called_once_with(
            mocks["retriever"].retrieve.return_value,
            complexity=task_info.complexity,
            strategy="adaptive",
        )
        self.assertEqual(result.selected_tools, ["read_file", "run_tests"])

    def test_selected_tools_are_taken_from_selector(self) -> None:
        controller, mocks = build_controller()
        mocks["selector"].select.return_value = ["write_file", "run_tests"]

        result = controller.run("修复除零异常", mode="fixed_topk")

        self.assertEqual(result.selected_tools, ["write_file", "run_tests"])
        mocks["registry"].get_schemas.assert_called_once_with(["write_file", "run_tests"])


class TestTaskAnalysis(unittest.TestCase):
    """任务输入与 TaskAnalyzer 调用。"""

    def test_text_task_is_analyzed(self) -> None:
        controller, mocks = build_controller()
        controller.run("修复除零异常", mode="baseline")
        mocks["analyzer"].analyze.assert_called_once_with("修复除零异常", task_id=None)

    def test_benchmark_task_mapping_passes_task_id(self) -> None:
        controller, mocks = build_controller()
        task = {"task_id": "task_042", "description": "修复除零异常"}
        controller.run(task, mode="baseline")
        mocks["analyzer"].analyze.assert_called_once_with("修复除零异常", task_id="task_042")

    def test_task_info_input_skips_analysis(self) -> None:
        controller, mocks = build_controller()
        task_info = build_task_info(task_id="task_777")
        result = controller.run(task_info, mode="baseline")

        mocks["analyzer"].analyze.assert_not_called()
        self.assertEqual(result.task_id, "task_777")


class TestLLMInteraction(unittest.TestCase):
    """LLMClient 调用与消息构造。"""

    def test_llm_receives_messages_and_selected_schemas(self) -> None:
        controller, mocks = build_controller()
        mocks["analyzer"].analyze.return_value = build_task_info(description="给工具加日志")

        result = controller.run("给工具加日志", mode="baseline")

        mocks["llm"].chat.assert_called_once()
        messages = mocks["llm"].chat.call_args.args[0]
        tools = mocks["llm"].chat.call_args.kwargs["tools"]

        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1], {"role": "user", "content": "给工具加日志"})
        self.assertEqual(tools, mocks["registry"].get_schemas.return_value)
        self.assertEqual(result.total_tokens, 0)

    def test_token_usage_is_accumulated(self) -> None:
        response = LLMResponse(
            content="完成", input_tokens=120, output_tokens=30, total_tokens=150
        )
        controller, _ = build_controller(response=response)

        result = controller.run("修复除零异常", mode="baseline")

        self.assertEqual(result.input_tokens, 120)
        self.assertEqual(result.output_tokens, 30)
        self.assertEqual(result.total_tokens, 150)
        self.assertGreaterEqual(result.execution_time, 0.0)

    def test_llm_failure_is_recorded(self) -> None:
        controller, mocks = build_controller()
        mocks["llm"].chat.side_effect = RuntimeError("api down")

        result = controller.run("修复除零异常", mode="fixed_topk")

        self.assertFalse(result.task_success)
        self.assertIn("api down", result.error_message)
        self.assertEqual(result.selected_tools, ["read_file", "run_tests"])
        self.assertEqual(result.steps, [])
        self.assertEqual(result.termination_reason, "error")


class TestSingleRoundToolCalls(unittest.TestCase):
    """单轮内的工具调用执行与记录。"""

    def test_tool_call_is_executed_and_recorded(self) -> None:
        responses = [
            tool_response(
                ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}),
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            text_response(),
        ]
        controller, mocks = build_controller(responses=responses)

        result = controller.run("读取 a.py", mode="baseline")

        mocks["registry"].execute_tool.assert_called_once_with("read_file", {"path": "a.py"})
        self.assertEqual(result.tool_call_count, 1)
        self.assertTrue(result.task_success)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(len(result.steps), 3)

        assistant_step, tool_step, final_step = result.steps
        self.assertEqual(assistant_step.step_type, "assistant")
        self.assertEqual(tool_step.step_type, "tool")
        self.assertEqual(tool_step.step_index, 1)
        self.assertEqual(tool_step.tool_name, "read_file")
        self.assertEqual(tool_step.tool_arguments, {"path": "a.py"})
        self.assertEqual(tool_step.tool_result, "file content")
        self.assertIsNone(tool_step.error_message)
        self.assertEqual(final_step.step_type, "assistant")
        self.assertEqual(final_step.content, "已完成")

    def test_multiple_tool_calls_are_all_executed(self) -> None:
        responses = [
            tool_response(
                ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call_2", name="run_tests", arguments={"timeout": 30}),
            ),
            text_response(),
        ]
        controller, mocks = build_controller(responses=responses)

        result = controller.run("修复并验证", mode="baseline")

        self.assertEqual(mocks["registry"].execute_tool.call_count, 2)
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(
            [step.tool_name for step in result.steps if step.step_type == "tool"],
            ["read_file", "run_tests"],
        )

    def test_no_tool_call_only_records_assistant_step(self) -> None:
        controller, mocks = build_controller()

        result = controller.run("解释这段代码", mode="baseline")

        self.assertEqual(result.tool_call_count, 0)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].step_type, "assistant")
        self.assertEqual(result.termination_reason, "completed")
        mocks["registry"].execute_tool.assert_not_called()

    def test_tool_execution_failure_is_fed_back(self) -> None:
        responses = [
            tool_response(
                ToolCall(id="call_1", name="read_file", arguments={"path": "missing.py"})
            ),
            text_response("已根据错误信息调整"),
        ]
        controller, mocks = build_controller(responses=responses)
        mocks["registry"].execute_tool.side_effect = FileNotFoundError("missing.py")

        result = controller.run("读取 missing.py", mode="baseline")

        # 工具失败属于可恢复错误：不标记整体失败，错误信息回填给模型
        self.assertTrue(result.task_success)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.tool_call_count, 1)
        tool_step = result.steps[1]
        self.assertIn("FileNotFoundError", tool_step.error_message)
        self.assertIsNone(tool_step.tool_result)

        second_messages = mocks["llm"].chat.call_args_list[1].args[0]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("工具执行失败", tool_messages[0]["content"])
        self.assertIn("missing.py", tool_messages[0]["content"])

    def test_analysis_failure_is_recorded(self) -> None:
        controller, mocks = build_controller()
        mocks["analyzer"].analyze.side_effect = RuntimeError("analyze failed")

        result = controller.run("修复除零异常", mode="baseline")

        self.assertFalse(result.task_success)
        self.assertIn("analyze failed", result.error_message)
        self.assertEqual(result.task_id, "")
        self.assertEqual(result.termination_reason, "error")
        mocks["llm"].chat.assert_not_called()


class TestMultiTurnLoop(unittest.TestCase):
    """多轮 Agent Loop：结果回填、tool_call_id 关联与终止条件。"""

    def test_two_consecutive_tool_rounds(self) -> None:
        responses = [
            tool_response(
                ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"})
            ),
            tool_response(
                ToolCall(id="call_2", name="run_tests", arguments={"timeout": 60})
            ),
            text_response("修复完成，测试通过"),
        ]
        controller, mocks = build_controller(responses=responses)

        result = controller.run("修复 a.py 并运行测试", mode="baseline")

        self.assertEqual(mocks["llm"].chat.call_count, 3)
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(result.termination_reason, "completed")
        self.assertTrue(result.task_success)
        self.assertEqual(
            [step.step_type for step in result.steps],
            ["assistant", "tool", "assistant", "tool", "assistant"],
        )

    def test_tool_results_and_ids_are_fed_back(self) -> None:
        responses = [
            tool_response(
                ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="call_2", name="run_tests", arguments={"timeout": 60}),
            ),
            text_response(),
        ]
        controller, mocks = build_controller(responses=responses)
        mocks["registry"].execute_tool.side_effect = ["print('hi')", {"passed": 3}]

        controller.run("读取并运行测试", mode="baseline")

        second_messages = mocks["llm"].chat.call_args_list[1].args[0]
        assistant_messages = [m for m in second_messages if m.get("role") == "assistant"]
        tool_messages = [m for m in second_messages if m.get("role") == "tool"]

        self.assertEqual(len(assistant_messages), 1)
        assistant_msg = assistant_messages[0]
        self.assertEqual([call["id"] for call in assistant_msg["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(
            [call["function"]["name"] for call in assistant_msg["tool_calls"]],
            ["read_file", "run_tests"],
        )
        self.assertEqual(
            json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]),
            {"path": "a.py"},
        )

        # tool_call_id 与请求一一对应，结果按顺序回填
        self.assertEqual(
            [(m["tool_call_id"], m["content"]) for m in tool_messages],
            [("call_1", "print('hi')"), ("call_2", '{"passed": 3}')],
        )
        self.assertEqual([m["name"] for m in tool_messages], ["read_file", "run_tests"])

    def test_max_steps_terminates_loop(self) -> None:
        responses = [
            tool_response(
                ToolCall(id=f"call_{i}", name="read_file", arguments={"path": f"{i}.py"})
            )
            for i in range(1, 3)
        ]
        controller, mocks = build_controller(responses=responses)

        result = controller.run("持续读取文件", mode="baseline", max_steps=2)

        self.assertEqual(mocks["llm"].chat.call_count, 2)
        self.assertEqual(result.tool_call_count, 2)
        self.assertEqual(result.termination_reason, "max_steps")
        self.assertTrue(result.task_success)
        self.assertEqual(len(result.steps), 4)

    def test_default_max_steps_is_used(self) -> None:
        controller, mocks = build_controller()
        controller.default_max_steps = 2
        controller.llm_client = mocks["llm"]
        mocks["llm"].chat.side_effect = [
            tool_response(ToolCall(id="call_1", name="read_file", arguments={})),
            tool_response(ToolCall(id="call_2", name="read_file", arguments={})),
        ]

        result = controller.run("持续读取", mode="baseline")

        self.assertEqual(result.termination_reason, "max_steps")
        self.assertEqual(mocks["llm"].chat.call_count, 2)

    def test_tokens_accumulate_across_rounds(self) -> None:
        responses = [
            tool_response(
                ToolCall(id="call_1", name="read_file", arguments={"path": "a.py"}),
                input_tokens=10, output_tokens=5, total_tokens=15,
            ),
            tool_response(
                ToolCall(id="call_2", name="run_tests", arguments={}),
                input_tokens=20, output_tokens=10, total_tokens=30,
            ),
            text_response("完成", input_tokens=7, output_tokens=3, total_tokens=10),
        ]
        controller, _ = build_controller(responses=responses)

        result = controller.run("修复并验证", mode="baseline")

        self.assertEqual(result.input_tokens, 37)
        self.assertEqual(result.output_tokens, 18)
        self.assertEqual(result.total_tokens, 55)

    def test_selected_tools_stay_constant_across_rounds(self) -> None:
        responses = [
            tool_response(ToolCall(id="call_1", name="read_file", arguments={})),
            text_response(),
        ]
        controller, mocks = build_controller(responses=responses)

        result = controller.run("读取文件", mode="fixed_topk")

        tools_per_call = [call.kwargs["tools"] for call in mocks["llm"].chat.call_args_list]
        self.assertEqual(len(tools_per_call), 2)
        self.assertEqual(tools_per_call[0], tools_per_call[1])
        self.assertEqual(result.selected_tools, ["read_file", "run_tests"])
        mocks["registry"].get_schemas.assert_called_once()

    def test_all_three_modes_run_multi_turn(self) -> None:
        expected = {
            "baseline": ["read_file", "write_file", "run_tests"],
            "fixed_topk": ["read_file", "run_tests"],
            "adaptive_topk": ["read_file", "run_tests"],
        }
        for mode, tools in expected.items():
            with self.subTest(mode=mode):
                responses = [
                    tool_response(ToolCall(id="call_1", name="read_file", arguments={})),
                    text_response(),
                ]
                controller, mocks = build_controller(responses=responses)

                result = controller.run("修复除零异常", mode=mode)

                self.assertEqual(result.mode, mode)
                self.assertEqual(result.selected_tools, tools)
                self.assertEqual(mocks["llm"].chat.call_count, 2)
                self.assertEqual(result.tool_call_count, 1)
                self.assertEqual(result.termination_reason, "completed")


class TestValidation(unittest.TestCase):
    """非法参数与缺失依赖。"""

    def test_invalid_mode_raises(self) -> None:
        controller, mocks = build_controller()
        with self.assertRaises(ValueError):
            controller.run("修复除零异常", mode="unknown")
        mocks["analyzer"].analyze.assert_not_called()

    def test_fixed_topk_without_retriever_raises(self) -> None:
        controller = AgentController(
            llm_client=MagicMock(),
            task_analyzer=MagicMock(),
            tool_registry=MagicMock(),
            retriever=None,
            selector=MagicMock(),
        )
        with self.assertRaises(ValueError):
            controller.run("修复除零异常", mode="fixed_topk")

    def test_adaptive_topk_without_selector_raises(self) -> None:
        controller = AgentController(
            llm_client=MagicMock(),
            task_analyzer=MagicMock(),
            tool_registry=MagicMock(),
            retriever=MagicMock(),
            selector=None,
        )
        with self.assertRaises(ValueError):
            controller.run("修复除零异常", mode="adaptive_topk")

    def test_invalid_max_steps_raises(self) -> None:
        controller, _ = build_controller()
        with self.assertRaises(ValueError):
            controller.run("修复除零异常", mode="baseline", max_steps=0)

    def test_empty_description_raises(self) -> None:
        controller, _ = build_controller()
        with self.assertRaises(ValueError):
            controller.run("   ", mode="baseline")

    def test_mapping_without_description_raises(self) -> None:
        controller, _ = build_controller()
        with self.assertRaises(ValueError):
            controller.run({"task_id": "task_001"}, mode="baseline")

    def test_unsupported_task_type_raises(self) -> None:
        controller, _ = build_controller()
        with self.assertRaises(TypeError):
            controller.run(123, mode="baseline")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main(verbosity=2)
