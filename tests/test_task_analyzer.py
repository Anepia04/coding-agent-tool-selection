"""TaskAnalyzer 测试。

使用 Mock 模拟 LLMClient，不调用真实 API；覆盖正常分析、task_id 处理、
JSON 解析（含代码块包裹）、中文任务、非法字段与调用异常。
"""

from __future__ import annotations

import json
import re
import unittest
from unittest.mock import MagicMock

from agent.llm_client import LLMResponse
from agent.task_analyzer import (
    SYSTEM_PROMPT,
    TASK_TYPES,
    TaskAnalysisError,
    TaskAnalyzer,
    TaskInfo,
)
from config import COMPLEXITY_LEVELS


def build_payload(**overrides) -> dict:
    """构造一份合法的模型 JSON 输出。"""
    payload = {
        "description": "修复计算器的除零异常并补充测试",
        "task_type": "bug_fix",
        "language": "python",
        "intents": ["读取代码", "修改函数", "运行测试"],
        "complexity": "simple",
    }
    payload.update(overrides)
    return payload


def make_analyzer(content=None, side_effect=None):
    """构造带 Mock LLMClient 的 TaskAnalyzer。"""
    llm_client = MagicMock()
    if side_effect is not None:
        llm_client.chat.side_effect = side_effect
    else:
        llm_client.chat.return_value = LLMResponse(content=content)
    return TaskAnalyzer(llm_client=llm_client), llm_client


class TestNormalAnalysis(unittest.TestCase):
    """正常任务分析。"""

    def test_analyze_returns_task_info(self) -> None:
        content = json.dumps(build_payload(), ensure_ascii=False)
        analyzer, llm_client = make_analyzer(content)

        task = analyzer.analyze("修复除零异常，并补充单元测试")

        self.assertIsInstance(task, TaskInfo)
        self.assertEqual(task.description, "修复计算器的除零异常并补充测试")
        self.assertEqual(task.task_type, "bug_fix")
        self.assertEqual(task.language, "python")
        self.assertEqual(task.intents, ["读取代码", "修改函数", "运行测试"])
        self.assertEqual(task.complexity, "simple")
        self.assertTrue(task.task_id.startswith("task_"))

        llm_client.chat.assert_called_once()
        messages = llm_client.chat.call_args.args[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("JSON", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("修复除零异常", messages[1]["content"])

    def test_system_prompt_defines_allowed_values(self) -> None:
        for task_type in TASK_TYPES:
            self.assertIn(task_type, SYSTEM_PROMPT)
        for complexity in COMPLEXITY_LEVELS:
            self.assertIn(complexity, SYSTEM_PROMPT)

    def test_missing_language_and_intents_use_defaults(self) -> None:
        content = json.dumps(
            {"task_type": "refactor", "complexity": "medium"}, ensure_ascii=False
        )
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("重构数据加载模块")
        self.assertEqual(task.language, "unknown")
        self.assertEqual(task.intents, [])
        self.assertEqual(task.description, "重构数据加载模块")

    def test_empty_model_description_falls_back_to_input(self) -> None:
        content = json.dumps(build_payload(description="   "), ensure_ascii=False)
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("修复除零异常")
        self.assertEqual(task.description, "修复除零异常")


class TestTaskId(unittest.TestCase):
    """task_id 传入与默认值。"""

    def test_explicit_task_id_has_priority(self) -> None:
        content = json.dumps(build_payload(task_id="model-generated"), ensure_ascii=False)
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("修复除零异常", task_id="task_042")
        self.assertEqual(task.task_id, "task_042")

    def test_default_task_id_format(self) -> None:
        content = json.dumps(build_payload(), ensure_ascii=False)
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("修复除零异常")
        self.assertRegex(task.task_id, r"^task_[0-9a-f]{8}$")

    def test_default_task_id_is_not_from_model(self) -> None:
        content = json.dumps(build_payload(task_id="model-generated"), ensure_ascii=False)
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("修复除零异常")
        self.assertNotEqual(task.task_id, "model-generated")


class TestJsonParsing(unittest.TestCase):
    """模型返回 JSON 的解析。"""

    def test_json_wrapped_in_code_fence(self) -> None:
        content = "```json\n" + json.dumps(build_payload(), ensure_ascii=False) + "\n```"
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("修复除零异常")
        self.assertEqual(task.task_type, "bug_fix")

    def test_json_with_surrounding_text(self) -> None:
        content = "分析结果如下：" + json.dumps(build_payload(), ensure_ascii=False) + " 以上。"
        analyzer, _ = make_analyzer(content)
        task = analyzer.analyze("修复除零异常")
        self.assertEqual(task.complexity, "simple")

    def test_top_level_json_must_be_object(self) -> None:
        analyzer, _ = make_analyzer('["bug_fix", "simple"]')
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_invalid_json_raises(self) -> None:
        analyzer, _ = make_analyzer("{task_type: bug_fix, complexity: simple}")
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_empty_content_raises(self) -> None:
        for content in (None, "", "   "):
            with self.subTest(content=content):
                analyzer, _ = make_analyzer(content)
                with self.assertRaises(TaskAnalysisError):
                    analyzer.analyze("修复除零异常")


class TestChineseTask(unittest.TestCase):
    """中文任务描述。"""

    def test_chinese_description_and_intents(self) -> None:
        payload = build_payload(
            description="为字符串工具模块新增 slugify 函数",
            task_type="feature_add",
            language="python",
            intents=["新建文件", "编写函数", "补充测试"],
            complexity="medium",
        )
        analyzer, _ = make_analyzer(json.dumps(payload, ensure_ascii=False))
        task = analyzer.analyze("给文本工具加一个 slugify，把标题转成短横线格式")

        self.assertEqual(task.task_type, "feature_add")
        self.assertEqual(task.complexity, "medium")
        self.assertEqual(task.intents, ["新建文件", "编写函数", "补充测试"])
        self.assertIn("slugify", task.description)


class TestInvalidFields(unittest.TestCase):
    """缺少必要字段与非法取值。"""

    def test_missing_task_type_raises(self) -> None:
        payload = build_payload()
        payload.pop("task_type")
        analyzer, _ = make_analyzer(json.dumps(payload, ensure_ascii=False))
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_missing_complexity_raises(self) -> None:
        payload = build_payload()
        payload.pop("complexity")
        analyzer, _ = make_analyzer(json.dumps(payload, ensure_ascii=False))
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_invalid_task_type_raises(self) -> None:
        analyzer, _ = make_analyzer(
            json.dumps(build_payload(task_type="optimize"), ensure_ascii=False)
        )
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("优化性能")

    def test_invalid_complexity_raises(self) -> None:
        analyzer, _ = make_analyzer(
            json.dumps(build_payload(complexity="extreme"), ensure_ascii=False)
        )
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_intents_must_be_list(self) -> None:
        analyzer, _ = make_analyzer(
            json.dumps(build_payload(intents="读取代码"), ensure_ascii=False)
        )
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_intents_items_are_normalized(self) -> None:
        analyzer, _ = make_analyzer(
            json.dumps(build_payload(intents=[" 读取代码 ", "", 123]), ensure_ascii=False)
        )
        task = analyzer.analyze("修复除零异常")
        self.assertEqual(task.intents, ["读取代码", "123"])


class TestLLMErrors(unittest.TestCase):
    """LLMClient 调用异常与前置校验。"""

    def test_llm_call_exception_wrapped(self) -> None:
        analyzer, _ = make_analyzer(side_effect=RuntimeError("network down"))
        with self.assertRaises(TaskAnalysisError) as ctx:
            analyzer.analyze("修复除零异常")
        self.assertIn("network down", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    def test_missing_llm_client_raises(self) -> None:
        analyzer = TaskAnalyzer(llm_client=None)
        with self.assertRaises(TaskAnalysisError):
            analyzer.analyze("修复除零异常")

    def test_empty_description_raises(self) -> None:
        analyzer, _ = make_analyzer(json.dumps(build_payload(), ensure_ascii=False))
        for empty in ("", "   ", None):
            with self.subTest(description=empty):
                with self.assertRaises(ValueError):
                    analyzer.analyze(empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
