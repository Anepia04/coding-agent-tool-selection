"""TF-IDF 工具检索模块测试。

覆盖：索引构建、排序、分数与排名、Top-K、中文查询、未初始化索引的异常处理。
测试仅使用模拟的 ToolDefinition / TaskInfo，不调用任何大模型 API。
"""

from __future__ import annotations

import unittest

from agent.task_analyzer import TaskInfo
from retrieval.tfidf_retriever import RetrievedTool, TFIDFRetriever, _tokenize
from tools.definitions import ToolDefinition


def build_mock_tools() -> list[ToolDefinition]:
    """构造一组用于检索测试的模拟工具。"""
    return [
        ToolDefinition(
            name="read_file",
            description="读取指定文件的内容，支持查看指定行区间的代码",
            category="file",
        ),
        ToolDefinition(
            name="write_file",
            description="创建或覆盖写入文件内容，用于修改代码",
            category="file",
        ),
        ToolDefinition(
            name="run_tests",
            description="运行测试用例并返回测试通过或失败的结果",
            category="test",
        ),
        ToolDefinition(
            name="search_code",
            description="在项目中搜索函数名、变量名或关键词",
            category="code_search",
        ),
        ToolDefinition(
            name="git_diff",
            description="查看当前代码变更的差异内容",
            category="git",
        ),
    ]


class TestTFIDFRetriever(unittest.TestCase):
    """TFIDFRetriever 行为测试。"""

    def setUp(self) -> None:
        self.tools = build_mock_tools()
        self.retriever = TFIDFRetriever()

    # ------------------------------------------------------------------
    # 1. 索引构建
    # ------------------------------------------------------------------

    def test_fit_builds_index(self) -> None:
        self.assertFalse(self.retriever.is_fitted)

        result = self.retriever.fit(self.tools)

        self.assertIs(result, self.retriever)  # 支持链式调用
        self.assertTrue(self.retriever.is_fitted)

    def test_fit_rebuilds_index(self) -> None:
        self.retriever.fit(self.tools)
        self.retriever.fit(self.tools[:2])
        results = self.retriever.retrieve("读取文件", top_k=10)
        self.assertEqual(len(results), 2)

    def test_fit_empty_tool_library_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.retriever.fit([])
        self.assertFalse(self.retriever.is_fitted)

    # ------------------------------------------------------------------
    # 2 & 3. 检索排序、分数与排名
    # ------------------------------------------------------------------

    def test_retrieve_orders_by_relevance(self) -> None:
        self.retriever.fit(self.tools)
        results = self.retriever.retrieve("读取文件内容")

        self.assertEqual(len(results), len(self.tools))
        self.assertIsInstance(results[0], RetrievedTool)
        self.assertEqual(results[0].tool_name, "read_file")
        self.assertGreater(results[0].score, 0.0)
        self.assertTrue(all(isinstance(item, RetrievedTool) for item in results))

    def test_scores_and_ranks_are_consistent(self) -> None:
        self.retriever.fit(self.tools)
        results = self.retriever.retrieve("修复代码里的异常")

        scores = [item.score for item in results]
        self.assertEqual([item.rank for item in results], list(range(1, len(results) + 1)))
        self.assertEqual(scores, sorted(scores, reverse=True))  # 降序
        for item in results:
            self.assertIsInstance(item.tool_name, str)
            self.assertIsInstance(item.score, float)
            self.assertGreaterEqual(item.score, 0.0)
            self.assertLessEqual(item.score, 1.0)

    # ------------------------------------------------------------------
    # 4. Top-K
    # ------------------------------------------------------------------

    def test_top_k_limits_results(self) -> None:
        self.retriever.fit(self.tools)
        full = self.retriever.retrieve("运行测试")
        top2 = self.retriever.retrieve("运行测试", top_k=2)

        self.assertEqual(len(top2), 2)
        self.assertEqual([item.tool_name for item in top2],
                         [item.tool_name for item in full[:2]])
        self.assertEqual([item.score for item in top2], [item.score for item in full[:2]])
        self.assertEqual([item.rank for item in top2], [1, 2])

    def test_top_k_larger_than_tool_count(self) -> None:
        self.retriever.fit(self.tools)
        results = self.retriever.retrieve("读取文件", top_k=100)
        self.assertEqual(len(results), len(self.tools))

    def test_invalid_top_k_raises(self) -> None:
        self.retriever.fit(self.tools)
        for invalid in (0, -1, True, 2.0, "3"):
            with self.subTest(top_k=invalid):
                with self.assertRaises(ValueError):
                    self.retriever.retrieve("读取文件", top_k=invalid)

    # ------------------------------------------------------------------
    # 5. 中文任务描述
    # ------------------------------------------------------------------

    def test_chinese_tokenizer_outputs_char_ngrams(self) -> None:
        tokens = _tokenize("运行测试 run_tests")
        for expected in ("运", "运行", "测试", "run_tests", "run", "tests"):
            self.assertIn(expected, tokens)

    def test_retrieve_with_chinese_query(self) -> None:
        self.retriever.fit(self.tools)
        results = self.retriever.retrieve("修复除零异常并运行测试")

        self.assertEqual(results[0].tool_name, "run_tests")
        self.assertGreater(results[0].score, 0.0)
        self.assertIn("run_tests", [item.tool_name for item in results[:3]])

    def test_retrieve_accepts_task_info(self) -> None:
        self.retriever.fit(self.tools)
        task_info = TaskInfo(
            task_id="task_001",
            description="修复除零异常并运行测试",
            task_type="bug_fix",
            language="python",
            intents=["测试"],
            complexity="simple",
        )
        from_task_info = self.retriever.retrieve(task_info)
        from_text = self.retriever.retrieve(task_info.description)

        self.assertEqual(from_task_info[0].tool_name, "run_tests")
        self.assertEqual(from_task_info[0].tool_name, from_text[0].tool_name)

    def test_retrieve_rejects_invalid_query_type(self) -> None:
        self.retriever.fit(self.tools)
        with self.assertRaises(TypeError):
            self.retriever.retrieve(12345)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 6. 未初始化索引
    # ------------------------------------------------------------------

    def test_retrieve_before_fit_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.retriever.retrieve("读取文件")
        with self.assertRaises(RuntimeError):
            self.retriever.retrieve("读取文件", top_k=2)

    # ------------------------------------------------------------------
    # 7. 相同分数的稳定排序
    # ------------------------------------------------------------------

    def test_equal_scores_keep_registration_order(self) -> None:
        tools = [
            ToolDefinition(name="tool_a", description="运行单元测试并返回结果", category="test"),
            ToolDefinition(name="tool_b", description="运行单元测试并返回结果", category="test"),
            ToolDefinition(name="git_diff", description="查看代码差异", category="git"),
        ]
        self.retriever.fit(tools)
        results = self.retriever.retrieve("运行单元测试")

        self.assertAlmostEqual(results[0].score, results[1].score, places=12)
        self.assertEqual([results[0].tool_name, results[1].tool_name], ["tool_a", "tool_b"])
        self.assertEqual([results[0].rank, results[1].rank], [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
