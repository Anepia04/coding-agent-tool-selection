"""TF-IDF 工具检索模块测试。

覆盖：索引构建、排序、分数与排名、Top-K、中文查询、未初始化索引的异常处理。
测试仅使用模拟的 ToolDefinition / TaskInfo，不调用任何大模型 API。
"""

from __future__ import annotations

import unittest

from agent.task_analyzer import TaskInfo
from retrieval.query_utils import build_retrieval_query, sanitize_query
from retrieval.tfidf_retriever import RetrievedTool, TFIDFRetriever, _tokenize
from tools.definitions import ToolDefinition


def build_mock_tools() -> list[ToolDefinition]:
    """构造一组用于检索测试的模拟工具。"""
    return [
        ToolDefinition(
            name="read_file",
            description="读取指定文件的内容，支持查看指定行区间的代码",
            category="file",
            search_hints=["查看", "读取", "文件内容", "阅读代码"],
        ),
        ToolDefinition(
            name="write_file",
            description="创建或覆盖写入文件内容，用于修改代码",
            category="file",
            search_hints=["新建文件", "写入", "修改代码"],
        ),
        ToolDefinition(
            name="run_tests",
            description="运行测试用例并返回测试通过或失败的结果",
            category="test",
            search_hints=["运行测试", "跑测试", "测试通过"],
        ),
        ToolDefinition(
            name="search_code",
            description="在项目中搜索函数名、变量名或关键词",
            category="code_search",
            search_hints=["搜索", "查找", "关键词"],
        ),
        ToolDefinition(
            name="git_diff",
            description="查看当前代码变更的差异内容",
            category="git",
            search_hints=["差异", "对比改动", "改了什么"],
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
        for expected in ("运行", "测试", "run_tests", "run", "tests"):
            self.assertIn(expected, tokens)
        # 中文只输出双字：单字不出现，避免「清除→除零」「差异→异常」这类碎片误匹配
        self.assertNotIn("运", tokens)
        self.assertNotIn("试", tokens)

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
            retrieval_query="修复除零异常 运行测试",
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

    # ------------------------------------------------------------------
    # 8. 否定约束清洗
    # ------------------------------------------------------------------

    def test_sanitize_drops_negated_prefix_clause(self) -> None:
        cleaned = sanitize_query("不要修改任何代码，只解释除零异常")
        self.assertEqual(cleaned, "只解释除零异常")

    def test_sanitize_drops_negated_clause_with_conjunction(self) -> None:
        cleaned = sanitize_query(
            "不要修改任何代码，也不要运行测试，只读取并解释"
        )
        self.assertEqual(cleaned, "只读取并解释")

    def test_sanitize_drops_pure_constraint_clause(self) -> None:
        cleaned = sanitize_query("读取并解释代码，保持代码不变")
        self.assertEqual(cleaned, "读取并解释代码")

    def test_sanitize_keeps_positive_clause_with_mid_sentence_negation(self) -> None:
        query = "修复输入不能为空的校验 bug"
        self.assertEqual(sanitize_query(query), query)

    def test_sanitize_falls_back_when_all_clauses_negated(self) -> None:
        query = "不要修改任何代码"
        self.assertEqual(sanitize_query(query), query)

    def test_sanitize_handles_english_negation(self) -> None:
        cleaned = sanitize_query("do not edit the file, just explain it")
        self.assertEqual(cleaned, "just explain it")

    def test_built_query_no_longer_ranks_forbidden_tool_first(self) -> None:
        self.retriever.fit(self.tools)

        affirmative = self.retriever.retrieve("运行测试")
        cleaned = build_retrieval_query("不要运行测试，只读取文件内容")
        negated = self.retriever.retrieve(cleaned)

        self.assertEqual(cleaned, "只读取文件内容")
        self.assertEqual(affirmative[0].tool_name, "run_tests")
        self.assertEqual(negated[0].tool_name, "read_file")
        self.assertNotEqual(negated[0].tool_name, "run_tests")

    def test_task_info_without_retrieval_query_raises(self) -> None:
        self.retriever.fit(self.tools)
        task_info = TaskInfo(
            task_id="task_missing_query",
            description="不要修改代码，只读取文件内容",
            task_type="doc",
            language="python",
            complexity="simple",
        )

        with self.assertRaises(ValueError):
            self.retriever.retrieve(task_info)

    # ------------------------------------------------------------------
    # 9. 预标注 retrieval_query 优先
    # ------------------------------------------------------------------

    def test_retrieval_query_takes_priority_over_description(self) -> None:
        self.retriever.fit(self.tools)
        task_info = TaskInfo(
            task_id="task_preanalyzed",
            description="不要修改任何代码，也不要运行测试，只解释逻辑",
            task_type="doc",
            language="python",
            retrieval_query="读取文件内容",
            negative_constraints=["不修改代码", "不运行测试"],
            complexity="simple",
        )

        results = self.retriever.retrieve(task_info)

        self.assertEqual(results[0].tool_name, "read_file")
        self.assertNotIn("run_tests", [item.tool_name for item in results[:1]])


class TestCaseInsensitiveMatching(unittest.TestCase):
    """英文大小写归一化。

    ``TF-IDF`` / ``Tf-Idf`` / ``tf-idf`` 表达同一个概念，``TEST`` / ``Test`` /
    ``test`` 也是同一个词；检索时必须折叠成同一词项，否则任务里写 ``TODO`` 就
    匹配不到 hints 里写的 ``todo``。
    """

    def setUp(self) -> None:
        # hints 与 description 故意用混合大小写，顺带验证「文档侧」也被折叠
        self.tools = [
            ToolDefinition(
                name="run_tests",
                description="Run the Test Suite",
                category="test",
                search_hints=["Test", "PYTEST", "Run"],
            ),
            ToolDefinition(
                name="git_diff",
                description="Show Diff",
                category="git",
                search_hints=["Diff", "Changes"],
            ),
        ]
        self.retriever = TFIDFRetriever().fit(self.tools)

    def test_tokenize_folds_english_case(self) -> None:
        # 连字符形式：tf/idf 是两个词项，大小写必须被抹平
        for variant in ("TF-IDF", "Tf-Idf", "tf-idf", "tF-iDf"):
            with self.subTest(variant=variant):
                self.assertEqual(_tokenize(variant), ["tf", "idf"])
        # 单词形式
        for variant in ("TEST", "Test", "test", "TeSt"):
            with self.subTest(variant=variant):
                self.assertEqual(_tokenize(variant), ["test"])

    def test_tokenize_folds_identifier_subwords(self) -> None:
        for variant in ("Read_File", "READ_FILE", "read_file", "Read_file"):
            with self.subTest(variant=variant):
                self.assertEqual(_tokenize(variant), ["read_file", "read", "file"])

    def test_vocabulary_contains_no_uppercase_terms(self) -> None:
        vocabulary = list(self.retriever._vectorizer.get_feature_names_out())
        self.assertTrue(vocabulary)
        for term in vocabulary:
            with self.subTest(term=term):
                self.assertEqual(term, term.casefold())

    def test_lowercase_query_hits_uppercase_hint(self) -> None:
        results = self.retriever.retrieve("test pytest run")

        self.assertEqual(results[0].tool_name, "run_tests")
        self.assertGreater(results[0].score, 0.0)

    def test_query_case_does_not_change_scores(self) -> None:
        baseline = [
            (item.tool_name, item.score)
            for item in self.retriever.retrieve("TEST PYTEST RUN")
        ]
        # 确认基线本身匹配成功，避免「全 0 相等」的假通过
        self.assertGreater(baseline[0][1], 0.0)

        for query in ("test pytest run", "TeSt PyTeSt RuN", "tEsT pytest RUN"):
            with self.subTest(query=query):
                actual = [
                    (item.tool_name, item.score)
                    for item in self.retriever.retrieve(query)
                ]
                self.assertEqual(
                    [name for name, _ in actual], [name for name, _ in baseline]
                )
                for (_, expected), (_, score) in zip(baseline, actual):
                    self.assertAlmostEqual(score, expected, places=12)


class TestLengthCompensation(unittest.TestCase):
    """文档长度补偿：hints 条目多的工具不该被 l2 归一化摊薄到吃亏。"""

    def test_factor_is_log_normalized(self) -> None:
        retriever = TFIDFRetriever().fit(build_mock_tools())
        factors = retriever._length_factor

        # 最长文档的补偿因子为 1.0，其余按 log1p 比例下调
        self.assertAlmostEqual(float(factors.max()), 1.0, places=12)
        self.assertTrue(all(0.0 < float(f) <= 1.0 for f in factors))

    def test_longer_document_gets_larger_factor(self) -> None:
        tools = [
            ToolDefinition(name="short", description="a", category="file",
                           search_hints=["甲甲"]),
            ToolDefinition(name="long", description="b", category="file",
                           search_hints=["乙乙", "丙丙", "丁丁", "戊戊", "己己"]),
        ]
        retriever = TFIDFRetriever().fit(tools)
        factors = dict(zip([t.name for t in tools], retriever._length_factor))
        self.assertGreater(float(factors["long"]), float(factors["short"]))

    def test_scores_stay_in_unit_interval(self) -> None:
        retriever = TFIDFRetriever().fit(build_mock_tools())
        for item in retriever.retrieve("读取文件内容 运行测试"):
            self.assertGreaterEqual(item.score, 0.0)
            self.assertLessEqual(item.score, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
