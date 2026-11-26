"""Adaptive Top-K 工具选择模块测试。

使用模拟的 RetrievedTool 数据，覆盖固定 / 自适应两种策略、排序保持、数量不足、
空结果、非法输入与自定义配置；不依赖真实工具库，也不调用任何大模型 API。
"""

from __future__ import annotations

import unittest

from retrieval.adaptive_selector import (
    COMPLEXITY_TOP_K,
    AdaptiveSelector,
    SelectionConfig,
)
from retrieval.tfidf_retriever import RetrievedTool


def build_ranked_tools(count: int = 8) -> list[RetrievedTool]:
    """构造按分数降序、排名连续的模拟检索结果。"""
    return [
        RetrievedTool(
            tool_name=f"tool_{index + 1}",
            score=round(1.0 - index * 0.1, 2),
            rank=index + 1,
        )
        for index in range(count)
    ]


class TestFixedSelection(unittest.TestCase):
    """fixed 策略。"""

    def test_fixed_uses_default_top_k(self) -> None:
        selector = AdaptiveSelector()
        selected = selector.select(build_ranked_tools(8), strategy="fixed")
        self.assertEqual(selected, ["tool_1", "tool_2", "tool_3", "tool_4", "tool_5"])

    def test_fixed_uses_custom_top_k(self) -> None:
        selector = AdaptiveSelector(SelectionConfig(top_k=3))
        selected = selector.select(build_ranked_tools(8), strategy="fixed")
        self.assertEqual(selected, ["tool_1", "tool_2", "tool_3"])

    def test_default_strategy_is_fixed(self) -> None:
        selector = AdaptiveSelector(SelectionConfig(top_k=2))
        self.assertEqual(selector.select(build_ranked_tools(8)), ["tool_1", "tool_2"])


class TestAdaptiveSelection(unittest.TestCase):
    """adaptive 策略：按复杂度确定数量。"""

    def test_adaptive_k_by_complexity(self) -> None:
        selector = AdaptiveSelector()
        ranked = build_ranked_tools(8)
        expected = {"simple": 3, "medium": 5, "complex": 7}
        for complexity, count in expected.items():
            with self.subTest(complexity=complexity):
                selected = selector.select(ranked, complexity=complexity, strategy="adaptive")
                self.assertEqual(len(selected), count)
                self.assertEqual(selected, [tool.tool_name for tool in ranked[:count]])

    def test_complexity_top_k_mapping(self) -> None:
        self.assertEqual(COMPLEXITY_TOP_K, {"simple": 3, "medium": 5, "complex": 7})

    def test_adaptive_without_complexity_raises(self) -> None:
        selector = AdaptiveSelector()
        with self.assertRaises(ValueError):
            selector.select(build_ranked_tools(8), strategy="adaptive")

    def test_unknown_complexity_raises(self) -> None:
        selector = AdaptiveSelector()
        with self.assertRaises(ValueError):
            selector.select(build_ranked_tools(8), complexity="extreme", strategy="adaptive")


class TestOrdering(unittest.TestCase):
    """两种策略都必须保持检索结果的相关性顺序。"""

    def test_selection_preserves_rank_order(self) -> None:
        ranked = build_ranked_tools(8)
        selector = AdaptiveSelector(SelectionConfig(top_k=4))

        fixed = selector.select(ranked, strategy="fixed")
        adaptive = selector.select(ranked, complexity="complex", strategy="adaptive")

        self.assertEqual(fixed, [tool.tool_name for tool in ranked[:4]])
        self.assertEqual(adaptive, [tool.tool_name for tool in ranked[:7]])

    def test_input_list_is_not_mutated(self) -> None:
        ranked = build_ranked_tools(5)
        snapshot = list(ranked)
        AdaptiveSelector(SelectionConfig(top_k=2)).select(ranked, strategy="fixed")
        self.assertEqual(ranked, snapshot)


class TestInsufficientTools(unittest.TestCase):
    """可用工具不足时最多返回全部工具。"""

    def test_fewer_tools_than_fixed_k(self) -> None:
        ranked = build_ranked_tools(2)
        selector = AdaptiveSelector(SelectionConfig(top_k=5))
        self.assertEqual(selector.select(ranked, strategy="fixed"), ["tool_1", "tool_2"])

    def test_fewer_tools_than_adaptive_k(self) -> None:
        ranked = build_ranked_tools(2)
        selector = AdaptiveSelector()
        self.assertEqual(
            selector.select(ranked, complexity="complex", strategy="adaptive"),
            ["tool_1", "tool_2"],
        )

    def test_single_tool(self) -> None:
        selector = AdaptiveSelector()
        self.assertEqual(
            selector.select(build_ranked_tools(1), complexity="medium", strategy="adaptive"),
            ["tool_1"],
        )


class TestEmptyInput(unittest.TestCase):
    """空检索结果。"""

    def test_empty_fixed(self) -> None:
        self.assertEqual(AdaptiveSelector().select([], strategy="fixed"), [])

    def test_empty_adaptive(self) -> None:
        selector = AdaptiveSelector()
        for complexity in ("simple", "medium", "complex"):
            with self.subTest(complexity=complexity):
                self.assertEqual(
                    selector.select([], complexity=complexity, strategy="adaptive"),
                    [],
                )


class TestErrors(unittest.TestCase):
    """非法参数。"""

    def test_invalid_strategy_raises(self) -> None:
        selector = AdaptiveSelector()
        for strategy in ("topk", "", "FIXED"):
            with self.subTest(strategy=strategy):
                with self.assertRaises(ValueError):
                    selector.select(build_ranked_tools(5), strategy=strategy)

    def test_invalid_config_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            AdaptiveSelector(config=123)  # type: ignore[arg-type]

    def test_invalid_config_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            SelectionConfig(top_k=0)
        with self.assertRaises(ValueError):
            SelectionConfig(min_k=5, max_k=3)


class TestCustomConfig(unittest.TestCase):
    """自定义 SelectionConfig 在两种策略下生效。"""

    def test_custom_config_affects_fixed(self) -> None:
        selector = AdaptiveSelector(SelectionConfig(top_k=2))
        self.assertEqual(selector.select(build_ranked_tools(8), strategy="fixed"),
                         ["tool_1", "tool_2"])

    def test_custom_config_clamps_adaptive(self) -> None:
        # 区间 [4, 6]：simple(3)->4，medium(5)->5，complex(7)->6
        selector = AdaptiveSelector(SelectionConfig(top_k=6, min_k=4, max_k=6))
        ranked = build_ranked_tools(8)
        expected = {"simple": 4, "medium": 5, "complex": 6}
        for complexity, count in expected.items():
            with self.subTest(complexity=complexity):
                selected = selector.select(ranked, complexity=complexity, strategy="adaptive")
                self.assertEqual(len(selected), count)
                self.assertEqual(selected, [tool.tool_name for tool in ranked[:count]])

    def test_config_dict_is_supported(self) -> None:
        selector = AdaptiveSelector({"top_k": 3, "min_k": 1, "max_k": 2})
        ranked = build_ranked_tools(8)
        self.assertEqual(selector.select(ranked, strategy="fixed"),
                         ["tool_1", "tool_2", "tool_3"])
        self.assertEqual(
            selector.select(ranked, complexity="simple", strategy="adaptive"),
            ["tool_1", "tool_2"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
