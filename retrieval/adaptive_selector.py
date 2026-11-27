"""工具选择器。

职责：在检索结果的基础上确定最终提供给大模型的工具集合：
- ``fixed``   ：固定选择前 K 个工具（Fixed Top-K 模式）；
- ``adaptive``：结合任务复杂度动态确定工具数量（Adaptive Top-K 模式）。

自适应规则（初始版本）：
    simple -> 3，medium -> 5，complex -> 7，
并受 ``SelectionConfig.min_k`` / ``max_k`` 约束。

负责成员：成员 A
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from .tfidf_retriever import RetrievedTool

logger = logging.getLogger(__name__)

#: 选择策略取值
SELECTION_STRATEGIES = ("fixed", "adaptive")

#: 任务复杂度 -> 初始工具数量（adaptive 策略），实际取值还会被 min_k / max_k 约束
COMPLEXITY_TOP_K = {
    "simple": 3,
    "medium": 5,
    "complex": 7,
}


@dataclass(frozen=True)
class SelectionConfig:
    """工具选择配置。

    Attributes:
        top_k: Fixed Top-K 模式的固定 K 值。
        min_k: Adaptive 模式下允许的最少工具数。
        max_k: Adaptive 模式下允许的最多工具数。
    """

    top_k: int = 5
    min_k: int = 3
    max_k: int = 8

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        if not 0 < self.min_k <= self.max_k:
            raise ValueError("需要满足 0 < min_k <= max_k")


class AdaptiveSelector:
    """工具选择器：固定 Top-K 与自适应 Top-K。"""

    def __init__(self, config: Optional[Union[SelectionConfig, dict]] = None) -> None:
        """
        Args:
            config: SelectionConfig 或同名字典；为空时使用默认配置。
        """
        if config is None:
            self.config = SelectionConfig()
        elif isinstance(config, SelectionConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SelectionConfig(**config)
        else:
            raise TypeError("config 必须是 SelectionConfig、dict 或 None")

    def select(
        self,
        ranked_tools: Sequence["RetrievedTool"],
        complexity: Optional[str] = None,
        strategy: str = "fixed",
    ) -> List[str]:
        """根据选择策略返回最终工具集合。

        Args:
            ranked_tools: 检索器返回的已排序结果。
            complexity: 任务复杂度（simple / medium / complex），adaptive 策略使用。
            strategy: ``"fixed"`` 或 ``"adaptive"``。

        Returns:
            List[str]: 最终选中的工具名称列表（顺序与 ranked_tools 一致）。

        Raises:
            ValueError: strategy 不合法；adaptive 策略缺少 complexity 或 complexity 未知。
        """
        if strategy not in SELECTION_STRATEGIES:
            raise ValueError(f"不支持的策略: {strategy!r}，可选值: {SELECTION_STRATEGIES}")

        if strategy == "fixed":
            selected_count = self.config.top_k
        else:
            if complexity is None:
                raise ValueError("adaptive 策略必须提供 complexity（simple / medium / complex）")
            if complexity not in COMPLEXITY_TOP_K:
                raise ValueError(
                    f"未知的任务复杂度: {complexity!r}，可选值: {tuple(COMPLEXITY_TOP_K)}"
                )
            # 先按复杂度取初始 K，再用配置区间约束
            selected_count = COMPLEXITY_TOP_K[complexity]
            selected_count = max(self.config.min_k, min(selected_count, self.config.max_k))

        tools = list(ranked_tools)
        if not tools:
            logger.debug("检索结果为空，返回空工具集合（strategy=%s）", strategy)
            return []

        selected = tools[:selected_count]
        logger.debug(
            "工具选择完成：strategy=%s complexity=%s 选中 %d/%d 个工具",
            strategy,
            complexity,
            len(selected),
            len(tools),
        )
        return [tool.tool_name for tool in selected]


__all__ = ["AdaptiveSelector", "SelectionConfig", "SELECTION_STRATEGIES", "COMPLEXITY_TOP_K"]
