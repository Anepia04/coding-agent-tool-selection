"""TF-IDF 工具检索器。

职责：根据任务描述与工具描述计算相关性，返回按相关性排序的工具列表。
当前阶段只定义接口与返回结构，不实现 TF-IDF / Cosine Similarity / 排序逻辑。

负责成员：成员 A
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from agent.task_analyzer import TaskInfo
    from tools.definitions import ToolDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedTool:
    """检索结果条目。

    Attributes:
        tool_name: 工具名称，需与 ToolDefinition.name 一致。
        score: 相关性分数（越大越相关），具体取值范围由实现决定。
        rank: 排名，从 1 开始，1 表示最相关。
    """

    tool_name: str
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {"tool_name": self.tool_name, "score": self.score, "rank": self.rank}


class TFIDFRetriever:
    """基于 TF-IDF 的工具检索器。"""

    def __init__(self) -> None:
        self._tool_definitions: List["ToolDefinition"] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """索引是否已构建。"""
        return self._is_fitted

    def fit(self, tool_definitions: Sequence["ToolDefinition"]) -> "TFIDFRetriever":
        """接收工具定义，建立检索索引。

        Args:
            tool_definitions: 工具定义列表，通常来自 ToolRegistry.get_all_tools()。

        Returns:
            self，便于链式调用。

        Raises:
            NotImplementedError: 当前阶段未实现。
        """
        # TODO(成员 A)：把 name + description + category 拼接为检索文档，
        # 使用 TF-IDF 向量化并缓存词表 / IDF 等索引结构。
        raise NotImplementedError("待成员 A 实现：TFIDFRetriever.fit 索引构建")

    def retrieve(
        self,
        task_info: Union["TaskInfo", str],
        top_k: Optional[int] = None,
    ) -> List[RetrievedTool]:
        """根据任务检索工具，返回按相关性降序排列的结果。

        Args:
            task_info: TaskInfo 或纯任务描述文本（作为查询）。
            top_k: 返回数量上限；为 None 时返回全部排序结果。

        Returns:
            List[RetrievedTool]: 已排序的检索结果，rank 从 1 开始。

        Raises:
            NotImplementedError: 当前阶段未实现。
        """
        # TODO(成员 A)：查询向量化 -> 计算 Cosine Similarity -> 降序排序
        # -> 截断 Top-K -> 包装为 RetrievedTool（rank 从 1 开始）。
        raise NotImplementedError("待成员 A 实现：TFIDFRetriever.retrieve 工具检索")


__all__ = ["TFIDFRetriever", "RetrievedTool"]
