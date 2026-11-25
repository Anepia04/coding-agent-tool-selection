"""任务分析模块。

TaskAnalyzer 把用户的自然语言任务转换成结构化 ``TaskInfo``，供工具检索
（TF-IDF）与自适应工具选择（Adaptive Top-K）使用。

当前阶段只定义数据结构与接口，不实现任务分类 / 复杂度判断算法。

负责成员：成员 A
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from config import COMPLEXITY_LEVELS

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """结构化任务描述。

    Attributes:
        task_id: 任务唯一标识，与 benchmark/tasks.json 中的 task_id 对应。
        description: 原始自然语言任务描述（检索的主要输入）。
        task_type: 任务类型，如 bug_fix / feature_add / refactor / test_write / doc。
        language: 主要编程语言，如 python。
        intents: 任务意图关键词列表，供检索增强与自适应选择使用。
        complexity: 任务复杂度，取值见 ``config.COMPLEXITY_LEVELS``。
    """

    task_id: str
    description: str
    task_type: str = "unknown"
    language: str = "unknown"
    intents: List[str] = field(default_factory=list)
    complexity: str = "simple"

    def __post_init__(self) -> None:
        if self.complexity not in COMPLEXITY_LEVELS:
            raise ValueError(
                f"complexity 必须是 {COMPLEXITY_LEVELS} 之一，当前为 {self.complexity!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type,
            "language": self.language,
            "intents": list(self.intents),
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskInfo":
        """从字典（如 benchmark/tasks.json 中的一条任务）构造 TaskInfo。"""
        return cls(
            task_id=str(data["task_id"]),
            description=str(data.get("description", "")),
            task_type=str(data.get("task_type", "unknown")),
            language=str(data.get("language", "unknown")),
            intents=list(data.get("intents") or []),
            complexity=str(data.get("complexity", "simple")),
        )


class TaskAnalyzer:
    """任务分析器。

    实现方式（LLM 分析 / 规则匹配）由成员 A 后续决定；本阶段只固定输入输出。
    """

    def __init__(self, llm_client: Optional["LLMClient"] = None) -> None:
        """
        Args:
            llm_client: 可选的 LLM 客户端。若使用大模型做任务分析则传入，
                纯规则实现时可为 None。
        """
        self.llm_client = llm_client

    def analyze(self, task_description: str, task_id: Optional[str] = None) -> TaskInfo:
        """分析自然语言任务，返回结构化 TaskInfo。

        Args:
            task_description: 用户输入的自然语言任务描述。
            task_id: 可选任务 ID；为空时由实现生成临时 ID。

        Returns:
            TaskInfo: 结构化任务信息。

        Raises:
            NotImplementedError: 当前阶段未实现。
        """
        # TODO(成员 A)：识别 task_type / language / intents，
        # 并给出 complexity（simple / medium / complex）。
        raise NotImplementedError("待成员 A 实现：TaskAnalyzer.analyze 任务分析")


__all__ = ["TaskInfo", "TaskAnalyzer"]
