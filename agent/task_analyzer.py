"""任务分析模块。

TaskAnalyzer 把用户的自然语言任务转换成结构化 ``TaskInfo``，供工具检索
（TF-IDF）与自适应工具选择（Adaptive Top-K）使用。实现方式：通过 ``LLMClient``
发送带 JSON 输出约束的 Prompt，再解析、校验模型返回的结构化结果。

只负责「分析」，不负责检索、选择或执行工具。

负责成员：成员 A
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from config import COMPLEXITY_LEVELS

if TYPE_CHECKING:
    from .llm_client import LLMClient

logger = logging.getLogger(__name__)

#: 任务类型取值（与 docs/architecture.md 保持一致）
TASK_TYPES: tuple[str, ...] = ("bug_fix", "feature_add", "refactor", "test_write", "doc")

#: 提供给大模型的系统提示词：约束输出为纯 JSON 并限定枚举取值
SYSTEM_PROMPT = """你是一个资深的软件工程任务分析助手。请分析用户给出的编程任务，并只输出一个 JSON 对象（不要输出 Markdown 代码块或任何解释文字）。

JSON 字段要求：
- "description": 对任务的一句话复述（字符串）；
- "task_type": 任务类型，必须是以下之一：bug_fix（修复缺陷）、feature_add（新增功能）、refactor（重构）、test_write（编写测试）、doc（编写文档）；
- "language": 任务主要涉及的编程语言（如 python），无法确定时输出 "unknown"；
- "intents": 字符串数组，描述完成任务需要执行的操作类型，例如 ["读取代码", "修改函数", "运行测试"]；
- "complexity": 任务复杂度，必须是以下之一：simple、medium、complex。

示例：
{"description": "修复计算器的除零异常并补充测试", "task_type": "bug_fix", "language": "python", "intents": ["读取代码", "修改函数", "运行测试"], "complexity": "simple"}
"""


class TaskAnalysisError(RuntimeError):
    """任务分析失败（模型调用异常、返回非 JSON、字段缺失或取值非法）。"""


@dataclass
class TaskInfo:
    """结构化任务描述。

    Attributes:
        task_id: 任务唯一标识，与 benchmark/tasks.json 中的 task_id 对应。
        description: 原始自然语言任务描述（检索的主要输入）。
        task_type: 任务类型，取值见 ``TASK_TYPES``。
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
    """基于大模型的任务分析器。"""

    def __init__(self, llm_client: Optional["LLMClient"] = None) -> None:
        """
        Args:
            llm_client: 已配置好的 LLMClient；不传则无法执行分析（调用时抛错）。
        """
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(self, task_description: str, task_id: Optional[str] = None) -> TaskInfo:
        """分析自然语言任务，返回结构化 TaskInfo。

        Args:
            task_description: 用户输入的自然语言任务描述。
            task_id: 可选任务 ID；传入时优先使用，否则生成 ``task_xxxxxxxx`` 形式的默认 ID。

        Returns:
            TaskInfo: 结构化任务信息。

        Raises:
            ValueError: task_description 为空。
            TaskAnalysisError: 未提供 LLMClient、模型调用失败、返回内容不是合法 JSON
                对象，或字段缺失 / 取值非法。
        """
        if not isinstance(task_description, str) or not task_description.strip():
            raise ValueError("task_description 不能为空")
        if self.llm_client is None:
            raise TaskAnalysisError("TaskAnalyzer 需要配置 LLMClient 才能执行任务分析")

        description = task_description.strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请分析以下编程任务：\n{description}"},
        ]

        try:
            response = self.llm_client.chat(messages)
        except Exception as exc:  # LLMClientError / 网络异常等统一转换为分析错误
            raise TaskAnalysisError(f"调用大模型进行任务分析失败: {exc}") from exc

        content = getattr(response, "content", None)
        payload = self._parse_json_object(content)
        return self._build_task_info(payload, description, task_id)

    # ------------------------------------------------------------------
    # 解析与校验
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_object(content: Any) -> Dict[str, Any]:
        """从模型文本中提取 JSON 对象，兼容 ```json 代码块包裹的返回。"""
        if not isinstance(content, str) or not content.strip():
            raise TaskAnalysisError("模型未返回可解析的文本内容")
        text = content.strip()

        candidates = [text]
        if "```" in text:  # 去掉 Markdown 代码块围栏
            unfenced = text.replace("```json", "```").replace("```JSON", "```")
            parts = [part.strip() for part in unfenced.split("```") if part.strip()]
            candidates.extend(parts)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:  # 兜底：截取最外层大括号内容
            candidates.append(text[start:end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                raise TaskAnalysisError("模型返回的 JSON 必须是对象（dict）")
            return parsed

        raise TaskAnalysisError(f"模型返回内容不是合法 JSON: {content[:200]!r}")

    @staticmethod
    def _build_task_info(
        payload: Dict[str, Any],
        description: str,
        task_id: Optional[str],
    ) -> TaskInfo:
        """校验字段并组装 TaskInfo。"""
        task_type = payload.get("task_type")
        if not isinstance(task_type, str) or task_type not in TASK_TYPES:
            raise TaskAnalysisError(
                f"task_type 缺失或非法: {task_type!r}，可选值: {TASK_TYPES}"
            )

        complexity = payload.get("complexity")
        if not isinstance(complexity, str) or complexity not in COMPLEXITY_LEVELS:
            raise TaskAnalysisError(
                f"complexity 缺失或非法: {complexity!r}，可选值: {COMPLEXITY_LEVELS}"
            )

        language = payload.get("language")
        if not isinstance(language, str) or not language.strip():
            language = "unknown"
        else:
            language = language.strip()

        raw_intents = payload.get("intents", [])
        if raw_intents is None:
            raw_intents = []
        if not isinstance(raw_intents, list):
            raise TaskAnalysisError(f"intents 必须是字符串数组，实际为 {type(raw_intents).__name__}")
        intents = [str(intent).strip() for intent in raw_intents if str(intent).strip()]

        model_description = payload.get("description")
        if isinstance(model_description, str) and model_description.strip():
            description = model_description.strip()

        return TaskInfo(
            task_id=task_id or f"task_{uuid.uuid4().hex[:8]}",
            description=description,
            task_type=task_type,
            language=language,
            intents=intents,
            complexity=complexity,
        )


__all__ = ["TaskAnalyzer", "TaskInfo", "TaskAnalysisError", "TASK_TYPES", "SYSTEM_PROMPT"]
