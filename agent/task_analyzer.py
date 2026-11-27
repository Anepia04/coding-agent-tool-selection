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
from retrieval.query_utils import build_retrieval_query, sanitize_query

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
- "retrieval_query": 从任务描述中抽取的**正向**检索关键词，用空格分隔，只写任务要执行的动作和对象。
  然后推理完成任务的流程所需的全部动作，先思考流程，再将每一步的动作写入（如 修复代码问题 除了需要修改文件，也隐含读取文件的需求）；
  禁止出现「不要 / 禁止 / 无需」等否定词，禁止写工具名（如 read_file），否定要求放到 negative_constraints；
- "negative_constraints": 字符串数组，任务明确禁止的操作，例如 ["不修改其他模块"]；没有则为 []；
- "complexity": 任务复杂度，必须是以下之一：simple、medium、complex。

示例：
{"description": "修复计算器的除零异常并补充测试", "task_type": "bug_fix", "language": "python", "retrieval_query": "计算器 除零异常 修复 修改代码 单元测试", "negative_constraints": [], "complexity": "simple"}
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
        retrieval_query: 面向检索器的正向关键词（不含否定词与工具名）；分析阶段保证非空。
        negative_constraints: 任务明确禁止的操作，供评价或工具过滤使用。
        complexity: 任务复杂度，取值见 ``config.COMPLEXITY_LEVELS``。
    """

    task_id: str
    description: str
    task_type: str = "unknown"
    language: str = "unknown"
    retrieval_query: str = ""
    negative_constraints: List[str] = field(default_factory=list)
    complexity: str = "simple"

    def __post_init__(self) -> None:
        if self.complexity not in COMPLEXITY_LEVELS:
            raise ValueError(
                f"complexity 必须是 {COMPLEXITY_LEVELS} 之一，当前为 {self.complexity!r}"
            )
        if not isinstance(self.retrieval_query, str):
            raise ValueError("retrieval_query 必须是字符串")
        if not isinstance(self.negative_constraints, list):
            raise ValueError("negative_constraints 必须是字符串列表")

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type,
            "language": self.language,
            "retrieval_query": self.retrieval_query,
            "negative_constraints": list(self.negative_constraints),
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskInfo":
        """从字典（如 benchmark/tasks.json 中的一条任务）构造 TaskInfo。"""
        raw_negative = data.get("negative_constraints") or []
        negative_constraints = (
            [str(item).strip() for item in raw_negative if str(item).strip()]
            if isinstance(raw_negative, list)
            else []
        )
        retrieval_query = data.get("retrieval_query") or ""
        if not isinstance(retrieval_query, str):
            retrieval_query = ""

        return cls(
            task_id=str(data["task_id"]),
            description=str(data.get("description", "")),
            task_type=str(data.get("task_type", "unknown")),
            language=str(data.get("language", "unknown")),
            retrieval_query=retrieval_query.strip(),
            negative_constraints=negative_constraints,
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

        raw_query = payload.get("retrieval_query")
        if raw_query is None:
            retrieval_query = ""
        elif not isinstance(raw_query, str):
            raise TaskAnalysisError(
                f"retrieval_query 必须是字符串，实际为 {type(raw_query).__name__}"
            )
        else:
            # 即使模型违规带入了否定句，也在这里清洗掉
            retrieval_query = sanitize_query(raw_query).strip()

        raw_negative = payload.get("negative_constraints", [])
        if raw_negative is None:
            raw_negative = []
        if not isinstance(raw_negative, list):
            raise TaskAnalysisError(
                f"negative_constraints 必须是字符串数组，实际为 {type(raw_negative).__name__}"
            )
        negative_constraints = [
            str(item).strip() for item in raw_negative if str(item).strip()
        ]

        model_description = payload.get("description")
        if isinstance(model_description, str) and model_description.strip():
            description = model_description.strip()

        # 分析阶段保证检索查询非空：模型没给或被清洗为空时，从描述构造
        if not retrieval_query:
            retrieval_query = build_retrieval_query(description)

        return TaskInfo(
            task_id=task_id or f"task_{uuid.uuid4().hex[:8]}",
            description=description,
            task_type=task_type,
            language=language,
            retrieval_query=retrieval_query,
            negative_constraints=negative_constraints,
            complexity=complexity,
        )


__all__ = ["TaskAnalyzer", "TaskInfo", "TaskAnalysisError", "TASK_TYPES", "SYSTEM_PROMPT"]
