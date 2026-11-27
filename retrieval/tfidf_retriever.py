"""TF-IDF 工具检索器。

实现要点：
- 使用 ``sklearn.feature_extraction.text.TfidfVectorizer`` 对工具描述建立索引；
- 中文不依赖默认分词（默认按空格切分对中文无效），改为「单字 + 相邻双字」切分，
  英文/标识符按单词切分（``read_file`` 同时保留整体与下划线子词）；
- 查询（任务描述）与工具矩阵之间使用 Cosine Similarity 计算相关性；
- 查询由分析阶段构造（``retrieval_query``）：检索器只消费现成查询词，
  不做否定清洗，也不回退到 description；
- 结果按分数降序排列，分数相同时保持 ``fit()`` 时的工具注册顺序（稳定排序）。

负责成员：成员 A
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from agent.task_analyzer import TaskInfo
    from tools.definitions import ToolDefinition

logger = logging.getLogger(__name__)

#: 连续中文字符片段（基本汉字 + 扩展 A 区）
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

#: 英文单词 / 标识符 / 数字
_WORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")



def _tokenize(text: str) -> List[str]:
    """轻量中英混合分词器（供 TfidfVectorizer 使用）。

    规则：
    - 英文单词、标识符、数字整体保留，并统一转小写；
    - 含下划线的标识符额外拆分出子词（``read_file`` -> ``read``、``file``）；
    - 连续中文片段同时输出单字与相邻双字，例如「运行测试」-> 运/行/测/试/运行/行测/测试。

    采用字级 n-gram 而非中文分词器，可以避免依赖额外词典，同时保留较好的中文匹配能力。
    """
    if text is None:
        return []
    text = str(text)

    tokens: List[str] = []
    for word in _WORD_PATTERN.findall(text):
        word = word.lower()
        tokens.append(word)
        if "_" in word:
            tokens.extend(part for part in word.split("_") if part)

    for segment in _CJK_PATTERN.findall(text):
        tokens.extend(segment[index:index + 2] for index in range(len(segment) - 1))

    return tokens


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
    """基于 TF-IDF + Cosine Similarity 的工具检索器。"""

    def __init__(self) -> None:
        self._tool_definitions: List["ToolDefinition"] = []
        self._tool_names: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tool_matrix = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """索引是否已构建。"""
        return self._is_fitted

    def fit(self, tool_definitions: Sequence["ToolDefinition"]) -> "TFIDFRetriever":
        """接收工具定义，建立检索索引。

        每个工具会拼接为「名称 + 分类 + 描述」的检索文档，再做 TF-IDF 向量化。

        Args:
            tool_definitions: 工具定义列表，通常来自 ToolRegistry.get_all_tools()。

        Returns:
            self，便于链式调用。

        Raises:
            ValueError: 工具库为空，或所有工具描述都无法提取有效词项。
        """
        definitions = list(tool_definitions)
        if not definitions:
            raise ValueError("工具库为空，无法建立 TF-IDF 索引")

        documents = [self._build_document(definition) for definition in definitions]
        vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            lowercase=False,
            token_pattern=None,
            sublinear_tf=True,
        )
        try:
            tool_matrix = vectorizer.fit_transform(documents)
        except ValueError as exc:  # 例如所有描述都为空，词表为空
            raise ValueError("工具描述无法提取有效词项（词表为空），请检查工具描述") from exc

        self._tool_definitions = definitions
        self._tool_names = [definition.name for definition in definitions]
        self._vectorizer = vectorizer
        self._tool_matrix = tool_matrix
        self._is_fitted = True
        logger.info(
            "TF-IDF 索引构建完成：%d 个工具，%d 个词项",
            len(definitions),
            len(vectorizer.vocabulary_),
        )
        return self

    def retrieve(
        self,
        task_info: Union["TaskInfo", str],
        top_k: Optional[int] = None,
    ) -> List[RetrievedTool]:
        """根据任务检索工具，返回按相关性降序排列的结果。

        Args:
            task_info: TaskInfo 或纯任务描述文本（作为查询）。
            top_k: 返回数量上限；为 None 时返回全部排序结果；
                超过工具总数时返回全部。

        Returns:
            List[RetrievedTool]: 已排序的检索结果，rank 从 1 开始。

        Raises:
            RuntimeError: 尚未调用 fit() 建立索引。
            TypeError: task_info 既不是字符串也不包含 description 字段。
            ValueError: top_k 不是正整数。
        """
        if not self._is_fitted or self._vectorizer is None or self._tool_matrix is None:
            raise RuntimeError("TF-IDF 索引尚未构建，请先调用 fit() 建立工具索引")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k 必须为正整数或 None")

        query = self._build_query(task_info)
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._tool_matrix)[0]

        # sorted 为稳定排序：分数相同时保持工具注册顺序
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        if top_k is not None:
            order = order[:top_k]

        return [
            RetrievedTool(
                tool_name=self._tool_names[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _build_document(definition: "ToolDefinition") -> str:
        """把工具定义拼接成用于向量化的检索文档。"""
        # description 只面向模型，不参与工具检索；search_hints 专门承载
        # 任务中可能出现的场景词、同义词和动作词。
        parts = [definition.name, definition.category]
        hints = getattr(definition, "search_hints", None) or []
        if hints:
            parts.append(" ".join(str(hint) for hint in hints))
        elif getattr(definition, "description", None):
            # 兼容尚未迁移 search_hints 的旧工具定义；新定义应始终提供 hints。
            parts.append(str(definition.description))
        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def _build_query(task_info: Union["TaskInfo", str]) -> str:
        """把查询输入统一转换为检索文本。

        - 字符串：视为已构造好的查询，原样使用；
        - TaskInfo：必须已经包含非空 ``retrieval_query``（由分析阶段生成），
          检索器只使用 ``retrieval_query``，不做否定清洗，也不回退 description。

        Raises:
            ValueError: TaskInfo 的 ``retrieval_query`` 为空。
        """
        if isinstance(task_info, str):
            return task_info

        description = getattr(task_info, "description", None)
        if description is None:
            raise TypeError("task_info 必须是查询字符串，或包含 description 字段的 TaskInfo")

        retrieval_query = str(getattr(task_info, "retrieval_query", "") or "").strip()
        if not retrieval_query:
            raise ValueError(
                "TaskInfo.retrieval_query 为空：请先在分析阶段生成检索查询"
            )

        return retrieval_query


__all__ = ["TFIDFRetriever", "RetrievedTool"]
