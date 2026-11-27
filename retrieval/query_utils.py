"""检索查询构造工具。

职责：在**分析阶段**把任务描述整理成检索器可用的查询词，包括：

- 否定约束清洗：丢弃「不要修改 / 禁止运行测试 / 保持代码不变」这类句子。

检索器本身不负责这些规则，只消费构造好的 ``retrieval_query``。
"""

from __future__ import annotations

import re
from typing import List

#: 查询分句符：否定约束通常以独立短句出现，按标点拆开后整句丢弃
_CLAUSE_SPLITTER = re.compile(r"[，,。；;！!？?\n\r]+")

#: 句首礼貌词，剥离后再判断否定前缀
_POLITE_PREFIX = re.compile(r"^(?:请|务必|千万|注意|记得)\s*")

#: 句首连接词，剥离后再判断否定前缀（如「也不要运行测试」）
_CONJUNCTION_PREFIX = re.compile(r"^(?:也|并且|并|而且|且|另外|同时|但|但是|不过|然后|再|还要)\s*")

#: 位于句首的否定词：丢弃整句
_NEGATION_PREFIXES = (
    "不要", "不需要", "不用", "无需", "无须", "不必",
    "禁止", "严禁", "不准", "不得", "不允许", "不能", "不可", "不应",
    "避免", "别", "勿", "切勿",
    "do not", "don't", "never", "without", "avoid", "no need",
)

#: 出现在句中的否定性约束：整句通常是纯约束，不含可执行的正向动作
_NEGATION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"保持.{0,8}(?:不变|原样|不动)",
        r"不改(?:动|变)?",
        r"不修改",
        r"不运行",
        r"不执行",
        r"不删除",
        r"不创建",
        r"不新增",
        r"不破坏",
    )
)


def sanitize_query(text: str) -> str:
    """剥离查询中的否定约束，只保留正向动作句。

    局限：隐含否定和整句均为否定的查询无法完全处理；若清洗后为空，
    则回退到原始查询。
    """
    if not isinstance(text, str):
        return ""
    clauses = [clause.strip() for clause in _CLAUSE_SPLITTER.split(text) if clause.strip()]
    kept: List[str] = []
    for clause in clauses:
        normalized = _POLITE_PREFIX.sub("", clause).strip()
        normalized = _CONJUNCTION_PREFIX.sub("", normalized).strip()
        normalized = _POLITE_PREFIX.sub("", normalized).strip().lower()
        if any(normalized.startswith(prefix) for prefix in _NEGATION_PREFIXES):
            continue
        if any(pattern.search(clause) for pattern in _NEGATION_PATTERNS):
            continue
        kept.append(clause)

    cleaned = "，".join(kept).strip()
    return cleaned or text


def build_retrieval_query(description: str) -> str:
    """由任务描述构造检索查询，保证返回非空。"""
    cleaned = sanitize_query(description).strip()
    return cleaned or str(description).strip()


__all__ = ["sanitize_query", "build_retrieval_query"]
