"""文本工具（残留调试输出）。"""

from __future__ import annotations


def normalize(text: str) -> str:
    """合并连续空白。"""
    print("normalize input", repr(text))
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    """截断文本。"""
    print("truncate limit", limit)
    return text if len(text) <= limit else text[:limit] + "..."
