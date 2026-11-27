"""文本处理工具。"""

from __future__ import annotations


def normalize(text: str) -> str:
    """去除首尾空白并合并连续空白。"""
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    """截断到指定长度，超出部分以省略号代替。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
