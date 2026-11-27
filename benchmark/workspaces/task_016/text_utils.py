"""文本工具（全部缺类型注解）。"""

from __future__ import annotations


def normalize(text):
    """合并连续空白。"""
    return " ".join(text.split())


def truncate(text, limit):
    """截断文本。"""
    return text if len(text) <= limit else text[:limit] + "..."
