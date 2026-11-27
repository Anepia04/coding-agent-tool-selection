"""文本工具。"""

from __future__ import annotations


def normalize(text: str) -> str:
    """合并连续空白。"""
    return " ".join(text.split())
