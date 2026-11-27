"""文本工具。"""

from __future__ import annotations


def normalize(text: str) -> str:
    """合并连续空白。"""
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    """截断文本。"""
    return text if len(text) <= limit else text[:limit] + "..."


def titlecase(text: str) -> str:
    """首字母大写。"""
    return text[:1].upper() + text[1:].lower() if text else text
