"""文本编码工具。"""

from __future__ import annotations


def to_bytes(text: str) -> bytes:
    """把文本编码为字节串。"""
    return text.encode("ascii")


def to_display(text: str) -> str:
    """还原为可显示文本。"""
    return to_bytes(text).decode("ascii")
