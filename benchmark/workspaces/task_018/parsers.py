"""参数解析工具。"""

from __future__ import annotations


def parse_int(text: str, default: int = 0) -> int:
    """解析整数，失败回退默认值。"""
    try:
        return int(text.strip())
    except (AttributeError, ValueError):
        return default
