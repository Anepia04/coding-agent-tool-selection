"""参数解析工具。"""

from __future__ import annotations


def parse_int(text: str, default: int = 0) -> int:
    """解析整数，失败回退默认值。"""
    print("parse_int", text)
    try:
        return int(text.strip())
    except (AttributeError, ValueError):
        print("parse_int fallback")
        return default
