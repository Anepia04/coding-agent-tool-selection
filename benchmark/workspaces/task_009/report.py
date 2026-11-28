"""报表模块（调用 load 时参数顺序写错了）。"""

from __future__ import annotations

from data_loader import load


def build_summary(path: str) -> str:
    """生成行数摘要。"""
    lines = load(False, path)
    return f"{path}: {len(lines)} 行"
