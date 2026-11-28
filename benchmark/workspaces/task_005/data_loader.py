"""数据加载模块（仍在依赖 legacy_io）。"""

from __future__ import annotations

from legacy_io import read_lines


def load_names(path: str) -> list[str]:
    """读取每行一个名字的文件。"""
    return [line.strip() for line in read_lines(path) if line.strip()]
