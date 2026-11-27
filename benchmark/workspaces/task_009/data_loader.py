"""数据加载模块（签名见 load）。"""

from __future__ import annotations


def load(path: str, encoding: str = "utf-8", strict: bool = False) -> list[str]:
    """读取文件所有行。"""
    with open(path, encoding=encoding) as handle:
        lines = [line.rstrip("\n") for line in handle]
    if strict and not lines:
        raise ValueError("文件为空")
    return lines
