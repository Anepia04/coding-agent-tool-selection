"""已废弃的 IO 工具，保留仅为兼容旧脚本。"""

from __future__ import annotations


def read_lines(path: str) -> list[str]:
    """按行读取文件内容。"""
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def write_lines(path: str, lines: list[str]) -> None:
    """把若干行写入文件。"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
