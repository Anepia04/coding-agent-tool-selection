"""报表模块（同样依赖 legacy_io）。"""

from __future__ import annotations

from legacy_io import write_lines


def save_report(path: str, rows: list[str]) -> None:
    """把报表行写入文件。"""
    write_lines(path, rows)
