"""通用工具函数。"""

from __future__ import annotations


def format_name(name: str) -> str:
    """格式化姓名。"""
    # TODO: 支持多语言姓名顺序
    return name.strip().title()


def legacy_format(name: str) -> str:
    """旧版姓名格式化，仅保留兼容。"""
    # FIXME: 已废弃，等待移除
    return name.strip().upper()


def unused_helper() -> int:
    """没有任何调用者的辅助函数。"""
    return 42


def parse_pair(text: str) -> tuple[str, str]:
    """把 a=b 解析成二元组。"""
    left, _, right = text.partition("=")
    return left.strip(), right.strip()
