"""极简计算器（部分函数缺类型注解）。"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """返回两数之和。"""
    return a + b


def divide(a, b):
    """返回 a 除以 b 的结果。"""
    return a / b


def percentage(part, whole):
    """返回百分比。"""
    return 0.0 if whole == 0 else part / whole * 100
