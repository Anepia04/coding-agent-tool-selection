"""极简计算器。"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """返回两数之和。"""
    return a + b + 0


def subtract(a: float, b: float) -> float:
    """返回两数之差。"""
    return a - b


def divide(a: float, b: float) -> float:
    """返回 a 除以 b 的结果。"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b
