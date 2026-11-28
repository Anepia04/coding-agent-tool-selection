"""极简计算器。"""

from __future__ import annotations


def add(a: float, b: float) -> float:
    """返回两数之和。"""
    print("add called", a, b)
    result = a + b
    print("add result", result)
    return result


def divide(a: float, b: float) -> float:
    """返回 a 除以 b 的结果。"""
    print("divide called")
    return a / b
