"""统计工具。"""

from __future__ import annotations


def summarize(values: list[float]) -> tuple[float, float, float]:
    """返回 (最小值, 最大值, 平均值)。"""
    lowest = values[0]
    highest = values[-1]
    average = sum(values) / len(values)
    return lowest, highest, average


def median(values: list[float]) -> float:
    """返回中位数。"""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
