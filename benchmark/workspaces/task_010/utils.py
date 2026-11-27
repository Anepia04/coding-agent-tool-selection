"""通用工具函数（文本与数值混在一起）。"""

from __future__ import annotations


def normalize(text: str) -> str:
    """合并连续空白。"""
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    """截断文本。"""
    return text if len(text) <= limit else text[:limit] + "..."


def titlecase(text: str) -> str:
    """首字母大写。"""
    return text[:1].upper() + text[1:].lower() if text else text


def word_count(text: str) -> int:
    """统计单词数。"""
    return len(text.split())


def clamp(value: float, low: float, high: float) -> float:
    """把数值限制在区间内。"""
    return max(low, min(high, value))


def percent(part: float, whole: float) -> float:
    """计算百分比。"""
    return 0.0 if whole == 0 else part / whole * 100


def average(values: list[float]) -> float:
    """求平均值。"""
    return sum(values) / len(values) if values else 0.0


def round_to(value: float, digits: int) -> float:
    """四舍五入到指定小数位。"""
    return round(value, digits)
