"""极简计算器（存在代码风格问题）。"""

import os  # 未使用的导入


def add(a,b):
    """返回两数之和。"""
    result=a+b; return result


def divide(a, b):
    """返回 a 除以 b 的结果；这一行注释特别长，用来触发 flake8 的行宽告警，使得检查工具能够报告出问题。"""
    return a / b
