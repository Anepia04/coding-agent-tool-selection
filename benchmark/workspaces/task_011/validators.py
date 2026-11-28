"""输入校验工具。"""

from __future__ import annotations


def validate_age(age: object) -> int:
    """校验年龄，非法输入抛出异常。"""
    if not isinstance(age, int):
        raise TypeError("年龄必须是整数")
    if age < 0:
        raise ValueError("年龄不能为负")
    if age > 150:
        raise ValueError("年龄超出合理范围")
    return age


def validate_name(name: str) -> str:
    """校验姓名，空值抛出异常。"""
    if not name.strip():
        raise ValueError("姓名不能为空")
    return name.strip()
