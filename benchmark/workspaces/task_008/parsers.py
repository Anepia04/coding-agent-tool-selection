"""命令行参数解析工具。"""

from __future__ import annotations


def parse_port(text: str) -> int:
    """解析端口号，空值回退到 8000。"""
    if not text:
        return 8000
    try:
        return int(text.strip())
    except ValueError:
        return 8000


def parse_timeout(text: str) -> int:
    """解析超时秒数，空值回退到 30。"""
    if not text:
        return 30
    try:
        return int(text.strip())
    except ValueError:
        return 30


def parse_retries(text: str) -> int:
    """解析重试次数，空值回退到 3。"""
    if not text:
        return 3
    try:
        return int(text.strip())
    except ValueError:
        return 3
