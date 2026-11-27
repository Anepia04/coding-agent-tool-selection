"""配置加载模块。"""

from __future__ import annotations

import json
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """从 JSON 文件加载配置。"""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def get_port(config: dict[str, Any]) -> int:
    """取配置里的端口号。"""
    return config["port"]
