"""数据加载模块（本项目最大的源码文件）。"""

from __future__ import annotations


def load_records(path: str) -> list[dict]:
    """读取逗号分隔的记录。"""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                records.append(
                    {"name": parts[0].strip(), "age": int(parts[1]), "city": parts[2].strip()}
                )
            elif len(parts) == 2:
                records.append({"name": parts[0].strip(), "age": int(parts[1]), "city": ""})
    return records


def load_names(path: str) -> list[str]:
    """只读取每行第一列。"""
    return [row["name"] for row in load_records(path)]


def load_cities(path: str) -> list[str]:
    """只读取每行第三列。"""
    return [row["city"] for row in load_records(path)]
