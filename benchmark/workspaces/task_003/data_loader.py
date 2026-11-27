"""数据加载模块。"""

from __future__ import annotations


def load_records(path: str) -> list[dict]:
    """从逗号分隔的文本文件加载记录（3 列格式）。"""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                name = parts[0].strip()
                age = int(parts[1].strip())
                city = parts[2].strip()
                records.append({"name": name, "age": age, "city": city})
            elif len(parts) == 2:
                name = parts[0].strip()
                age = int(parts[1].strip())
                records.append({"name": name, "age": age, "city": ""})
    return records


def load_scores(path: str) -> list[dict]:
    """从逗号分隔的文本文件加载成绩（同样 3 列格式）。"""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) == 3:
                name = parts[0].strip()
                age = int(parts[1].strip())
                city = parts[2].strip()
                records.append({"name": name, "age": age, "city": city})
            elif len(parts) == 2:
                name = parts[0].strip()
                age = int(parts[1].strip())
                records.append({"name": name, "age": age, "city": ""})
    return records
