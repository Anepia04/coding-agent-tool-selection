"""数据加载模块（每次都重新读文件）。"""

from __future__ import annotations


def load_records(path: str) -> list[dict]:
    """读取逗号分隔的记录。"""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            name, _, score = line.partition(",")
            records.append({"name": name, "score": int(score)})
    return records


def total_score(path: str) -> int:
    """汇总所有记录的分数。"""
    return sum(row["score"] for row in load_records(path))
