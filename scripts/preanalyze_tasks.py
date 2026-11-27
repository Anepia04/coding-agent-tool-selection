"""Benchmark 任务预标注脚本（一次性，手动执行）。

对 ``benchmark/tasks.json`` 中尚未包含分析结果的任务调用 TaskAnalyzer，
把 ``retrieval_query / negative_constraints`` 写回 JSON。
实验运行时只读取这些字段，不再调用 LLM 分析，从而保证三种实验模式、
多次重复运行拿到的 TaskInfo 完全一致。

用法::

    python scripts/preanalyze_tasks.py            # 只补缺失字段
    python scripts/preanalyze_tasks.py --force    # 重新生成所有任务
    python scripts/preanalyze_tasks.py --dry-run  # 只打印，不写文件

规则：
- 只写上述两个字段，不覆盖已有的 description / task_type / complexity 等人工标注；
- retrieval_query 只从 description 生成，脚本不读取 expected_tools；
- 默认跳过已标注任务，避免覆盖人工复核结果（需要重算时用 --force）；
- 输出告警：空检索词、含否定词、含工具名。

退出码：0 成功；2 缺少 API Key；3 存在分析失败的任务。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import LLMClient  # noqa: E402
from agent.task_analyzer import TaskAnalyzer  # noqa: E402
from config import get_settings  # noqa: E402
from retrieval.query_utils import sanitize_query  # noqa: E402
from tools.registry import build_default_registry  # noqa: E402

DEFAULT_TASKS_PATH = PROJECT_ROOT / "benchmark" / "tasks.json"
ANALYSIS_FIELDS = ("retrieval_query", "negative_constraints")


def needs_analysis(task: Dict[str, Any]) -> bool:
    """任务是否缺少必要的预标注字段。"""
    return not str(task.get("retrieval_query") or "").strip()


def validate_analysis(task: Dict[str, Any], tool_names: set[str]) -> List[str]:
    """检查预标注字段是否符合数据规范，返回告警列表。"""
    warnings: List[str] = []
    task_id = task.get("task_id", "<unknown>")
    query = str(task.get("retrieval_query") or "").strip()
    if not query:
        warnings.append(f"{task_id}: retrieval_query 为空，检索将无法执行，请重新分析")
        return warnings

    sanitized = sanitize_query(query)
    if sanitized != query:
        warnings.append(
            f"{task_id}: retrieval_query 含否定约束，建议移到 negative_constraints: {query!r}"
        )

    lowered = query.lower()
    leaked = [name for name in tool_names if name.lower() in lowered]
    if leaked:
        warnings.append(
            f"{task_id}: retrieval_query 出现工具名 {leaked}，会造成答案泄漏"
        )
    return warnings


def load_tasks(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """读取任务文件，兼容顶层列表与 {"tasks": [...]} 两种格式。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, data
    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        return data, data["tasks"]
    raise ValueError(f"{path} 格式不正确：应为任务列表或包含 tasks 列表的对象")


def write_tasks(path: Path, payload: Any) -> None:
    """原子写回 JSON，保持可读缩进与中文不转义。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark 任务预标注")
    parser.add_argument("--path", type=Path, default=DEFAULT_TASKS_PATH, help="tasks.json 路径")
    parser.add_argument("--force", action="store_true", help="重新生成所有任务的分析字段")
    parser.add_argument("--dry-run", action="store_true", help="只打印结果，不写文件")
    args = parser.parse_args(argv)

    settings = get_settings()
    if not settings.model.api_key:
        print("错误：缺少 LLM_API_KEY，请在项目根目录 .env 中配置。", file=sys.stderr)
        return 2

    llm_client = LLMClient(
        model=settings.model.model,
        api_key=settings.model.api_key,
        base_url=settings.model.base_url,
        temperature=settings.model.temperature,
    )
    analyzer = TaskAnalyzer(llm_client)
    tool_names = set(build_default_registry().tool_names)

    payload, tasks = load_tasks(args.path)
    failed: List[str] = []
    warnings: List[str] = []
    processed = 0

    for task in tasks:
        task_id = str(task.get("task_id", ""))
        if not task_id or not str(task.get("description") or "").strip():
            failed.append(f"{task_id or '<no id>'}: 缺少 task_id 或 description")
            continue
        if not args.force and not needs_analysis(task):
            warnings.extend(validate_analysis(task, tool_names))
            continue

        try:
            info = analyzer.analyze(task["description"], task_id=task_id)
        except Exception as exc:  # noqa: BLE001 - 批量脚本逐条记录失败
            failed.append(f"{task_id}: {exc}")
            continue

        task["retrieval_query"] = info.retrieval_query
        task["negative_constraints"] = list(info.negative_constraints)
        processed += 1

        print(f"[已分析] {task_id}")
        print(f"  retrieval_query: {task['retrieval_query']}")
        print(f"  negative_constraints: {task['negative_constraints']}")
        warnings.extend(validate_analysis(task, tool_names))

    if not args.dry_run and processed:
        write_tasks(args.path, payload)
        print(f"\n已写回 {args.path}（更新 {processed} 条）")
    elif args.dry_run:
        print("\n[dry-run] 未写文件")

    if warnings:
        print("\n---- 告警 ----")
        for warning in warnings:
            print(f"- {warning}")

    if failed:
        print("\n---- 失败 ----", file=sys.stderr)
        for item in failed:
            print(f"- {item}", file=sys.stderr)
        return 3

    print(f"\n完成：分析 {processed} 条，跳过 {len(tasks) - processed - len(failed)} 条已标注任务。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
