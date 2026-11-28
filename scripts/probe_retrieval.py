"""TF-IDF 工具检索诊断脚本。

为什么必须落成文件：
    中文若通过 PowerShell / cmd 管道传给 python，会被替换成 '?'，
    导致分词器收到空查询、所有工具分数为 0、排名退化成工具定义顺序。
    本脚本的中文全部来自 tasks.json / expected_tools.json / 模块，不经过管道。

用法：
    py scripts/probe_retrieval.py                      # 分析 tasks.json 全部任务
    py scripts/probe_retrieval.py --task task_001      # 只分析指定任务
    py scripts/probe_retrieval.py --query "把 utils.py 改名并移动到 lib"   # 自定义查询
    py scripts/probe_retrieval.py --show-tokens        # 显示查询分词与交集词
    py scripts/probe_retrieval.py --noise              # 统计跨工具共享词（低区分度噪声）
    py scripts/probe_retrieval.py --extra-tools candidates.json           # 扩充工具池后对比

每个任务的输出依次是：原始提示词、成功判据、检索查询、否定约束、必选工具，
然后是每个工具的分数 / 交集数 / 交集词。

把「原始提示词」和「检索查询」并排显示是有意为之：检索器只消费
``retrieval_query``，但判断「查询漏词」时必须对照原文，才能分清是任务本身
没提出该动作（如原文没写"测试"，但 success_criteria 要求测试通过），
还是提取阶段把动作词漏掉了。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.task_analyzer import TaskInfo  # noqa: E402
from retrieval.query_utils import build_retrieval_query  # noqa: E402
from retrieval.tfidf_retriever import TFIDFRetriever, _tokenize  # noqa: E402
from tools.definitions import EXAMPLE_TOOL_DEFINITIONS, ToolDefinition  # noqa: E402

TASKS_FILE = PROJECT_ROOT / "benchmark" / "tasks.json"
EXPECTED_FILE = PROJECT_ROOT / "benchmark" / "expected_tools.json"

CJK_LO, CJK_HI = "\u4e00", "\u9fff"


def setup_stdout() -> None:
    """让 Windows 控制台也能正确打印中文。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def cjk_tokens(text: str) -> Set[str]:
    """只保留含汉字的 token，过滤英文与标点碎片。

    **仅用于 ``--noise`` 的低区分度词统计**：英文 hint 数量少、跨工具共享性低，
    混进来会干扰判断。诊断交集请用 :func:`text_tokens`。
    """
    return {
        token
        for token in _tokenize(text)
        if any(CJK_LO <= ch <= CJK_HI for ch in token)
    }


def text_tokens(text: str) -> Set[str]:
    """保留全部分词结果（中文 + 英文 + 标识符）。

    诊断交集**必须**用它。英文 token 同样参与 TF-IDF 打分，之前这里误用了
    :func:`cjk_tokens`，导致 ``todo`` / ``fixme`` / ``diff`` 这类命中被悄悄
    过滤掉，看上去像「关键词根本没被扫出来」，实际是显示把证据藏了。
    """
    return set(_tokenize(text))


#: 直接复用检索器的文档构建逻辑，**不要**在这里复制一份。
#: 之前这里手写了一份 ``name + category + description``，result: 检索器加入
#: ``search_hints`` 后本脚本的诊断输出仍然是旧数据（交集词、噪声统计全部失真）。
#: 通过别名绑定真实实现，从此不会再漂移。
tool_document = TFIDFRetriever._build_document


def load_tasks() -> List[Dict[str, Any]]:
    data = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return data.get("tasks", data) if isinstance(data, dict) else data


def load_expected() -> Dict[str, Dict[str, Any]]:
    data = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))
    items = data.get("tasks", data) if isinstance(data, dict) else data
    return {str(item["task_id"]): item for item in items}


def load_extra_tools(path: Path) -> List[ToolDefinition]:
    """从 JSON 读取候选工具定义，格式与 ToolDefinition.to_dict() 一致。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("tools", raw) if isinstance(raw, dict) else raw
    tools = []
    for item in items:
        tools.append(
            ToolDefinition(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                category=str(item.get("category", "other")),
                parameters=item.get("parameters") or {"type": "object", "properties": {}},
                search_hints=list(item.get("search_hints") or []),
            )
        )
    return tools


def report_noise(definitions: List[ToolDefinition]) -> None:
    """统计在多个工具描述里重复出现的中文 token——这些词 IDF 低，几乎无区分力。"""
    counter: Counter = Counter()
    per_tool = {}
    for definition in definitions:
        tokens = cjk_tokens(tool_document(definition))
        per_tool[definition.name] = tokens
        counter.update(tokens)

    total_tools = len(definitions)
    threshold = max(3, total_tools // 2)
    print(f"工具总数 {total_tools}，出现在 >= {threshold} 个描述中的中文 token：")
    noisy = [(token, count) for token, count in counter.items() if count >= threshold]
    if not noisy:
        print("  （无）")
    for token, count in sorted(noisy, key=lambda kv: (-kv[1], kv[0])):
        print(f"  {token:8} 出现在 {count}/{total_tools} 个工具描述中")

    print(f"\n各工具描述的中文 token 数 / 其中噪声词数（>= {threshold} 个工具共享）：")
    noisy_set = {token for token, _ in noisy}
    for name, tokens in per_tool.items():
        print(f"  {name:14} 总数 {len(tokens):3} | 噪声 {len(tokens & noisy_set):3}")


def analyze(
    label: str,
    query: str,
    retriever: TFIDFRetriever,
    definitions: List[ToolDefinition],
    required: Set[str],
    docs: Dict[str, Set[str]],
    show_tokens: bool,
    description: str = "",
    negative_constraints: Optional[List[str]] = None,
    success_criteria: str = "",
) -> None:
    """打印单个任务的诊断详情。

    Args:
        query: 实际用于检索的查询串（``retrieval_query``）。
        description: 任务的**原始提示词**。它不参与检索（检索器只消费
            ``retrieval_query``），但诊断时必须一起看，否则无法判断
            「查询漏词」是任务本身没要求，还是提取时被漏掉了。
        negative_constraints: 任务明确禁止的操作，供人工核对是否被清洗逻辑误伤。
        success_criteria: 成功判据，常含 description 里没写但任务隐含要求的动作。
    """
    info = TaskInfo(
        task_id="probe",
        description=description or query,
        retrieval_query=build_retrieval_query(query),
    )
    ranked = retriever.retrieve(info)
    names = [item.tool_name for item in ranked]

    print(f"\n=== {label} ===")
    if description:
        print(f"原始提示词：{description}")
    if success_criteria:
        print(f"成功判据　：{success_criteria}")
    same_as_description = description and query.strip() == description.strip()
    suffix = "  （任务未提供 retrieval_query，回退为原始提示词）" if same_as_description else ""
    print(f"检索查询　：{query}{suffix}")
    if negative_constraints:
        print(f"否定约束　：{negative_constraints}")
    if required:
        print(f"必选工具（{len(required)} 个）：{sorted(required)}")

    query_tokens = text_tokens(query)
    if show_tokens:
        print(f"查询 token（{len(query_tokens)} 个，含英文）：{sorted(query_tokens)}")

    print(f"\n{'排名':<5}{'工具':<16}{'分数':<10}{'交集数':<8}交集词")
    print("-" * 78)
    for item in ranked:
        overlap = query_tokens & docs.get(item.tool_name, set())
        mark = "*" if item.tool_name in required else " "
        shown = sorted(overlap)
        detail = ", ".join(shown[:10]) + ("…" if len(shown) > 10 else "")
        print(
            f"{mark}{item.rank:<4}{item.tool_name:<16}{item.score:<10.4f}"
            f"{len(overlap):<8}{detail}"
        )

    if required:
        for k in (3, 5, len(names)):
            hit = len(set(names[:k]) & required)
            print(f"recall@{k} = {hit}/{len(required)} = {hit / len(required):.2f}")

    missing = [name for name in required if name not in names[:5]]
    if missing:
        print(f"前 5 名遗漏的必选工具：{sorted(missing)}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TF-IDF 工具检索诊断")
    parser.add_argument("--task", default=None, help="只分析指定 task_id")
    parser.add_argument("--query", default=None, help="自定义检索查询文本")
    parser.add_argument("--show-tokens", action="store_true", help="显示分词与交集细节")
    parser.add_argument("--noise", action="store_true", help="统计跨工具共享的低区分度词")
    parser.add_argument("--extra-tools", default=None, help="候选工具 JSON，扩充工具池后对比")
    args = parser.parse_args(argv)

    setup_stdout()

    definitions = list(EXAMPLE_TOOL_DEFINITIONS)
    extra: List[ToolDefinition] = []
    if args.extra_tools:
        extra = load_extra_tools(Path(args.extra_tools))
        print(f"已从 {args.extra_tools} 载入 {len(extra)} 个候选工具")
        definitions = definitions + extra

    docs = {definition.name: text_tokens(tool_document(definition)) for definition in definitions}
    retriever = TFIDFRetriever().fit(definitions)

    if args.noise:
        report_noise(definitions)
        print()

    if args.query:
        analyze(
            "自定义查询",
            args.query,
            retriever,
            definitions,
            set(),
            docs,
            args.show_tokens,
            # 自定义查询没有原始提示词可对照
            description="",
        )
        return 0

    expected = load_expected()
    tasks = load_tasks()
    for task in tasks:
        task_id = str(task["task_id"])
        if args.task and task_id != args.task:
            continue
        required = set(expected.get(task_id, {}).get("required_tools", []))
        label = f"任务 {task_id}（complexity={task.get('complexity')}）"
        description = str(task.get("description") or "")
        analyze(
            label,
            str(task.get("retrieval_query") or description),
            retriever,
            definitions,
            required,
            docs,
            args.show_tokens,
            description=description,
            negative_constraints=[
                str(item) for item in (task.get("negative_constraints") or [])
            ],
            success_criteria=str(task.get("success_criteria") or ""),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())