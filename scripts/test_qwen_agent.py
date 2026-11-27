"""Qwen4-Max 真实 Function Calling / 多轮 Agent 联调脚本（手动执行）。

覆盖三种实验模式：
    baseline      -> 全部临时工具
    fixed_topk    -> TF-IDF 检索 + 固定 Top-K（K=3）
    adaptive_topk -> TF-IDF 检索 + 按任务复杂度自适应（simple -> 3）

每个模式均验证：工具筛选数量、模型发起 ToolCall、ToolRegistry 执行、
结果回填、模型二次调用生成最终回答、正常终止与 Token 累计。

工具仅在脚本内部注册，全部只读（不写文件、不执行 Shell 命令），不进入正式工具库。

用法：
    python scripts/test_qwen_agent.py

退出码：0 全部通过；2 缺少 API Key；3 调用失败；4 未正常结束；
        5 未发生工具调用；6 工具筛选数量不符合配置。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import AgentController, AgentRunResult  # noqa: E402
from agent.llm_client import LLMClient  # noqa: E402
from agent.task_analyzer import TaskAnalyzer, TaskInfo  # noqa: E402
from config import get_settings  # noqa: E402
from retrieval.adaptive_selector import AdaptiveSelector, SelectionConfig  # noqa: E402
from retrieval.tfidf_retriever import TFIDFRetriever  # noqa: E402
from tools.definitions import ToolDefinition  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402

#: 固定 Top-K 模式的 K（小于工具总数，便于观察筛选效果）
FIXED_TOP_K = 3

_EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache"}


# ---------------------------------------------------------------------------
# 只读临时工具实现
# ---------------------------------------------------------------------------

def _project_python_files() -> List[Path]:
    return [
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not (_EXCLUDED_PARTS & set(path.parts))
    ]


def get_project_status() -> dict:
    """项目整体状态：文件数量与 benchmark 就绪情况。"""
    py_files = _project_python_files()
    return {
        "project": PROJECT_ROOT.name,
        "python_files": len(py_files),
        "test_files": len([p for p in py_files if p.name.startswith("test_")]),
        "benchmark_ready": (PROJECT_ROOT / "benchmark" / "tasks.json").exists(),
        "status": "ready",
    }


def get_git_branch() -> dict:
    """只读读取 .git/HEAD 获取当前分支名称。"""
    head_file = PROJECT_ROOT / ".git" / "HEAD"
    try:
        content = head_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return {"error": f"无法读取 .git/HEAD: {exc}"}
    if content.startswith("ref: refs/heads/"):
        return {"branch": content[len("ref: refs/heads/"):], "detached": False}
    return {"branch": content[:12], "detached": True}


def list_python_modules() -> dict:
    """列出项目中的 Python 文件相对路径（最多 20 个）。"""
    paths = sorted(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in _project_python_files()
    )
    return {"total": len(paths), "modules": paths[:20]}


def check_benchmark_tasks() -> dict:
    """检查 benchmark/tasks.json 并返回任务数量。"""
    tasks_file = PROJECT_ROOT / "benchmark" / "tasks.json"
    try:
        data = json.loads(tasks_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"exists": False, "error": str(exc)}
    tasks = data.get("tasks", data) if isinstance(data, dict) else data
    return {"exists": True, "task_count": len(tasks) if isinstance(tasks, list) else 0}


def get_test_summary() -> dict:
    """统计测试文件数量与 def test_ 方法数量。"""
    test_files = [p for p in _project_python_files() if p.name.startswith("test_")]
    method_count = 0
    for path in test_files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        method_count += len(re.findall(r"^\s*def test_", text, flags=re.MULTILINE))
    return {"test_files": len(test_files), "test_methods": method_count}


# ---------------------------------------------------------------------------
# 注册表 / 检索器 / 任务
# ---------------------------------------------------------------------------

TOOL_SPECS: List[Tuple[ToolDefinition, object]] = [
    (
        ToolDefinition(
            name="get_project_status",
            description=(
                "获取当前项目的整体只读状态：项目名称、Python 文件数量、测试文件数量、"
                "benchmark 是否就绪。无需参数，不会修改任何文件。"
            ),
            category="other",
            parameters={"type": "object", "properties": {}},
        ),
        get_project_status,
    ),
    (
        ToolDefinition(
            name="get_git_branch",
            description="读取当前 Git 分支名称（只读读取 .git/HEAD，不执行 git 命令）。",
            category="other",
            parameters={"type": "object", "properties": {}},
        ),
        get_git_branch,
    ),
    (
        ToolDefinition(
            name="list_python_modules",
            description="列出项目中的 Python 文件相对路径，最多返回 20 个（只读目录扫描）。",
            category="file",
            parameters={"type": "object", "properties": {}},
        ),
        list_python_modules,
    ),
    (
        ToolDefinition(
            name="check_benchmark_tasks",
            description="检查 benchmark/tasks.json 是否存在，并返回其中的任务数量（只读）。",
            category="file",
            parameters={"type": "object", "properties": {}},
        ),
        check_benchmark_tasks,
    ),
    (
        ToolDefinition(
            name="get_test_summary",
            description="统计项目中的测试文件数量，以及 def test_ 测试方法的总数（只读）。",
            category="test",
            parameters={"type": "object", "properties": {}},
        ),
        get_test_summary,
    ),
]


def build_registry() -> ToolRegistry:
    """创建只注册临时只读工具的注册表。"""
    registry = ToolRegistry()
    for definition, implementation in TOOL_SPECS:
        registry.register_tool(definition, implementation)
    return registry


def build_retriever(registry: ToolRegistry) -> TFIDFRetriever:
    """用真实 ToolDefinition 建立 TF-IDF 索引。"""
    retriever = TFIDFRetriever()
    retriever.fit(registry.get_all_tools())
    return retriever


def build_task() -> TaskInfo:
    """构造一个需要先调用工具才能回答的任务。"""
    return TaskInfo(
        task_id="smoke_qwen_fc",
        description=(
            "请先调用可用的只读工具获取项目状态，包括当前的 Python 文件数量和测试文件数量，"
            "然后严格根据工具返回的结果，用中文简要总结这两项数据。"
            "必须先调用工具，不要凭空编造数据。"
        ),
        task_type="doc",
        language="python",
        complexity="simple",
    )


# ---------------------------------------------------------------------------
# 运行与输出
# ---------------------------------------------------------------------------

def print_steps(result: AgentRunResult) -> None:
    """打印逐轮执行记录。"""
    print("---- 执行记录 ----")
    if not result.steps:
        print("<无步骤>")
        return
    for step in result.steps:
        if step.step_type == "assistant":
            text = (step.content or "").replace("\n", " ")
            preview = text if len(text) <= 160 else text[:160] + "..."
            print(f"[{step.step_index}] assistant: {preview or '<无文本（请求工具）>'}")
            continue
        arguments = json.dumps(step.tool_arguments, ensure_ascii=False)
        print(f"[{step.step_index}] tool: {step.tool_name}({arguments})")
        if step.error_message:
            print(f"    错误: {step.error_message}")
        else:
            output = json.dumps(step.tool_result, ensure_ascii=False, default=str)
            print(f"    结果: {output}")


def final_answer(result: AgentRunResult) -> str:
    """取最后一条有文本内容的 assistant 消息。"""
    for step in reversed(result.steps):
        if step.step_type == "assistant" and step.content:
            return step.content
    return ""


def run_mode(controller: AgentController, task: TaskInfo, mode: str) -> AgentRunResult:
    """执行单个模式并打印详细结果。"""
    print("=" * 72)
    print(f"模式: {mode}")
    print("=" * 72)
    result = controller.run(task, mode=mode, max_steps=4)
    print_steps(result)
    print()
    print(
        f"selected_tools({len(result.selected_tools)}): {result.selected_tools}"
    )
    print(f"tool_call_count: {result.tool_call_count}")
    print(f"termination_reason: {result.termination_reason}")
    print(f"task_success: {result.task_success}")
    print(f"耗时: {result.execution_time:.3f}s")
    print(
        "Token: input={} / output={} / total={}".format(
            result.input_tokens, result.output_tokens, result.total_tokens
        )
    )
    print()
    print("---- 最终回答 ----")
    print(final_answer(result) or "<无最终文本回答>")
    print()
    return result


def main() -> int:
    settings = get_settings()
    model = settings.model.model or "qwen4-max"
    base_url = settings.model.base_url

    print("==== Qwen4-Max 三模式 Function Calling 联调 ====")
    print(f"模型: {model}")
    print(f"Base URL: {base_url or '<默认 OpenAI 地址>'}")

    if not settings.model.api_key:
        print(
            "错误：缺少 LLM_API_KEY，请在项目根目录 .env 中配置后重试。",
            file=sys.stderr,
        )
        return 2
    print("API Key: 已配置（内容不打印）")

    registry = build_registry()
    total_tools = len(registry)
    retriever = build_retriever(registry)
    selector = AdaptiveSelector(
        SelectionConfig(top_k=FIXED_TOP_K, min_k=FIXED_TOP_K, max_k=total_tools)
    )
    llm_client = LLMClient(
        model=model,
        api_key=settings.model.api_key,
        base_url=base_url,
        temperature=settings.model.temperature,
    )
    controller = AgentController(
        llm_client=llm_client,
        task_analyzer=TaskAnalyzer(llm_client),
        tool_registry=registry,
        retriever=retriever,
        selector=selector,
        default_max_steps=4,
    )

    task = build_task()
    print(f"临时工具数: {total_tools}（全部只读）")
    print(f"Fixed Top-K: {FIXED_TOP_K}；Adaptive: complexity={task.complexity}")
    print()
    print(f"任务: {task.description}")
    print()

    modes = ("baseline", "fixed_topk", "adaptive_topk")
    expected_counts = {
        "baseline": total_tools,
        "fixed_topk": FIXED_TOP_K,
        "adaptive_topk": FIXED_TOP_K,  # simple -> 3，并在 [min_k, max_k] 内
    }
    results: Dict[str, AgentRunResult] = {}
    failures: List[str] = []
    exit_code = 0

    for mode in modes:
        result = run_mode(controller, task, mode)
        results[mode] = result

        if result.error_message:
            failures.append(f"{mode}: 调用失败 - {result.error_message}")
            exit_code = exit_code or 3
            continue
        if len(result.selected_tools) != expected_counts[mode]:
            failures.append(
                f"{mode}: 工具数量 {len(result.selected_tools)} != 预期 {expected_counts[mode]}"
            )
            exit_code = exit_code or 6
        if result.tool_call_count == 0:
            failures.append(f"{mode}: 模型未发起工具调用")
            exit_code = exit_code or 5
        if result.termination_reason != "completed":
            failures.append(
                f"{mode}: termination_reason={result.termination_reason!r}"
            )
            exit_code = exit_code or 4

    print("=" * 72)
    print("模式汇总")
    print("=" * 72)
    header = f"{'mode':<14}{'selected':<10}{'tool_calls':<11}{'termination':<13}{'tokens(in/out/total)'}"
    print(header)
    for mode in modes:
        result = results[mode]
        print(
            f"{mode:<14}{len(result.selected_tools):<10}{result.tool_call_count:<11}"
            f"{str(result.termination_reason):<13}"
            f"{result.input_tokens}/{result.output_tokens}/{result.total_tokens}"
        )

    print()
    if failures:
        print("联调未通过：", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return exit_code

    print("联调通过：三种模式均完成工具筛选、真实 ToolCall、结果回填、二次调用与正常终止。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
