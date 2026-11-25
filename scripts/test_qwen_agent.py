"""Qwen4-Max 真实 Function Calling / 多轮 Agent 联调脚本（手动执行）。

验证链路：
    AgentController.run -> LLMClient.chat -> 模型发起 get_project_status ToolCall
    -> ToolRegistry.execute_tool -> 工具结果回填 messages -> 模型再次调用并给出最终回答

用法：
    python scripts/test_qwen_agent.py

说明：
- 工具仅在脚本内部注册，只读统计项目文件，不写文件、不执行 Shell 命令；
- 不打印 API Key；
- 本脚本不属于 unittest 流程。

退出码：0 成功；2 缺少 API Key；3 调用失败；4 未正常结束；5 未发生工具调用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import AgentController, AgentRunResult  # noqa: E402
from agent.llm_client import LLMClient  # noqa: E402
from agent.task_analyzer import TaskAnalyzer, TaskInfo  # noqa: E402
from config import get_settings  # noqa: E402
from tools.definitions import ToolDefinition  # noqa: E402
from tools.registry import ToolRegistry  # noqa: E402

TOOL_NAME = "get_project_status"
_EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache"}


def get_project_status() -> dict:
    """只读统计项目基本信息（不写入、不执行命令）。"""
    py_files = [
        path
        for path in PROJECT_ROOT.rglob("*.py")
        if not (_EXCLUDED_PARTS & set(path.parts))
    ]
    test_files = [path for path in py_files if path.name.startswith("test_")]
    return {
        "project": PROJECT_ROOT.name,
        "python_files": len(py_files),
        "test_files": len(test_files),
        "benchmark_ready": (PROJECT_ROOT / "benchmark" / "tasks.json").exists(),
        "status": "ready",
    }


def build_registry() -> ToolRegistry:
    """创建只注册了 get_project_status 的临时工具库。"""
    registry = ToolRegistry()
    registry.register_tool(
        ToolDefinition(
            name=TOOL_NAME,
            description=(
                "获取当前项目的只读状态概览：Python 文件数量、测试文件数量、"
                "benchmark 是否就绪。无需参数，不会修改任何文件。"
            ),
            category="other",
            parameters={"type": "object", "properties": {}},
            function_name="get_project_status",
        ),
        get_project_status,
    )
    return registry


def build_task() -> TaskInfo:
    """构造一个必须调用工具才能回答的任务。"""
    return TaskInfo(
        task_id="smoke_qwen_fc",
        description=(
            "请先调用 get_project_status 工具获取项目状态，"
            "然后严格根据工具返回的结果，用中文简要总结项目当前的 Python 文件数和测试文件数。"
            "必须先调用工具，不要凭空编造数据。"
        ),
        task_type="doc",
        language="python",
        intents=["调用工具获取项目状态", "总结统计结果"],
        complexity="simple",
    )


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
    """取最后一条有文本内容的 assistant 消息作为最终回答。"""
    for step in reversed(result.steps):
        if step.step_type == "assistant" and step.content:
            return step.content
    return ""


def main() -> int:
    settings = get_settings()
    model = settings.model.model or "qwen4-max"
    base_url = settings.model.base_url

    print("==== Qwen4-Max 多轮 Function Calling 联调 ====")
    print(f"模型: {model}")
    print(f"Base URL: {base_url or '<默认 OpenAI 地址>'}")

    if not settings.model.api_key:
        print(
            "错误：缺少 LLM_API_KEY，请在项目根目录 .env 中配置后重试。",
            file=sys.stderr,
        )
        return 2
    print("API Key: 已配置（内容不打印）")
    print()

    llm_client = LLMClient(
        model=model,
        api_key=settings.model.api_key,
        base_url=base_url,
        temperature=settings.model.temperature,
    )
    controller = AgentController(
        llm_client=llm_client,
        task_analyzer=TaskAnalyzer(llm_client),
        tool_registry=build_registry(),
        retriever=None,
        selector=None,
        default_max_steps=4,
    )

    task = build_task()
    print(f"任务: {task.description}")
    print()

    result = controller.run(task, mode="baseline", max_steps=4)

    print_steps(result)
    print()
    print("---- 运行摘要 ----")
    print(f"模式: {result.mode}")
    print(f"选定工具: {result.selected_tools}")
    print(f"工具调用次数: {result.tool_call_count}")
    print(f"终止原因: {result.termination_reason}")
    print(f"task_success: {result.task_success}")
    print(f"耗时: {result.execution_time:.3f}s")
    print("---- Token usage ----")
    print(f"input_tokens : {result.input_tokens}")
    print(f"output_tokens: {result.output_tokens}")
    print(f"total_tokens : {result.total_tokens}")
    print()
    print("---- 最终回答 ----")
    print(final_answer(result) or "<无最终文本回答>")
    print()

    if result.error_message:
        print(f"失败：{result.error_message}", file=sys.stderr)
        return 3
    if result.tool_call_count == 0:
        print("未通过：模型未发起任何工具调用。", file=sys.stderr)
        return 5
    if result.termination_reason != "completed":
        print(
            f"未通过：termination_reason={result.termination_reason!r}，任务未正常结束。",
            file=sys.stderr,
        )
        return 4
    print("联调通过：工具调用、结果回填与多轮回答均已完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
