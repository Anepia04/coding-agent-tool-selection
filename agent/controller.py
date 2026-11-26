"""Agent 控制中心。

AgentController 负责编排一次完整任务，是各模块的组装点：

    任务 -> TaskAnalyzer -> TaskInfo
         -> 工具选择（baseline: 全部工具 / fixed_topk: TFIDF + 固定 K /
                       adaptive_topk: TFIDF + AdaptiveSelector）
         -> ToolRegistry.get_schemas(selected_tools)
         -> LLMClient.chat(messages, tools)
         -> 工具调用（ToolRegistry.execute_tool），整理为 AgentStep
         -> AgentRunResult

当前为**第一阶段**：只执行单轮工具调用（模型一次回复中请求的工具全部执行），
不实现「工具结果回填 → 再次请求模型」的多轮 Agent Loop。

注意：本模块不实现 TF-IDF、Cosine Similarity、Top-K 排序或自适应选择算法，
这些能力全部通过 retrieval 模块的接口调用。

负责成员：成员 A
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

from config import DEFAULT_EXPERIMENT_MODE, EXPERIMENT_MODES

from .task_analyzer import TaskInfo

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .task_analyzer import TaskAnalyzer
    from retrieval.adaptive_selector import AdaptiveSelector
    from retrieval.tfidf_retriever import TFIDFRetriever
    from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: 提供给模型的系统提示词（第一阶段：单轮工具调用）
SYSTEM_PROMPT = (
    "你是一个编程助手 Coding Agent。你可以调用提供的工具读取、修改和验证代码。"
    "请根据任务需要选择合适的工具；如果不需要工具，直接给出回答。"
)


@dataclass
class AgentStep:
    """Agent 执行过程中的单步记录。

    Attributes:
        step_index: 步骤序号，从 0 开始。
        step_type: 步骤类型，``"assistant"`` 表示模型回复，``"tool"`` 表示工具执行。
        content: 模型文本内容（step_type="assistant" 时有效）。
        tool_name: 工具名称（step_type="tool" 时有效）。
        tool_arguments: 工具参数。
        tool_result: 工具执行结果。
        error_message: 该步骤的错误信息。
    """

    step_index: int
    step_type: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result": self.tool_result,
            "error_message": self.error_message,
        }


@dataclass
class AgentRunResult:
    """一次 Agent 任务执行的完整结果（运行期记录）。

    与 ``experiments.schemas.ExperimentResult`` 的关系：本结构保留逐步执行轨迹
    （steps），由 experiments 模块抽取为 ExperimentResult 写入结果文件。

    Attributes:
        task_id: 任务 ID。
        mode: 本次运行使用的实验模式。
        task_success: 任务是否成功。
        selected_tools: 本次提供给模型的工具名称列表。
        steps: 逐步执行轨迹。
        input_tokens / output_tokens / total_tokens: Token 统计。
        execution_time: 执行耗时（秒）。
        tool_call_count: 工具调用次数。
        error_message: 运行失败时的错误信息。
        termination_reason: 终止原因，取值 ``completed``（模型给出最终回答）、
            ``max_steps``（达到最大轮数）、``error``（流程异常）。
    """

    task_id: str
    mode: str
    task_success: bool = False
    selected_tools: List[str] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    execution_time: float = 0.0
    tool_call_count: int = 0
    error_message: Optional[str] = None
    termination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典。"""
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "task_success": self.task_success,
            "selected_tools": list(self.selected_tools),
            "steps": [step.to_dict() for step in self.steps],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "execution_time": self.execution_time,
            "tool_call_count": self.tool_call_count,
            "error_message": self.error_message,
            "termination_reason": self.termination_reason,
        }


class AgentController:
    """Coding Agent 控制中心。"""

    def __init__(
        self,
        llm_client: "LLMClient",
        task_analyzer: "TaskAnalyzer",
        tool_registry: "ToolRegistry",
        retriever: Optional["TFIDFRetriever"] = None,
        selector: Optional["AdaptiveSelector"] = None,
        default_max_steps: int = 8,
    ) -> None:
        """
        Args:
            llm_client: 大模型客户端。
            task_analyzer: 任务分析器。
            tool_registry: 工具注册表。
            retriever: 工具检索器，fixed_topk / adaptive_topk 模式必需。
            selector: 工具选择器，adaptive_topk 模式必需。
            default_max_steps: 未在 run() 中指定时使用的最大执行轮数。
        """
        self.llm_client = llm_client
        self.task_analyzer = task_analyzer
        self.tool_registry = tool_registry
        self.retriever = retriever
        self.selector = selector
        self.default_max_steps = default_max_steps

    def run(
        self,
        task: Union["TaskInfo", Mapping[str, Any], str],
        mode: str = DEFAULT_EXPERIMENT_MODE,
        max_steps: Optional[int] = None,
    ) -> AgentRunResult:
        """执行一个任务（多轮 Agent Loop）。

        Args:
            task: 任务输入。可以是 TaskInfo（跳过分析）、benchmark 任务字典，
                或纯自然语言任务描述字符串。
            mode: 实验模式，取值见 ``config.EXPERIMENT_MODES``。
            max_steps: 最大执行轮数，为空时使用 ``default_max_steps``。

        Returns:
            AgentRunResult: 包含选中工具、完整执行轨迹、Token 统计与终止原因的运行结果。
            ``task_success`` 表示「流程执行未出错」，最终任务正确性由
            ``experiments/evaluate.py`` 判定。

        Raises:
            ValueError: 实验模式不合法、max_steps 非正数、任务描述为空，
                或当前模式缺少 retriever / selector。
            TypeError: task 类型不受支持。
        """
        if mode not in EXPERIMENT_MODES:
            raise ValueError(f"不支持的实验模式: {mode!r}，可选值: {EXPERIMENT_MODES}")
        self._validate_dependencies(mode)

        steps_limit = max_steps if max_steps is not None else self.default_max_steps
        if steps_limit <= 0:
            raise ValueError(f"max_steps 必须为正整数，当前为 {steps_limit!r}")

        pre_analyzed: Optional[TaskInfo] = None
        description: Optional[str] = None
        explicit_task_id: Optional[str] = None
        if isinstance(task, TaskInfo):
            pre_analyzed = task
        else:
            description, explicit_task_id = self._extract_task_input(task)

        logger.debug("开始执行任务 mode=%s max_steps=%s", mode, steps_limit)

        started = time.perf_counter()
        result = AgentRunResult(task_id="", mode=mode)
        try:
            task_info = pre_analyzed or self.task_analyzer.analyze(
                description or "", task_id=explicit_task_id
            )
            result.task_id = task_info.task_id

            # 工具集合在任务执行期间保持不变，三种模式仅在选定阶段有差异
            selected_tools = self._select_tools(task_info, mode)
            result.selected_tools = list(selected_tools)
            logger.debug("模式 %s 选中工具: %s", mode, result.selected_tools)

            schemas = self.tool_registry.get_schemas(selected_tools)
            messages = self._build_messages(task_info)

            for round_index in range(steps_limit):
                response = self.llm_client.chat(messages, tools=schemas)
                self._record_usage(result, response)

                tool_calls = list(getattr(response, "tool_calls", None) or [])
                result.steps.append(
                    AgentStep(
                        step_index=len(result.steps),
                        step_type="assistant",
                        content=getattr(response, "content", None),
                    )
                )

                if not tool_calls:
                    result.termination_reason = "completed"
                    logger.debug("第 %d 轮模型给出最终回答，任务结束", round_index + 1)
                    break

                # 回填 assistant 的 tool_calls 消息，再逐个执行并追加 tool 消息
                messages.append(self._build_assistant_tool_message(response, tool_calls))
                self._execute_tool_calls(result, tool_calls, messages)
                logger.debug(
                    "第 %d 轮执行 %d 个工具调用", round_index + 1, len(tool_calls)
                )
            else:
                result.termination_reason = "max_steps"
                logger.info("达到最大执行轮数 %d，任务终止", steps_limit)

            result.task_success = result.error_message is None
        except Exception as exc:  # 分析 / 检索 / 模型调用等运行期错误统一记录到结果
            logger.exception("任务执行失败: %s", exc)
            result.task_success = False
            result.error_message = f"{type(exc).__name__}: {exc}"
            result.termination_reason = "error"

        result.execution_time = time.perf_counter() - started
        return result

    def _select_tools(self, task_info: "TaskInfo", mode: str) -> List[str]:
        """按实验模式确定本轮提供给模型的工具名称列表。

        Args:
            task_info: 结构化任务信息。
            mode: 实验模式（调用前已通过依赖校验）。

        Returns:
            选中的工具名称列表，顺序即提供给模型的顺序。
        """
        if mode == "baseline":
            # 不做筛选：全部工具
            return list(self.tool_registry.tool_names)

        if self.retriever is None or self.selector is None:
            raise ValueError(f"mode={mode!r} 需要 retriever 与 selector")
        ranked_tools = self.retriever.retrieve(task_info)

        if mode == "fixed_topk":
            return list(self.selector.select(ranked_tools, strategy="fixed"))

        return list(
            self.selector.select(
                ranked_tools,
                complexity=task_info.complexity,
                strategy="adaptive",
            )
        )

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _validate_dependencies(self, mode: str) -> None:
        """校验当前模式所需的依赖是否齐备。"""
        if mode == "baseline":
            return
        if self.retriever is None:
            raise ValueError(f"mode={mode!r} 需要 retriever（TFIDFRetriever），当前未提供")
        if self.selector is None:
            raise ValueError(f"mode={mode!r} 需要 selector（AdaptiveSelector），当前未提供")

    @staticmethod
    def _extract_task_input(task: Union[Mapping[str, Any], str]) -> Tuple[str, Optional[str]]:
        """从字符串或 benchmark 任务字典中提取描述与任务 ID。"""
        if isinstance(task, str):
            description = task.strip()
            if not description:
                raise ValueError("task 描述不能为空")
            return description, None

        if isinstance(task, Mapping):
            description = str(task.get("description") or "").strip()
            if not description:
                raise ValueError("task 缺少 description 字段")
            task_id = task.get("task_id")
            return description, str(task_id) if task_id else None

        raise TypeError(f"task 必须是 TaskInfo、Mapping 或 str，实际为 {type(task).__name__}")

    @staticmethod
    def _build_messages(task_info: "TaskInfo") -> List[Dict[str, Any]]:
        """构造本轮的对话消息（system + user）。"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_info.description},
        ]

    @staticmethod
    def _record_usage(result: AgentRunResult, response: Any) -> None:
        """累加单轮响应的 Token 用量。"""
        result.input_tokens += int(getattr(response, "input_tokens", 0) or 0)
        result.output_tokens += int(getattr(response, "output_tokens", 0) or 0)
        result.total_tokens += int(getattr(response, "total_tokens", 0) or 0)

    @staticmethod
    def _build_assistant_tool_message(response: Any, tool_calls: List[Any]) -> Dict[str, Any]:
        """把模型回复转换为可回填到 messages 的 assistant tool_calls 消息。"""
        return {
            "role": "assistant",
            "content": getattr(response, "content", None),
            "tool_calls": [
                {
                    "id": str(getattr(tool_call, "id", "") or ""),
                    "type": "function",
                    "function": {
                        "name": str(getattr(tool_call, "name", "")),
                        "arguments": json.dumps(
                            dict(getattr(tool_call, "arguments", None) or {}),
                            ensure_ascii=False,
                        ),
                    },
                }
                for tool_call in tool_calls
            ],
        }

    def _execute_tool_calls(
        self,
        result: AgentRunResult,
        tool_calls: List[Any],
        messages: List[Dict[str, Any]],
    ) -> None:
        """逐个执行本轮工具调用，并把结果以 tool 消息回填到 messages。"""
        for tool_call in tool_calls:
            result.tool_call_count += 1
            name = getattr(tool_call, "name", None)
            call_id = str(getattr(tool_call, "id", "") or "")
            arguments = dict(getattr(tool_call, "arguments", None) or {})
            step = AgentStep(
                step_index=len(result.steps),
                step_type="tool",
                tool_name=name,
                tool_arguments=arguments,
            )
            try:
                step.tool_result = self.tool_registry.execute_tool(name, arguments)
                content = self._serialize_tool_result(step.tool_result)
            except Exception as exc:
                # 执行失败不终止任务：错误信息回填给模型，允许其修正后继续
                step.error_message = f"{type(exc).__name__}: {exc}"
                content = f"工具执行失败: {type(exc).__name__}: {exc}"
                logger.warning("工具 %s 执行失败: %s", name, exc)
            result.steps.append(step)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": str(name or ""),
                    "content": content,
                }
            )

    @staticmethod
    def _serialize_tool_result(value: Any) -> str:
        """把工具返回值序列化为回填给模型的文本。"""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)


__all__ = ["AgentController", "AgentRunResult", "AgentStep"]
