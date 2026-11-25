"""Agent 控制中心。

AgentController 负责编排一次完整任务，是各模块的组装点：

    任务 -> TaskAnalyzer -> TaskInfo
         -> 工具选择（baseline: 全部工具 / fixed_topk: TFIDF + 固定 K /
                       adaptive_topk: TFIDF + AdaptiveSelector）
         -> ToolRegistry.get_schemas(selected_tools)
         -> LLMClient.chat(messages, tools)
         -> 工具调用循环（ToolRegistry.execute_tool，最多 max_steps 轮）
         -> AgentRunResult

注意：本模块不实现 TF-IDF、Cosine Similarity、Top-K 排序或自适应选择算法，
这些能力全部通过 retrieval 模块的接口调用。

负责成员：成员 A
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Union

from config import DEFAULT_EXPERIMENT_MODE, EXPERIMENT_MODES

if TYPE_CHECKING:
    from .llm_client import LLMClient
    from .task_analyzer import TaskAnalyzer, TaskInfo
    from retrieval.adaptive_selector import AdaptiveSelector
    from retrieval.tfidf_retriever import TFIDFRetriever
    from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        """执行一个完整任务。

        Args:
            task: 任务输入。可以是 TaskInfo、benchmark 任务字典，
                或纯自然语言任务描述字符串。
            mode: 实验模式，取值见 ``config.EXPERIMENT_MODES``。
            max_steps: 最大执行轮数，为空时使用 ``default_max_steps``。

        Returns:
            AgentRunResult: 包含执行轨迹、Token 统计与成功标记的运行结果。

        Raises:
            ValueError: 实验模式不合法。
            NotImplementedError: 当前阶段未实现多轮执行逻辑。
        """
        if mode not in EXPERIMENT_MODES:
            raise ValueError(f"不支持的实验模式: {mode!r}，可选值: {EXPERIMENT_MODES}")
        steps_limit = max_steps if max_steps is not None else self.default_max_steps
        logger.debug("准备执行任务 mode=%s max_steps=%s", mode, steps_limit)

        # TODO(成员 A)：实现完整执行流程
        # 1) TaskAnalyzer.analyze -> TaskInfo
        # 2) _select_tools(task_info, mode) -> 工具名列表
        # 3) tool_registry.get_schemas(names) -> 工具 schema
        # 4) 循环：llm_client.chat -> 解析 tool_calls -> execute_tool -> 回填结果
        #    直到模型不再请求工具或达到 max_steps
        # 5) 汇总 AgentRunResult（Token、耗时、工具调用次数、成功标记）
        raise NotImplementedError("待成员 A 实现：AgentController.run 多轮执行流程")

    def _select_tools(self, task_info: "TaskInfo", mode: str) -> List[str]:
        """按实验模式确定本轮提供给模型的工具名称列表。

        Args:
            task_info: 结构化任务信息。
            mode: 实验模式。

        Returns:
            选中的工具名称列表，顺序即提供给模型的顺序。

        Raises:
            NotImplementedError: 当前阶段未实现。
        """
        # TODO(成员 A)：
        # baseline      -> self.tool_registry.tool_names（全部工具）
        # fixed_topk    -> self.retriever.retrieve(task_info)
        #                  -> self.selector.select(..., strategy="fixed")
        # adaptive_topk -> self.retriever.retrieve(task_info)
        #                  -> self.selector.select(..., complexity=task_info.complexity,
        #                                          strategy="adaptive")
        raise NotImplementedError("待成员 A 实现：AgentController._select_tools 工具选择分发")


__all__ = ["AgentController", "AgentRunResult", "AgentStep"]
