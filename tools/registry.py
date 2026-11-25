"""工具注册表。

ToolRegistry 是工具库对外的唯一入口：
- 注册 / 查询工具（元数据 + 实现函数）；
- 把工具定义转换为大模型 API 需要的 function calling schema；
- 根据名称与参数执行具体工具函数。

Baseline 模式通过 ``get_schemas()`` 获取全部工具；Fixed / Adaptive Top-K 模式
通过 ``get_schemas(names=selected_names)`` 只获取检索后筛选的工具。

负责成员：成员 B（工具库）；接口由成员 A 定义
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .definitions import EXAMPLE_TOOL_DEFINITIONS, ToolDefinition

logger = logging.getLogger(__name__)

#: 工具实现函数类型：接收关键字参数，返回任意结果
ToolImplementation = Callable[..., Any]


class ToolRegistry:
    """工具注册与执行中心。

    内部维护两张表：
    - ``_definitions``: 工具名 -> ToolDefinition（元数据）
    - ``_implementations``: 工具名 -> 可调用对象（实现）

    两张表始终同步注册，保证 ``get_schemas`` 中出现的工具都可以被执行。
    """

    def __init__(self) -> None:
        self._definitions: Dict[str, ToolDefinition] = {}
        self._implementations: Dict[str, ToolImplementation] = {}

    # ------------------------------------------------------------------
    # 注册与查询
    # ------------------------------------------------------------------

    def register_tool(
        self,
        definition: ToolDefinition,
        implementation: ToolImplementation,
    ) -> None:
        """注册一个工具。

        Args:
            definition: 工具元数据。
            implementation: 工具实现函数，参数名需与 definition.parameters 对应。

        Raises:
            TypeError: definition / implementation 类型不合法。
            ValueError: 工具名重复注册。
        """
        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition 必须是 ToolDefinition 实例")
        if not callable(implementation):
            raise TypeError("implementation 必须是可调用对象")
        if definition.name in self._definitions:
            raise ValueError(f"工具 {definition.name!r} 已注册，不允许重复注册")
        self._definitions[definition.name] = definition
        self._implementations[definition.name] = implementation
        logger.debug("已注册工具: %s (%s)", definition.name, definition.category)

    def get_tool(self, name: str) -> Tuple[ToolDefinition, ToolImplementation]:
        """根据名称获取工具元数据与实现函数。

        Args:
            name: 工具名称。

        Returns:
            ``(definition, implementation)`` 二元组。

        Raises:
            KeyError: 工具未注册。
        """
        if name not in self._definitions:
            raise KeyError(f"未注册的工具: {name!r}")
        return self._definitions[name], self._implementations[name]

    def get_all_tools(self) -> List[ToolDefinition]:
        """获取全部已注册工具的元数据（注册顺序）。"""
        return list(self._definitions.values())

    def get_tools_by_names(self, names: Iterable[str]) -> List[ToolDefinition]:
        """根据名称列表获取工具元数据，顺序与入参一致。

        Args:
            names: 工具名称序列，通常来自 retrieval 模块的选择结果。

        Raises:
            KeyError: 存在未注册的工具名（错误信息会列出所有缺失名称）。
        """
        ordered_names = list(names)
        missing = [name for name in ordered_names if name not in self._definitions]
        if missing:
            raise KeyError(f"以下工具未注册: {', '.join(missing)}")
        return [self._definitions[name] for name in ordered_names]

    def get_schemas(self, names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """生成大模型 function calling 所需的工具描述列表。

        Args:
            names: 需要包含的工具名；为 ``None`` 时返回全部工具（Baseline 模式）。

        Returns:
            OpenAI 风格的工具 schema 列表，元素形如::

                {
                    "type": "function",
                    "function": {
                        "name": "...",
                        "description": "...",
                        "parameters": {...},
                    },
                }
        """
        definitions = self.get_all_tools() if names is None else self.get_tools_by_names(names)
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                },
            }
            for definition in definitions
        ]

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def execute_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """执行指定工具。

        Args:
            name: 工具名称。
            arguments: 工具参数（关键字参数字典）。

        Returns:
            工具函数的返回值。

        Raises:
            KeyError: 工具未注册。
        """
        definition, implementation = self.get_tool(name)
        kwargs = dict(arguments or {})
        # TODO(成员 B)：执行前按 definition.parameters 校验参数，
        # 并强制所有文件操作 / 命令执行限制在 config.AgentConfig.workspace_dir 内。
        logger.info("执行工具 %s，参数=%s", name, kwargs)
        return implementation(**kwargs)

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> List[str]:
        """全部已注册工具名称。"""
        return list(self._definitions.keys())

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, name: object) -> bool:
        return name in self._definitions


def build_default_registry() -> ToolRegistry:
    """构建注册了示例工具的注册表。

    仅用于框架自检与演示流程打通；示例工具的实现仍为
    ``NotImplementedError``，真实工具由成员 B 注册。

    Returns:
        已注册 ``EXAMPLE_TOOL_DEFINITIONS`` 的 ToolRegistry。
    """
    from . import implementations

    registry = ToolRegistry()
    for definition in EXAMPLE_TOOL_DEFINITIONS:
        implementation = getattr(implementations, definition.function_name)
        registry.register_tool(definition, implementation)
    return registry


__all__ = ["ToolRegistry", "ToolImplementation", "build_default_registry"]
