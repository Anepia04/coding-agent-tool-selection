"""工具元数据定义。

本模块只负责描述「工具是什么」：名称、自然语言描述、分类、参数 JSON Schema
以及实际实现的函数名。工具检索（TF-IDF）与工具选择（Top-K）都依赖这里的
description / category 字段，因此描述文本的规范性会直接影响检索质量。

负责成员：成员 B（描述文本与工具清单），成员 A（数据结构约束）

统一数据结构：
    ToolDefinition = {
        "name": str,           # 全局唯一工具名，同时是检索与调用标识
        "description": str,    # 自然语言描述，供检索与 LLM 理解
        "category": str,       # 工具分类，取值见 config.TOOL_CATEGORIES
        "parameters": dict,    # JSON Schema（type=object）
        "function_name": str,  # tools.implementations 中的函数名
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config import TOOL_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """工具的统一元数据描述。

    Attributes:
        name: 工具唯一名称，同时作为检索与执行时的标识。
        description: 自然语言描述，检索模块与 LLM 都依赖该文本。
        category: 工具分类，建议取值见 ``config.TOOL_CATEGORIES``。
        parameters: 参数 JSON Schema，形如
            ``{"type": "object", "properties": {...}, "required": [...]}``。
        function_name: 对应 ``tools.implementations`` 中的函数名，
            为空时默认与 ``name`` 相同。
    """

    name: str
    description: str
    category: str = "other"
    parameters: Dict[str, Any] = field(default_factory=dict)
    function_name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDefinition.name 不能为空")
        if not self.function_name:
            self.function_name = self.name
        if not self.parameters:
            self.parameters = {"type": "object", "properties": {}}
        if not isinstance(self.parameters, dict) or "type" not in self.parameters:
            raise ValueError(
                f"工具 {self.name!r} 的 parameters 必须是 JSON Schema 对象（至少包含 type 字段）"
            )
        if not self.category:
            self.category = "other"
        elif self.category not in TOOL_CATEGORIES:
            # 分类由成员 B 维护，允许先行扩展，仅记录告警
            logger.warning(
                "工具 %s 使用了未在 config.TOOL_CATEGORIES 中登记的分类: %s",
                self.name,
                self.category,
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "function_name": self.function_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDefinition":
        """从字典构造工具定义。"""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            category=data.get("category", "other"),
            parameters=data.get("parameters") or {},
            function_name=data.get("function_name") or data["name"],
        )


# ---------------------------------------------------------------------------
# 占位示例工具定义
#
# 说明：仅用于打通框架与自检测试，不是最终工具清单。
#       TODO(成员 B)：补充完整工具库（文件读写、代码搜索、Git、测试执行等），
#       并打磨 description 文本以适配 TF-IDF 检索。
# ---------------------------------------------------------------------------

EXAMPLE_TOOL_DEFINITIONS: List[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description="读取实验工作目录内某个文本文件的内容，可指定起始行与结束行。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对实验工作目录的文件路径"},
                "start_line": {"type": "integer", "description": "起始行号（从 1 开始，可选）"},
                "end_line": {"type": "integer", "description": "结束行号（可选）"},
            },
            "required": ["path"],
        },
        function_name="read_file",
    ),
    ToolDefinition(
        name="write_file",
        description="在实验工作目录内创建或覆盖写入文本文件，用于修复代码或新增文件。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对实验工作目录的文件路径"},
                "content": {"type": "string", "description": "要写入的完整文本内容"},
                "overwrite": {"type": "boolean", "description": "是否允许覆盖已有文件"},
            },
            "required": ["path", "content"],
        },
        function_name="write_file",
    ),
    ToolDefinition(
        name="list_files",
        description="列出实验工作目录内某一目录下的文件，可按 glob 模式过滤。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "相对实验工作目录的目录路径"},
                "pattern": {"type": "string", "description": "文件名 glob 模式，默认 *"},
            },
            "required": [],
        },
        function_name="list_files",
    ),
    ToolDefinition(
        name="search_code",
        description="在实验工作目录的源码中按关键词或符号名搜索代码位置。",
        category="code_search",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "要搜索的关键词、函数名或变量名"},
                "directory": {"type": "string", "description": "搜索起始目录"},
                "file_pattern": {"type": "string", "description": "限定文件类型，如 *.py"},
            },
            "required": ["keyword"],
        },
        function_name="search_code",
    ),
    ToolDefinition(
        name="run_tests",
        description="在实验工作目录内运行指定测试文件或测试套件，并返回执行结果。",
        category="test",
        parameters={
            "type": "object",
            "properties": {
                "test_path": {"type": "string", "description": "测试文件或目录路径，可选"},
                "timeout": {"type": "integer", "description": "超时时间（秒）"},
            },
            "required": [],
        },
        function_name="run_tests",
    ),
    ToolDefinition(
        name="run_command",
        description="在实验工作目录内执行受限的 shell 命令，用于构建、检查等操作。",
        category="shell",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "cwd": {"type": "string", "description": "相对实验工作目录的工作目录"},
                "timeout": {"type": "integer", "description": "超时时间（秒）"},
            },
            "required": ["command"],
        },
        function_name="run_command",
    ),
    ToolDefinition(
        name="git_diff",
        description="查看实验工作目录对应仓库当前的代码差异。",
        category="git",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "限定查看差异的文件路径，可选"},
            },
            "required": [],
        },
        function_name="git_diff",
    ),
    ToolDefinition(
        name="apply_patch",
        description="将一个补丁应用实验工作目录内的源码文件，用于批量修改代码。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "统一 diff 格式的补丁内容"},
            },
            "required": ["patch"],
        },
        function_name="apply_patch",
    ),
]


__all__ = ["ToolDefinition", "EXAMPLE_TOOL_DEFINITIONS"]
