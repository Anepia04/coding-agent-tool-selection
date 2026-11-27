"""工具元数据定义。

本模块只负责描述「工具是什么」：名称、自然语言描述、分类、参数 JSON Schema
以及实际实现的函数名。

关于两个描述字段的职责划分（重要）
----------------------------------
``description``   给**模型**看：会原样进入 function calling 的 schema，
                  每轮请求都随 payload 发送，用于让模型判断「该不该调用」以及
                  「参数怎么填」。应保持简洁、以功能与约束为主。

``search_hints``  给**检索器**看：只参与 TF-IDF 文档构建，**不会**进入 schema，
                  因此不产生任何模型 token 成本。用于补充任务描述中可能出现
                  的场景词、同义词、动作词，弥补「功能描述与任务措辞不重合」
                  导致的相关性丢失（例如任务只说「修改 calculator.py」，
                  而 read_file 的功能描述里并不会出现「修改」）。

负责成员：成员 B（描述文本与工具清单），成员 A（数据结构约束）

统一数据结构：
    ToolDefinition = {
        "name": str,             # 全局唯一工具名，同时是检索与执行标识
        "description": str,      # 面向模型的功能描述（进 schema）
        "category": str,         # 工具分类，取值见 config.TOOL_CATEGORIES
        "parameters": dict,      # JSON Schema（type=object）
        "function_name": str,    # tools.implementations 中的函数名
        "search_hints": list,    # 面向检索器的场景/同义词（不进 schema）
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
        description: 面向模型的功能描述，会进入 function calling schema。
        category: 工具分类，建议取值见 ``config.TOOL_CATEGORIES``。
        parameters: 参数 JSON Schema，形如
            ``{"type": "object", "properties": {...}, "required": [...]}``。
        function_name: 对应 ``tools.implementations`` 中的函数名，
            为空时默认与 ``name`` 相同。
        search_hints: 面向检索器的场景词、同义词与动作词，只参与 TF-IDF
            文档构建，不会发送给模型。
    """

    name: str
    description: str
    category: str = "other"
    parameters: Dict[str, Any] = field(default_factory=dict)
    function_name: str = ""
    search_hints: List[str] = field(default_factory=list)

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

        if self.search_hints is None:
            self.search_hints = []
        if not isinstance(self.search_hints, list):
            raise ValueError(
                f"工具 {self.name!r} 的 search_hints 必须是字符串列表，"
                f"当前为 {type(self.search_hints).__name__}"
            )
        normalized: List[str] = []
        for hint in self.search_hints:
            text = str(hint).strip()
            if text:
                normalized.append(text)
        self.search_hints = normalized

        if not isinstance(self.description, str) or not self.description.strip():
            # 空描述对该工具没有硬性阻塞（检索仍可用 name 与 hints），但必须告警，
            # 因为它是模型判断是否调用该工具的唯一依据。
            logger.warning("工具 %s 的 description 为空，模型将无法理解其用途", self.name)

    def to_dict(self) -> Dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "function_name": self.function_name,
            "search_hints": list(self.search_hints),
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
            search_hints=list(data.get("search_hints") or []),
        )


# ---------------------------------------------------------------------------
# 工具清单
#
# description：给模型看的功能说明（进 schema，有 token 成本）
# search_hints：给检索器看的场景/同义词（不进 schema，零 token 成本）
#
# 编写约定：
# 1. description 不写「实验工作目录内」这类所有工具共有的模板前缀——它在 16 个
#    文档里同时出现，IDF 趋近于 0，只稀释向量、不提供区分度；
# 2. search_hints 按「动作阶段」分配独占词，避免多个工具抢同一个词：
#       read_file   -> 查看 / 定位 / 阅读
#       edit_file   -> 修改 / 替换 / 改
#       write_file  -> 新建 / 保存 / 覆盖
#       run_tests   -> 验证 / 回归 / 测试
# 3. 同一领域的工具用一种前缀区分：file 类用动作词，git 类用状态词。
# ---------------------------------------------------------------------------

EXAMPLE_TOOL_DEFINITIONS: List[ToolDefinition] = [
    # ---------------- 文件读取与列举 ----------------
    ToolDefinition(
        name="read_file",
        description="读取某个文本文件的内容，可按行区间截取。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作目录的文件路径"},
                "start_line": {"type": "integer", "description": "起始行号（从 1 开始，可选）"},
                "end_line": {"type": "integer", "description": "结束行号（可选）"},
            },
            "required": ["path"],
        },
        function_name="read_file",
        search_hints=[
            "查看", "浏览", "阅读", "打开文件", "看代码", "检查源码",
            "定位实现", "现有实现", "读一下", "内容是什么", "先看看",
        ],
    ),
    ToolDefinition(
        name="list_files",
        description="列出指定目录下的文件，可按 glob 模式过滤。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "相对工作目录的目录路径"},
                "pattern": {"type": "string", "description": "文件名 glob 模式，默认 *"},
            },
            "required": [],
        },
        function_name="list_files",
        search_hints=[
            "目录结构", "项目结构", "有哪些文件", "文件列表", "工程布局",
            "目录树", "看看有什么", "项目有哪些",
        ],
    ),
    ToolDefinition(
        name="file_info",
        description="查看文件或目录的大小、行数与修改时间等属性。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作目录的文件或目录路径"},
            },
            "required": ["path"],
        },
        function_name="file_info",
        search_hints=[
            "文件大小", "多少行", "文件属性", "体积", "修改时间",
            "多大", "有多大", "行数统计", "文件信息",
        ],
    ),

    # ---------------- 文件修改 ----------------
    ToolDefinition(
        name="edit_file",
        description=(
            "把文件中的一段旧文本精确替换为新文本，其余内容保持不变，"
            "适合只改动其中几行；旧文本必须在文件中唯一匹配。"
        ),
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作目录的文件路径"},
                "old_string": {"type": "string", "description": "要被替换的原文片段（需唯一匹配）"},
                "new_string": {"type": "string", "description": "替换后的新文本"},
                "replace_all": {"type": "boolean", "description": "是否替换全部匹配，默认 false"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        function_name="edit_file",
        search_hints=[
            "修改代码", "改动", "改成", "改为", "替换", "改一行", "改几行",
            "局部修改", "修正语句", "把某行", "改写", "调整代码",
        ],
    ),
    ToolDefinition(
        name="write_file",
        description="创建或整体覆盖写入一个文本文件，父目录会自动创建。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作目录的文件路径"},
                "content": {"type": "string", "description": "要写入的完整文本内容"},
                "overwrite": {"type": "boolean", "description": "是否允许覆盖已有文件"},
            },
            "required": ["path", "content"],
        },
        function_name="write_file",
        search_hints=[
            "新建文件", "创建文件", "生成文件", "从零写", "整体覆盖",
            "保存文件", "写一个", "新增文件", "补充新文件",
        ],
    ),
    ToolDefinition(
        name="apply_patch",
        description="把统一 diff 格式的补丁应用到源码文件，用于一次性改动多处代码。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "统一 diff 格式的补丁内容"},
            },
            "required": ["patch"],
        },
        function_name="apply_patch",
        search_hints=[
            "打补丁", "应用补丁", "批量修改", "diff 应用", "补丁文件",
            "多处改动", "统一 diff",
        ],
    ),
    ToolDefinition(
        name="delete_path",
        description="删除不再需要的文件或空目录。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对工作目录的文件或空目录路径"},
            },
            "required": ["path"],
        },
        function_name="delete_path",
        search_hints=[
            "删除", "删掉", "移除", "清理", "废弃文件", "不再需要",
            "去掉文件", "清除",
        ],
    ),
    ToolDefinition(
        name="move_path",
        description="把文件或目录重命名、移动到新的路径。",
        category="file",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "原路径（相对工作目录）"},
                "destination": {"type": "string", "description": "目标路径（相对工作目录）"},
                "overwrite": {"type": "boolean", "description": "目标已存在时是否覆盖"},
            },
            "required": ["source", "destination"],
        },
        function_name="move_path",
        search_hints=[
            "重命名", "改名", "移动文件", "挪到", "调整目录结构",
            "归档", "换个位置", "搬迁",
        ],
    ),

    # ---------------- 代码搜索与符号 ----------------
    ToolDefinition(
        name="search_code",
        description="按关键词在源码中搜索文本出现位置，返回文件、行号与命中行。",
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
        search_hints=[
            "搜索", "查找", "关键字", "哪里用到", "引用位置", "调用点",
            "全局搜索", "出现过", "谁用了",
        ],
    ),
    ToolDefinition(
        name="find_symbol",
        description="解析源码中的函数与类定义，列出符号名、所在文件与行号。",
        category="code_search",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "按符号名过滤，可选"},
                "directory": {"type": "string", "description": "搜索起始目录"},
                "file_pattern": {"type": "string", "description": "限定文件类型，如 *.py"},
            },
            "required": [],
        },
        function_name="find_symbol",
        search_hints=[
            "函数定义", "类定义", "符号", "接口", "在哪定义", "定义位置",
            "有哪些函数", "结构是什么", "方法列表",
        ],
    ),
    ToolDefinition(
        name="list_todos",
        description="扫描源码中的 TODO、FIXME 等未完成标记。",
        category="code_search",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "搜索起始目录"},
                "file_pattern": {"type": "string", "description": "限定文件类型，如 *.py"},
            },
            "required": [],
        },
        function_name="list_todos",
        search_hints=[
            "待办", "TODO", "FIXME", "未完成", "遗留问题", "待处理",
            "还差什么", "没做完",
        ],
    ),

    # ---------------- 测试与语法 ----------------
    ToolDefinition(
        name="run_tests",
        description="运行测试文件或测试套件，返回通过、失败与错误数量。",
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
        search_hints=[
            "跑测试", "运行测试", "单元测试", "验证改动", "测试是否通过",
            "回归", "用例", "测试失败", "检验结果",
        ],
    ),
    ToolDefinition(
        name="check_syntax",
        description="对源码做语法检查，报告无法解析的文件与具体错误位置。",
        category="test",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件或目录路径，默认整目录"},
                "file_pattern": {"type": "string", "description": "限定文件类型，如 *.py"},
            },
            "required": [],
        },
        function_name="check_syntax",
        search_hints=[
            "语法错误", "语法检查", "编译检查", "能否运行", "解析失败",
            "静态检查", "代码检查", "缩进错误",
        ],
    ),

    # ---------------- Git 与命令 ----------------
    ToolDefinition(
        name="git_status",
        description="查看仓库工作区状态，列出被修改、新增、删除的文件。",
        category="git",
        parameters={"type": "object", "properties": {}, "required": []},
        function_name="git_status",
        search_hints=[
            "工作区状态", "改了哪些文件", "变更清单", "未提交", "待提交",
            "有什么改动", "修改记录",
        ],
    ),
    ToolDefinition(
        name="git_diff",
        description="查看工作目录对应的代码差异。",
        category="git",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "限定查看差异的文件路径，可选"},
            },
            "required": [],
        },
        function_name="git_diff",
        search_hints=[
            "差异", "对比改动", "改了什么", "diff", "变化内容",
            "前后对比", "代码变更",
        ],
    ),
    ToolDefinition(
        name="run_command",
        description="执行受限的 shell 命令，用于构建、检查等操作。",
        category="shell",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的命令"},
                "cwd": {"type": "string", "description": "相对工作目录的工作目录"},
                "timeout": {"type": "integer", "description": "超时时间（秒）"},
            },
            "required": ["command"],
        },
        function_name="run_command",
        search_hints=[
            "命令行", "shell", "执行命令", "构建", "打包",
            "脚本", "批处理", "跑一下命令",
        ],
    ),
]


#: 便于外部按需检查工具池规模
TOOL_COUNT: int = len(EXAMPLE_TOOL_DEFINITIONS)


__all__ = ["ToolDefinition", "EXAMPLE_TOOL_DEFINITIONS", "TOOL_COUNT"]
