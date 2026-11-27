"""工具元数据定义。

本模块只负责描述「工具是什么」：名称、自然语言描述、分类、参数 JSON Schema
以及实际实现的函数名。

关于两个描述字段的职责划分（重要）
----------------------------------
``description``   给**模型**看：会原样进入 function calling 的 schema，
                  每轮请求都随 payload 发送，用于让模型判断「该不该调用」以及
                  「参数怎么填」。应保持简洁、以功能与约束为主。

``search_hints``  给**检索器**看：只参与 TF-IDF 文档构建，**不会**进入 schema，
                  因此不产生任何模型 token 成本。

search_hints 的编写规则
-----------------------
1. **中文只用 2 字词**。分词器对中文做「相邻双字」切分，因此：
   - 2 字词 → 恰好 1 个 token（干净）
   - 3 字词 → 2 个 token，其中 1 个是跨边界碎片（如「运行测试」→ 运行/行测/测试，
     「行测」不是词）
   - 4 字词 → 3 个 token，其中 2 个是碎片（「修改代码」→ 修改/改代/代码）

   若 hints 也写成 3 字以上，任务侧同一个短语会产生多个碎片并与 hints 的碎片
   重复命中，**一次语义命中被计成 2~3 次**，分数被虚高。只用 2 字词时，
   任务侧的碎片（改代/读代/行测/据加/元测…）在 hints 里不存在，自然无法命中，
   只有真正的概念词（修改/阅读/运行/数据/单元…）参与打分。

2. **英文保留完整单词**（不被 n-gram 切分），用于命中工具名、技术术语。

3. **每个高频词只分配给一个工具**，避免多个工具抢同一个词把 IDF 拉平：
       代码 / 修改 / 重构 → edit_file
       阅读 / 查看 / 定位 → read_file
       测试 / 运行 / 验证 → run_tests
       查找 / 引用 / 调用 → search_code
       函数 / 符号 / 定义 → find_symbol
       目录 / 项目 / 模块 → list_files

4. **按功能相似度排序**（见下方分组注释）。顺序会影响三处：注册表顺序、
   baseline 模式下模型看到的工具顺序、以及 TF-IDF 同分时的稳定排序 tie-break，
   所以把同类工具放在一起。

负责成员：成员 B（描述文本与工具清单），成员 A（数据结构约束）

统一数据结构：
    ToolDefinition = {
        "name": str,             # 全局唯一工具名，同时是检索与执行标识
        "description": str,      # 面向模型的功能描述（进 schema）
        "category": str,         # 工具分类，取值见 config.TOOL_CATEGORIES
        "parameters": dict,      # JSON Schema（type=object）
        "function_name": str,    # tools.implementations 中的函数名
        "search_hints": list,    # 面向检索器的 2 字场景词（不进 schema）
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
        search_hints: 面向检索器的 2 字场景词与英文术语，只参与 TF-IDF
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
# 工具清单（按功能相似度分组排序）
#
# 分组顺序：观察 → 修改 → 验证 → 状态 → 兜底。
# 同组内按「操作粒度」由小到大排列，便于检索同分时的 tie-break 落在正确一侧。
#
# hints 规则见模块 docstring：中文一律 2 字，每个高频词只归属一个工具。
# ---------------------------------------------------------------------------

EXAMPLE_TOOL_DEFINITIONS: List[ToolDefinition] = [
    # ==================== 一、观察类（只读，不改动任何文件） ====================
    # 逐级细化：整文件读取 → 目录遍历 → 单文件属性 → 文本搜索 → 符号解析 → 标记扫描
    ToolDefinition(
        name="read_file",
        description="读取文本文件的内容，可按行区间截取。",
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
            "阅读", "查看", "浏览", "分析", "异常", "逻辑", "数据", "加载", "代码", "读取", "检查",
            "read", "open", "view", "source",
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
            "目录", "结构", "项目", "文件", "列表", "布局", "路径", "列出",
            "ls", "list", "tree", "dir", "glob",
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
            "属性", "大小", "行数", "体积", "规模", "统计", "文件", "时间",
            "stat", "size", "info", "lines",
        ],
    ),
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
            "搜索", "查找", "检索", "引用", "调用", "使用", "位置", "出现",
            "代码", "阅读", "行号", "定位", "寻找",
            "search", "grep", "find", "reference", "usage",
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
            "函数", "符号", "定义", "声明", "接口", "签名", "公共", "参数", "代码",
            "symbol", "definition", "ast", "signature", "class", "function",
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
            "待办", "标记", "遗留", "完成", "废弃",
            "todo", "fixme", "hack", "pending",
        ],
    ),

    # ==================== 二、修改类（写操作） ====================
    # 按侵入范围：改片段 → 写整文件 → 打补丁 → 移路径 → 删除
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
            "修改", "改动", "改成", "替换", "修复", "修正", "更新", "补全",
            "重构", "拆分", "提取", "消除", "统一", "处理", "新增", "实现", 
            "编写", "编辑", "添加", "补充", "清理", "删除",
            "edit", "modify", "replace", "change", "fix", "update", "refactor",
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
            "新增", "创建", "生成", "编写", "新建", "写入", "保存", "补充",
            "修改", "文件",
            "create", "write", "generate", "save", "new",
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
            "补丁", "批量", "多处",
            "patch", "apply", "unified",
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
            "移动", "命名", "改名", "迁移", "归档", "导入",
            "move", "rename", "relocate",
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
            "删除", "清理", "移除", "去掉", "孤立",
            "delete", "remove", "clean", "drop",
        ],
    ),

    # ==================== 三、验证类 ====================
    # 行为验证 → 语法验证
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
            "测试", "运行", "单元", "用例", "验证", "覆盖", "通过", "失败", "回归", "断言",
            "test", "pytest", "unittest", "verify", "run", "assert",
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
            "语法", "检查", "静态", "解析", "缩进", "类型",
            "syntax", "compile", "lint", "check", "parse",
        ],
    ),

    # ==================== 四、状态类（Git） ====================
    # 变更清单 → 具体差异
    ToolDefinition(
        name="git_status",
        description="查看仓库工作区状态，列出被修改、新增、删除的文件。",
        category="git",
        parameters={"type": "object", "properties": {}, "required": []},
        function_name="git_status",
        search_hints=[
            "状态", "变更", "提交", "工作", "清单",
            "status", "staged", "working",
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
            "差异", "对比", "变化", "记录",
            "diff", "compare", "changes",
        ],
    ),

    # ==================== 五、兜底（受限 shell） ====================
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
            "命令", "执行", "构建", "安装", "打包", "脚本",
            "command", "shell", "exec", "build", "install",
        ],
    ),
]


#: 便于外部按需检查工具池规模
TOOL_COUNT: int = len(EXAMPLE_TOOL_DEFINITIONS)


__all__ = ["ToolDefinition", "EXAMPLE_TOOL_DEFINITIONS", "TOOL_COUNT"]
