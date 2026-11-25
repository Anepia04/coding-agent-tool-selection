"""工具函数实现。

本模块存放每个工具对应的实际 Python 函数，函数名与
``tools.definitions.ToolDefinition.function_name`` 一一对应，参数名与
``ToolDefinition.parameters`` 中的属性名保持一致（ToolRegistry 通过 kwargs 调用）。

当前阶段只定义函数签名与文档：
- 不实现任何文件读写、代码搜索、Git 操作、测试执行的业务逻辑；
- 所有函数统一抛出 ``NotImplementedError``。

TODO(成员 B)：按照 experiments 与 benchmark 的需求逐个实现以下工具，
并保证所有文件操作与命令执行都被限制在 ``config.AgentConfig.workspace_dir`` 内。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """读取文件内容。

    Args:
        path: 相对实验工作目录的文件路径。
        start_line: 起始行号（从 1 开始），为空表示从头读取。
        end_line: 结束行号，为空表示读到文件末尾。

    Returns:
        文件（或指定行区间）的文本内容。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：校验路径位于 workspace 内，读取文件并按行区间截取。
    raise NotImplementedError("待成员 B 实现：read_file 文件读取")


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """写入文件内容。

    Args:
        path: 相对实验工作目录的文件路径。
        content: 要写入的完整文本内容。
        overwrite: 文件已存在时是否允许覆盖。

    Returns:
        写入结果说明（例如写入的字节数或路径）。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：校验路径位于 workspace 内，处理覆盖策略并写入。
    raise NotImplementedError("待成员 B 实现：write_file 文件写入")


def list_files(directory: str = ".", pattern: str = "*") -> List[str]:
    """列出目录下的文件。

    Args:
        directory: 相对实验工作目录的目录路径。
        pattern: 文件名 glob 模式，默认匹配全部文件。

    Returns:
        匹配到的文件路径列表（相对实验工作目录）。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：校验目录位于 workspace 内，按 glob 返回文件列表。
    raise NotImplementedError("待成员 B 实现：list_files 目录列举")


def search_code(
    keyword: str,
    directory: str = ".",
    file_pattern: str = "*.py",
) -> List[Dict[str, Any]]:
    """在源码中搜索关键词。

    Args:
        keyword: 要搜索的关键词、函数名或变量名。
        directory: 搜索起始目录（相对实验工作目录）。
        file_pattern: 限定搜索的文件类型。

    Returns:
        命中列表，每个元素包含文件路径、行号与命中行内容。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：在 workspace 内做文本搜索，返回结构化命中结果。
    raise NotImplementedError("待成员 B 实现：search_code 代码搜索")


def run_tests(test_path: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """运行测试。

    Args:
        test_path: 测试文件或目录（相对实验工作目录），为空表示运行全部测试。
        timeout: 超时时间（秒）。

    Returns:
        结构化测试结果，至少包含 passed / failed / output 字段。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：在 workspace 内以子进程运行测试，解析并返回结果。
    raise NotImplementedError("待成员 B 实现：run_tests 测试执行")


def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """执行受限 shell 命令。

    Args:
        command: 要执行的命令。
        cwd: 命令工作目录（相对实验工作目录），为空表示 workspace 根目录。
        timeout: 超时时间（秒）。

    Returns:
        结构化执行结果，至少包含 returncode / stdout / stderr 字段。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：命令白名单 + 工作目录限制 + 超时控制后执行。
    raise NotImplementedError("待成员 B 实现：run_command 命令执行")


def git_diff(path: Optional[str] = None) -> str:
    """查看代码差异。

    Args:
        path: 限定查看差异的文件路径，为空表示查看整个仓库。

    Returns:
        统一 diff 格式的文本。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：在 workspace 内调用 git diff，并处理非 Git 目录的情况。
    raise NotImplementedError("待成员 B 实现：git_diff 差异查看")


def apply_patch(patch: str) -> Dict[str, Any]:
    """应用代码补丁。

    Args:
        patch: 统一 diff 格式的补丁内容。

    Returns:
        结构化结果，至少包含 applied 与 affected_files 字段。

    Raises:
        NotImplementedError: 当前阶段未实现。
    """
    # TODO(成员 B)：校验补丁作用路径位于 workspace 内后应用补丁。
    raise NotImplementedError("待成员 B 实现：apply_patch 补丁应用")


__all__ = [
    "read_file",
    "write_file",
    "list_files",
    "search_code",
    "run_tests",
    "run_command",
    "git_diff",
    "apply_patch",
]
