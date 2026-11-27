"""工具函数实现。

本模块存放每个工具对应的实际 Python 函数，函数名与
``tools.definitions.ToolDefinition.function_name`` 一一对应，参数名与
``ToolDefinition.parameters`` 中的属性名保持一致（ToolRegistry 通过 kwargs 调用）。

安全边界（成员 B 负责）
------------------------
1. **路径围栏**：所有路径参数都必须是相对路径，解析后必须落在
   ``config.AgentConfig.workspace_dir`` 内。绝对路径、盘符路径、UNC 路径、
   含 ``..`` 段的路径、以及指向工作目录之外的符号链接一律拒绝。
2. **命令执行默认关闭**：``run_command`` 受 ``AGENT_ALLOW_SHELL`` 控制，默认为
   ``false``（抛出 ``PermissionError``）。即使开启，也只允许白名单程序，并且
   不经过 shell 解析，因此 ``;`` / ``|`` / ``&&`` / ``>` 等元字符不会被解释。
   注意：这是**防误操作的弱约束，不是安全沙箱**，不应把不可信的模型放进特权环境执行。
3. **输出截断**：所有工具输出都限制在 ``MAX_TOOL_OUTPUT_CHARS`` 字符以内，
   避免 ``run_tests`` / ``git_diff`` 之类的长输出撑爆模型上下文。
4. **敏感环境变量隔离**：子进程不会继承名称中含 API_KEY / TOKEN / SECRET 等的环境变量。

工作目录切换
------------
``configure_workspace()`` 允许实验框架（成员 C）在同一进程内为不同任务工作副本
切换工具根目录；不调用时始终使用 ``config.get_settings().agent.workspace_dir``。

负责成员：成员 B
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 资源限制常量
# ---------------------------------------------------------------------------

#: 单次读取的文件字节上限，超过直接报错（提示改用行区间）
MAX_READ_BYTES: int = 1000000
#: 单次写入的字节上限
MAX_WRITE_BYTES: int = 2000000
#: 任意工具返回内容的字符上限
MAX_TOOL_OUTPUT_CHARS: int = 60000
#: search_code 的命中条数上限
MAX_SEARCH_HITS: int = 200
#: search_code 单行内容的字符上限
MAX_SEARCH_LINE_CHARS: int = 400
#: list_files 返回的文件条数上限
MAX_LISTED_FILES: int = 2000
#: 默认子进程超时（秒）与允许的最大值
DEFAULT_TIMEOUT_SECONDS: int = 60
MAX_TIMEOUT_SECONDS: int = 600

#: 列举 / 搜索时跳过的大目录（降低 Token 消耗与噪音）
IGNORED_DIR_NAMES: frozenset = frozenset(
    {
        ".git", ".hg", ".svn",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
        ".venv", "venv", "env", "node_modules", "dist", "build", ".idea", ".vscode",
    }
)

#: run_command 允许的程序白名单（不含 shell 内建命令，且刻意排除 rm/mv/cp 等破坏性命令）
SHELL_COMMAND_WHITELIST: frozenset = frozenset(
    {
        "python", "python3", "py", "pip", "pip3", "pytest", "coverage", "tox",
        "git", "ls", "dir", "cat", "type", "head", "tail", "wc",
        "find", "findstr", "grep", "rg", "pwd", "echo", "tree", "which", "where",
    }
)

#: git 中会写到仓库外或执行外部程序的参数，直接禁止
_GIT_FORBIDDEN_ARGS: frozenset = frozenset({"--global", "--system", "--exec-path", "--upload-pack"})

#: 子进程中应被剔除的环境变量关键字
_SENSITIVE_ENV_MARKERS: Tuple[str, ...] = (
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "PRIVATE_KEY",
)

#: 统一 diff 的 hunk 头
_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


# ---------------------------------------------------------------------------
# 工作目录与路径安全
# ---------------------------------------------------------------------------

#: 进程级工作目录覆盖（None 表示使用 config 配置）
_WORKSPACE_OVERRIDE: Optional[Path] = None


def configure_workspace(path: Optional[str]) -> None:
    """为当前进程覆盖工具的工作目录根。

    实验框架（成员 C）在同一进程内依次运行多个任务时，需要通过本函数把工具
    指向各自的工作副本；传 ``None`` 恢复使用 ``config`` 中的配置。

    Args:
        path: 新的工作目录（相对路径按项目根目录解析），或 ``None`` 取消覆盖。
    """
    global _WORKSPACE_OVERRIDE
    if path is None:
        _WORKSPACE_OVERRIDE = None
        logger.debug("工具工作目录覆盖已取消，回退到 config 配置")
        return
    _WORKSPACE_OVERRIDE = Path(path).expanduser()
    logger.debug("工具工作目录覆盖为: %s", _WORKSPACE_OVERRIDE)


def _workspace_root(*, must_exist: bool = True) -> Path:
    """返回已解析的工作目录根，并按需校验其存在性。"""
    raw = _WORKSPACE_OVERRIDE if _WORKSPACE_OVERRIDE is not None else get_settings().agent.workspace_dir
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    if must_exist and not root.is_dir():
        raise FileNotFoundError(
            f"工具工作目录不存在: {root}。请检查 AGENT_WORKSPACE_DIR 配置，"
            f"或先创建该目录（实验时应指向任务工作副本）。"
        )
    return root


def _is_within(path: Path, root: Path) -> bool:
    """判断 path 是否位于 root 内（Windows 大小写不敏感）。"""
    path_str = os.path.normcase(str(path))
    root_str = os.path.normcase(str(root))
    return path_str == root_str or path_str.startswith(root_str + os.sep)


def _relative_display(path: Path, root: Path) -> str:
    """把绝对路径转换为相对工作目录的 POSIX 风格路径。"""
    try:
        return Path(os.path.relpath(str(path), str(root))).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_relative(
    raw: Any,
    *,
    field_name: str = "path",
    must_exist: bool = False,
    expect_dir: Optional[bool] = None,
    create_root: bool = False,
) -> Path:
    """把相对路径参数解析为工作目录内的绝对路径，并执行越界检查。

    Args:
        raw: 调用方传入的路径参数（必须是字符串）。
        field_name: 参数名，用于错误信息。
        must_exist: 为 True 时要求目标已存在。
        expect_dir: True 要求是目录，False 要求是文件，None 不限制。
        create_root: 为 True 时允许工作目录根尚不存在（写操作场景）。

    Raises:
        TypeError: 参数不是字符串。
        ValueError: 参数为空、含 NUL、绝对路径、含 ``..`` 段或越界。
        FileNotFoundError: ``must_exist`` 为真但目标不存在。
        NotADirectoryError / IsADirectoryError: 目标类型不符。
    """
    if not isinstance(raw, str):
        raise TypeError(f"{field_name} 必须是字符串，当前为 {type(raw).__name__}")
    text = raw.strip()
    if not text:
        text = "."
    if "\x00" in text:
        raise ValueError(f"{field_name} 含有非法字符 NUL: {raw!r}")

    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise ValueError(
            f"{field_name} 必须是相对工作目录的路径，不接受绝对路径 / 盘符路径: {raw!r}"
        )
    if ".." in candidate.parts:
        raise ValueError(f"{field_name} 不能包含 '..' 段: {raw!r}")

    root = _workspace_root(must_exist=not create_root)
    resolved = (root / candidate).resolve()
    if not _is_within(resolved, root):
        raise ValueError(
            f"{field_name} 越界，解析后为 {resolved}，不在工作目录 {root} 内: {raw!r}"
        )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{field_name} 指向的路径不存在: {raw!r}（解析为 {resolved}）")
    if expect_dir is True and resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"{field_name} 必须是目录: {raw!r}")
    if expect_dir is False and resolved.exists() and not resolved.is_file():
        raise IsADirectoryError(f"{field_name} 必须是文件: {raw!r}")
    return resolved


def _skip_entry(path: Path, root: Path) -> bool:
    """判断是否应跳过某个文件/目录（越界符号链接、忽略目录）。

    只检查**相对工作目录**的路径分量：若工作目录本身所在的父级目录恰叫
    ``build`` / ``dist`` 之类，全路径匹配会把整个工作区误判为应忽略。
    """
    try:
        resolved = path.resolve()
    except OSError:  # 悬空链接等
        return True
    if not _is_within(resolved, root):
        return True
    try:
        relative = Path(os.path.relpath(str(resolved), str(root)))
    except ValueError:
        return True
    return any(part in IGNORED_DIR_NAMES for part in relative.parts)


# ---------------------------------------------------------------------------
# 文本 / 编码辅助
# ---------------------------------------------------------------------------

def _truncate_text(text: str, *, label: str = "内容", limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """按字符上限截断文本，并附加明确的截断提示。"""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        text[:limit]
        + f"\n\n...[{label}已截断：原始 {len(text)} 字符，仅返回前 {limit} 字符，省略 {omitted} 字符]"
    )


def _decode_bytes(data: Optional[bytes]) -> str:
    """把子进程输出字节解码为文本，兼容中文 Windows 的控制台编码。"""
    if not data:
        return ""
    if isinstance(data, str):  # 某些平台在 text 模式下会直接返回 str
        return data
    for encoding in ("utf-8", "gbk", "cp1252"):
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _read_text_file(path: Path, *, label: str = "文件", max_bytes: int = MAX_READ_BYTES) -> str:
    """读取文本文件，拒绝二进制内容与超大文件。"""
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise OSError(f"{label}无法访问: {path}（{exc}）") from exc
    if size > max_bytes:
        raise ValueError(
            f"{label}过大：{size} 字节，超过上限 {max_bytes} 字节。"
            f"请使用 start_line / end_line 分段读取，或改用 search_code 定位。"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OSError(f"{label}读取失败: {path}（{exc}）") from exc
    if b"\x00" in data:
        raise ValueError(f"{label}疑似二进制内容，拒绝按文本读取: {path}")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("%s 不是合法 UTF-8，按替换字符解码: %s", label, path)
        return data.decode("utf-8", errors="replace")


def _normalize_timeout(timeout: Any) -> int:
    """校验并规范化超时参数。"""
    try:
        value = int(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"timeout 必须是正整数，当前为 {timeout!r}") from exc
    if value <= 0:
        raise ValueError(f"timeout 必须为正整数，当前为 {value}")
    if value > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout 不能超过 {MAX_TIMEOUT_SECONDS} 秒，当前为 {value}")
    return value


def _child_env(root: Path) -> Dict[str, str]:
    """构造子进程环境变量：剔除敏感项、固定 UTF-8、补充工作目录到 PYTHONPATH。"""
    env: Dict[str, str] = {}
    for key, value in os.environ.items():
        if any(marker in key.upper() for marker in _SENSITIVE_ENV_MARKERS):
            continue
        env[key] = value
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + existing if existing else "")
    return env


def _split_command(command: str) -> List[str]:
    """把命令行拆成 argv。

    Windows 下必须用 ``posix=False``：否则反斜杠会被当成转义符，
    ``tests\\test_a.py`` 会被拆成 ``teststest_a.py``，是真实会踩的坑。
    """
    if os.name == "nt":
        argv = shlex.split(command, posix=False)
        return [
            token[1:-1] if len(token) >= 2 and token[0] == token[-1] == '"' else token
            for token in argv
        ]
    return shlex.split(command)


def _run_process(
    cmd: List[str],
    *,
    cwd: Path,
    timeout: int,
    label: str,
) -> Dict[str, Any]:
    """执行子进程并捕获输出（不使用 shell）。

    Returns:
        ``{"returncode", "stdout", "stderr", "timed_out"}``。
    """
    logger.info("%s: %s (cwd=%s)", label, cmd, cwd)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
            env=_child_env(cwd),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("%s 超时（%s 秒）: %s", label, timeout, cmd)
        return {
            "returncode": -1,
            "stdout": _decode_bytes(exc.stdout),
            "stderr": _decode_bytes(exc.stderr),
            "timed_out": True,
        }
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{label}无法启动程序 {cmd[0]!r}：未在当前环境找到可执行文件。{exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"{label}执行失败：{exc}") from exc

    return {
        "returncode": completed.returncode,
        "stdout": _decode_bytes(completed.stdout),
        "stderr": _decode_bytes(completed.stderr),
        "timed_out": False,
    }


# ---------------------------------------------------------------------------
# 工具：文件读写
# ---------------------------------------------------------------------------

def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """读取文件内容。

    Args:
        path: 相对实验工作目录的文件路径。
        start_line: 起始行号（从 1 开始），为空表示从头读取。
        end_line: 结束行号（含该行），为空表示读到文件末尾。

    Returns:
        文件（或指定行区间）的文本内容；空文件或空区间返回空字符串。
        超过 ``MAX_TOOL_OUTPUT_CHARS`` 时会被截断并附提示。

    Raises:
        FileNotFoundError: 文件不存在。
        IsADirectoryError: 目标是目录。
        ValueError: 路径越界、文件过大、疑似二进制、行号非法。
    """
    target = _resolve_relative(path, must_exist=True, expect_dir=False)
    content = _read_text_file(target, label=f"文件 {path!r}")

    lines = content.splitlines()
    total = len(lines)

    start = 1 if start_line is None else _coerce_line_no(start_line, "start_line")
    if end_line is None:
        end = total
    else:
        end = _coerce_line_no(end_line, "end_line")
        if end < start:
            raise ValueError(f"end_line({end}) 不能小于 start_line({start})")
    if start > total:
        logger.debug("start_line=%s 超过文件总行数 %s，返回空内容", start, total)
        return ""

    selected = lines[start - 1:end]
    return _truncate_text("\n".join(selected), label="文件内容")


def _coerce_line_no(value: Any, name: str) -> int:
    """校验行号参数。"""
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是正整数，当前为 {value!r}") from exc
    if number < 1:
        raise ValueError(f"{name} 必须从 1 开始，当前为 {number}")
    return number


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """写入文件内容。

    Args:
        path: 相对实验工作目录的文件路径；父目录不存在时会自动创建。
        content: 要写入的完整文本内容（统一以 ``\\n`` 换行落盘）。
        overwrite: 文件已存在时是否允许覆盖；默认不覆盖。

    Returns:
        写入结果说明（路径、字符数、字节数、新建或覆盖）。

    Raises:
        FileExistsError: 文件已存在且 ``overwrite=False``。
        IsADirectoryError: 目标是目录。
        TypeError: ``content`` 不是字符串。
        ValueError: 路径越界或内容超过 ``MAX_WRITE_BYTES``。
    """
    if not isinstance(content, str):
        raise TypeError(f"content 必须是字符串，当前为 {type(content).__name__}")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        raise ValueError(
            f"写入内容过大：{len(encoded)} 字节，超过上限 {MAX_WRITE_BYTES} 字节。"
            f"请拆分为多次写入或使用 apply_patch。"
        )

    target = _resolve_relative(path, expect_dir=False, create_root=True)
    existed = target.exists()
    if existed and target.is_dir():
        raise IsADirectoryError(f"path 指向目录，不能写入文件: {path!r}")
    if existed and not overwrite:
        raise FileExistsError(
            f"文件已存在且 overwrite=False: {path!r}。如需覆盖请显式传入 overwrite=True。"
        )

    text = content.replace("\r\n", "\n")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except OSError as exc:
        raise OSError(f"写入失败: {path!r}（{exc}）") from exc

    logger.info("已写入文件 %s（%s 字节）", path, len(encoded))
    action = "覆盖" if existed else "新建"
    return f"已{action}写入 {path}（{len(text)} 字符，{len(encoded)} 字节）"


# ---------------------------------------------------------------------------
# 工具：目录列举与代码搜索
# ---------------------------------------------------------------------------

def _normalize_glob(pattern: Any, *, name: str) -> str:
    """校验 glob 模式，拒绝绝对路径与 ``..`` 段。"""
    if pattern is None:
        return "*"
    if not isinstance(pattern, str):
        raise TypeError(f"{name} 必须是字符串，当前为 {type(pattern).__name__}")
    text = pattern.strip() or "*"
    if "\x00" in text:
        raise ValueError(f"{name} 含有非法字符 NUL")
    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise ValueError(f"{name} 必须是相对模式，不接受绝对路径: {pattern!r}")
    if ".." in candidate.parts:
        raise ValueError(f"{name} 不能包含 '..' 段: {pattern!r}")
    return text


def list_files(directory: str = ".", pattern: str = "*") -> List[str]:
    """列出目录下的文件。

    Args:
        directory: 相对实验工作目录的目录路径。
        pattern: 文件名 glob 模式，默认 ``*``；可用 ``**/*.py`` 递归匹配。

    Returns:
        匹配到的文件路径列表（相对实验工作目录、POSIX 风格、已排序）。
        跳过 ``.git`` / ``__pycache__`` / ``node_modules`` 等目录。
        超过 ``MAX_LISTED_FILES`` 条时，末尾追加 ``"<truncated>"`` 标记。

    Raises:
        NotADirectoryError: ``directory`` 不是目录。
        ValueError: 路径或模式越界。
    """
    root = _workspace_root()
    base = _resolve_relative(directory, field_name="directory", must_exist=True)
    if not base.is_dir():
        raise NotADirectoryError(f"directory 必须是目录: {directory!r}")
    glob_pattern = _normalize_glob(pattern, name="pattern")

    results: List[str] = []
    truncated = False
    try:
        for entry in base.glob(glob_pattern):
            if _skip_entry(entry, root):
                continue
            if not entry.is_file():
                continue
            if len(results) >= MAX_LISTED_FILES:
                truncated = True
                break
            results.append(_relative_display(entry, root))
    except (ValueError, OSError) as exc:
        raise ValueError(f"glob 模式无效: {pattern!r}（{exc}）") from exc

    results.sort()
    if truncated:
        logger.warning("list_files 命中超过 %s 条，结果已截断", MAX_LISTED_FILES)
        results.append("<truncated>")
    return results


def search_code(
    keyword: str,
    directory: str = ".",
    file_pattern: str = "*.py",
) -> List[Dict[str, Any]]:
    """在源码中搜索关键词。

    搜索大小写不敏感；跳过二进制文件、忽略目录与工作目录之外的内容。

    Args:
        keyword: 要搜索的关键词、函数名或变量名。
        directory: 搜索起始目录（相对实验工作目录）。
        file_pattern: 限定搜索的文件 glob 模式，默认 ``*.py``。

    Returns:
        命中列表，每个元素为 ``{"path": 相对路径, "line": 行号, "content": 该行内容}``。
        命中超过 ``MAX_SEARCH_HITS`` 条时，末尾追加一条哨兵记录
        （``path="<truncated>"``，``line=0``）提示需要缩小搜索范围。

    Raises:
        ValueError: 关键词为空、路径或模式越界。
        NotADirectoryError: ``directory`` 不是目录。
    """
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword 不能为空")
    needle = keyword.strip()
    needle_folded = needle.casefold()

    root = _workspace_root()
    base = _resolve_relative(directory, field_name="directory", must_exist=True)
    if not base.is_dir():
        raise NotADirectoryError(f"directory 必须是目录: {directory!r}")
    pattern = _normalize_glob(file_pattern, name="file_pattern")

    iterator = base.glob(pattern) if pattern.startswith("**/") else base.rglob(pattern)
    hits: List[Dict[str, Any]] = []
    truncated = False

    for entry in iterator:
        if _skip_entry(entry, root) or not entry.is_file():
            continue
        try:
            text = entry.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # 二进制或不可读文件直接跳过
        relative = _relative_display(entry, root)
        for line_no, line in enumerate(text.splitlines(), start=1):
            if needle_folded in line.casefold():
                snippet = line.strip()
                if len(snippet) > MAX_SEARCH_LINE_CHARS:
                    snippet = snippet[:MAX_SEARCH_LINE_CHARS] + "..."
                hits.append({"path": relative, "line": line_no, "content": snippet})
                if len(hits) >= MAX_SEARCH_HITS:
                    truncated = True
                    break
        if truncated:
            break

    if truncated:
        logger.warning("search_code 命中超过 %s 条，结果已截断", MAX_SEARCH_HITS)
        hits.append(
            {
                "path": "<truncated>",
                "line": 0,
                "content": f"命中超过 {MAX_SEARCH_HITS} 条，仅返回前 {MAX_SEARCH_HITS} 条，请缩小 keyword 或 directory 范围",
            }
        )
    return hits


# ---------------------------------------------------------------------------
# 工具：测试执行与命令执行
# ---------------------------------------------------------------------------

def _parse_test_output(runner: str, output: str, returncode: int, timed_out: bool) -> Dict[str, int]:
    """从测试运行器输出中解析通过 / 失败数量。"""
    passed = failed = errors = total = 0
    if runner == "pytest":
        match = re.search(r"(\d+) passed", output)
        passed = int(match.group(1)) if match else 0
        match = re.search(r"(\d+) failed", output)
        failed = int(match.group(1)) if match else 0
        match = re.search(r"(\d+) error", output)
        errors = int(match.group(1)) if match else 0
        if returncode != 0 and not timed_out and (failed + errors) == 0:
            failed = 1  # 无法解析但明确非零退出，保守记为 1 个失败
        total = passed + failed + errors
    else:
        match = re.search(r"Ran (\d+) tests?", output)
        total = int(match.group(1)) if match else 0
        match = re.search(r"failures=(\d+)", output)
        failures = int(match.group(1)) if match else 0
        match = re.search(r"errors=(\d+)", output)
        errors = int(match.group(1)) if match else 0
        failed = failures + errors
        passed = max(total - failed, 0)
        if returncode != 0 and not timed_out and failed == 0:
            failed = 1
    return {"passed": passed, "failed": failed, "errors": errors, "total": max(total, 0)}


def run_tests(test_path: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """运行测试。

    优先使用 pytest（若当前解释器可用），否则回退到标准库 ``unittest discover``。
    测试以工作目录为 cwd 运行，``PYTHONPATH`` 已包含工作目录。

    Args:
        test_path: 测试文件或目录（相对实验工作目录），为空表示运行全部测试。
        timeout: 超时时间（秒），上限 ``MAX_TIMEOUT_SECONDS``。

    Returns:
        结构化结果：``{"ok", "no_tests_found", "passed", "failed", "errors", "total",
        "returncode", "timed_out", "runner", "command", "test_path", "output"}``。
        其中 ``ok`` 的语义是"退出码为 0 **且** 确实发现了测试"——工作区里一个测试都没有时
        ``ok`` 为 ``False`` 且 ``no_tests_found`` 为 ``True``，避免"零测试"被误判为成功。
        ``output`` 为合并后的 stdout + stderr，超过上限时截断。

    Raises:
        FileNotFoundError: ``test_path`` 不存在。
        ValueError: 路径越界或超时参数非法。
    """
    root = _workspace_root()
    timeout_value = _normalize_timeout(timeout)

    relative_target: Optional[str] = None
    target: Optional[Path] = None
    if test_path is not None and str(test_path).strip():
        target = _resolve_relative(test_path, field_name="test_path", must_exist=True)
        relative_target = _relative_display(target, root)

    use_pytest = find_spec("pytest") is not None
    if use_pytest:
        runner = "pytest"
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]
        if relative_target:
            cmd.append(relative_target)
    else:
        runner = "unittest"
        if relative_target and target is not None:
            if target.is_file():
                cmd = [
                    sys.executable, "-m", "unittest", "discover",
                    "-s", _relative_display(target.parent, root),
                    "-p", target.name,
                ]
            else:
                cmd = [sys.executable, "-m", "unittest", "discover", "-s", relative_target]
        else:
            cmd = [sys.executable, "-m", "unittest", "discover"]

    result = _run_process(cmd, cwd=root, timeout=timeout_value, label="run_tests")
    combined = (result["stdout"] + "\n" + result["stderr"]).strip()

    counts = _parse_test_output(runner, combined, result["returncode"], result["timed_out"])

    # "退出码为 0"不等于"测试通过"：unittest 在目录里一个测试都没发现时同样返回 0。
    # 若不显式区分，会把"零测试"误判为成功，直接污染任务的 success 判定。
    no_tests_found = runner == "unittest" and bool(re.search(r"Ran 0 tests?", combined))
    if no_tests_found:
        combined += (
            "\n[提示] 未发现任何测试用例。unittest discover 只会递归包含 __init__.py 的"
            "测试子目录，请补上 __init__.py，或使用 test_path 直接指定测试目录 / 文件。"
        )
    ok = bool(result["returncode"] == 0 and not result["timed_out"] and not no_tests_found)

    return {
        "ok": ok,
        "no_tests_found": no_tests_found,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["errors"],
        "total": counts["total"],
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "runner": runner,
        "command": cmd,
        "test_path": relative_target,
        "output": _truncate_text(combined, label="测试输出"),
    }


def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """执行受限 shell 命令。

    默认被 ``AGENT_ALLOW_SHELL=false`` 完全禁用。启用后仅允许白名单程序，
    且不经过 shell 解析（``;`` / ``|`` / ``&&`` / 重定向不会生效）。

    Args:
        command: 要执行的命令（单行，不含 shell 元字符）。
        cwd: 命令工作目录（相对实验工作目录），为空表示工作目录根。
        timeout: 超时时间（秒），上限 ``MAX_TIMEOUT_SECONDS``。

    Returns:
        结构化执行结果：``{"returncode", "stdout", "stderr", "timed_out",
        "command", "argv", "cwd"}``；stdout/stderr 超长时截断。

    Raises:
        PermissionError: shell 执行被禁用，或程序不在白名单、参数被禁止。
        ValueError: 命令为空、含非法字符、路径越界或超时参数非法。
        FileNotFoundError: ``cwd`` 不存在。
    """
    settings = get_settings()
    if not settings.agent.allow_shell:
        raise PermissionError(
            "命令执行已被禁用（AGENT_ALLOW_SHELL=false）。"
            "如果任务确实需要执行命令，请在实验配置中显式开启该开关。"
        )
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command 不能为空")
    if "\x00" in command or "\n" in command or "\r" in command:
        raise ValueError("command 必须是单行字符串，且不能包含 NUL / 换行")

    argv = _split_command(command)
    if not argv:
        raise ValueError("command 解析后为空")

    program = os.path.basename(argv[0]).lower()
    program = re.sub(r"\.(exe|bat|cmd|com)$", "", program)
    if program not in SHELL_COMMAND_WHITELIST:
        allowed = ", ".join(sorted(SHELL_COMMAND_WHITELIST))
        raise PermissionError(
            f"程序 {argv[0]!r} 不在白名单内。允许的程序: {allowed}"
        )
    if program == "git":
        forbidden = [arg for arg in argv[1:] if arg in _GIT_FORBIDDEN_ARGS]
        if forbidden:
            raise PermissionError(
                f"git 参数 {forbidden} 会作用到工作目录之外，已被禁止"
            )

    root = _workspace_root()
    if cwd is None or not str(cwd).strip():
        workdir = root
        relative_cwd = "."
    else:
        workdir = _resolve_relative(cwd, field_name="cwd", must_exist=True)
        if not workdir.is_dir():
            raise NotADirectoryError(f"cwd 必须是目录: {cwd!r}")
        relative_cwd = _relative_display(workdir, root)

    timeout_value = _normalize_timeout(timeout)
    result = _run_process(argv, cwd=workdir, timeout=timeout_value, label="run_command")

    return {
        "returncode": result["returncode"],
        "stdout": _truncate_text(result["stdout"], label="stdout"),
        "stderr": _truncate_text(result["stderr"], label="stderr"),
        "timed_out": result["timed_out"],
        "command": command,
        "argv": argv,
        "cwd": relative_cwd,
    }


# ---------------------------------------------------------------------------
# 工具：Git 差异与补丁
# ---------------------------------------------------------------------------

def git_diff(path: Optional[str] = None) -> str:
    """查看代码差异。

    注意：工作目录必须位于某个 Git 仓库之内，否则抛出 ``RuntimeError``。返回的差异
    始终**限定在工作目录子树内**——任务工作区常位于外层仓库内部（如
    ``benchmark/workspaces/task_001/``），若不限制范围,外层仓库其它目录的改动会
    一起混进来，形成无法解释的噪声。

    Args:
        path: 限定查看差异的文件路径，为空表示查看整个仓库。

    Returns:
        统一 diff 格式的文本；无改动时返回说明字符串。超长时截断。

    Raises:
        RuntimeError: 当前工作目录不是 Git 仓库，或 git 执行失败。
        ValueError: 路径越界。
        FileNotFoundError: ``path`` 不存在。
    """
    root = _workspace_root()
    probe = _run_process(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        label="git_diff(probe)",
    )
    if probe["returncode"] != 0 or "true" not in probe["stdout"].lower():
        raise RuntimeError(
            f"工作目录不是 Git 仓库，无法查看差异: {root}。"
            f"请在该任务工作区执行 git init，或改用其他方式对比改动。"
        )

    cmd = ["git", "diff", "--no-color", "--no-ext-diff", "--"]
    if path is not None and str(path).strip():
        target = _resolve_relative(path, field_name="path", must_exist=True)
        cmd.append(_relative_display(target, root))
    else:
        # 任务工作区常常位于外层仓库内部（例如 benchmark/workspaces/task_001/）。
        # 不带 pathspec 的 git diff 会把外层仓库其它目录的改动一起返回，形成噪声，
        # 因此强制把差异范围限制在工作目录子树内。
        cmd.append(".")

    result = _run_process(cmd, cwd=root, timeout=DEFAULT_TIMEOUT_SECONDS, label="git_diff")
    if result["timed_out"]:
        raise RuntimeError("git diff 执行超时")
    if result["returncode"] != 0:
        raise RuntimeError(f"git diff 执行失败（{result['returncode']}）：{result['stderr'].strip()}")

    output = result["stdout"]
    if not output.strip():
        return "（工作区当前没有未提交的改动）"
    return _truncate_text(output, label="git diff")


# ---------------------------------------------------------------------------
# unified diff 解析与补丁应用
# ---------------------------------------------------------------------------

@dataclass
class _Hunk:
    """统一 diff 中的一个 hunk。"""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)


@dataclass
class _FilePatch:
    """统一 diff 中针对单个文件的改动集合。"""

    path: str
    hunks: List[_Hunk] = field(default_factory=list)
    creates_file: bool = False
    deletes_file: bool = False
    no_newline_at_end: bool = False


def _parse_diff_path(token: str) -> Optional[str]:
    """从 ``---`` / ``+++`` 行中提取文件路径；``/dev/null`` 返回 None。"""
    text = token.strip()
    if not text:
        return None
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1]
    text = text.split("\t")[0].strip()
    if text in ("/dev/null", "NUL", "nul"):
        return None
    if text.startswith(("a/", "b/")):
        text = text[2:]
    return text or None


def _parse_unified_diff(patch: str) -> List[_FilePatch]:
    """把统一 diff 文本解析为按文件分组的 hunk 列表。

    依赖 hunk 头部声明的行数判定 hunk 结束，因此内容行不会被误判为文件头。
    """
    files: List[_FilePatch] = []
    current: Optional[_FilePatch] = None
    hunk: Optional[_Hunk] = None
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    old_used = new_used = 0
    last_prefix = ""

    for raw in patch.splitlines():
        # ---- hunk 内部：按声明行数消费内容 ----
        if hunk is not None:
            if raw.startswith("\\"):
                current.no_newline_at_end = last_prefix == "+"
                continue
            prefix = raw[:1]
            if raw == "":
                prefix = " "
                raw = " "
            if prefix not in (" ", "-", "+"):
                raise ValueError(f"补丁 hunk 中出现无法解析的行: {raw!r}")
            hunk.lines.append(raw)
            if prefix in (" ",):
                old_used += 1
                new_used += 1
            elif prefix == "-":
                old_used += 1
            else:
                new_used += 1
            last_prefix = prefix
            if old_used > hunk.old_count or new_used > hunk.new_count:
                raise ValueError(
                    f"补丁 hunk 实际内容行数超过声明值 "
                    f"(@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@)：{raw!r}"
                )
            if old_used == hunk.old_count and new_used == hunk.new_count:
                hunk = None
            continue

        # ---- hunk 外部：文件头、hunk 头与元信息 ----
        if raw.startswith("--- "):
            old_path = _parse_diff_path(raw[4:])
            continue
        if raw.startswith("+++ "):
            new_path = _parse_diff_path(raw[4:])
            target_path = new_path or old_path
            if target_path is None:
                raise ValueError("补丁的文件头缺少有效路径（--- / +++ 均为 /dev/null）")
            current = _FilePatch(
                path=target_path,
                creates_file=old_path is None,
                deletes_file=new_path is None,
            )
            files.append(current)
            continue
        match = _HUNK_HEADER_RE.match(raw)
        if match:
            if current is None:
                raise ValueError("补丁缺少文件头（--- / +++），无法确定目标文件")
            hunk = _Hunk(
                old_start=int(match.group(1)),
                old_count=int(match.group(2)) if match.group(2) is not None else 1,
                new_start=int(match.group(3)),
                new_count=int(match.group(4)) if match.group(4) is not None else 1,
            )
            old_used = new_used = 0
            current.hunks.append(hunk)
            continue
        if raw.startswith("\\"):
            if current is not None and current.hunks:
                current.no_newline_at_end = last_prefix == "+"
            continue
        if raw[:1] in (" ", "-", "+"):
            raise ValueError(f"补丁 hunk 声明的行数与实际内容不一致，多余内容: {raw!r}")
        # diff --git / index / mode / similarity index 等元信息：忽略

    if hunk is not None:
        raise ValueError(
            f"补丁在 hunk 结束前截断 "
            f"(@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@)，"
            f"已收到 {len(hunk.lines)} 行"
        )
    return [item for item in files if item.hunks]


def _locate_hunk(
    lines: List[str],
    old_sequence: List[str],
    expected: int,
    *,
    relative_path: str,
    hunk_index: int,
    search_radius: int = 50,
) -> int:
    """在文件中定位 hunk 应插入/替换的起始下标（允许小幅位置漂移）。"""
    if not old_sequence:
        position = max(0, min(expected, len(lines)))
        return position

    lower = max(0, expected - search_radius)
    upper = min(len(lines) - len(old_sequence), expected + search_radius)
    candidates = [
        position
        for position in range(lower, upper + 1)
        if lines[position:position + len(old_sequence)] == old_sequence
    ]
    if not candidates:
        raise ValueError(
            f"补丁 hunk #{hunk_index} 无法应用到 {relative_path}："
            f"第 {expected + 1} 行附近找不到匹配的原始内容（可能上下文不匹配或文件已被改动）"
        )
    if expected in candidates:
        return expected
    if len(candidates) > 1:
        raise ValueError(
            f"补丁 hunk #{hunk_index} 在 {relative_path} 中匹配到多处（行号 {[c + 1 for c in candidates]}），"
            f"无法确定应用位置，请提供更多上下文行"
        )
    return candidates[0]


def _apply_file_patch(relative_path: str, original: Optional[str], file_patch: _FilePatch) -> Optional[str]:
    """在内存中计算补丁应用后的文件内容；返回 None 表示删除文件。"""
    if file_patch.deletes_file:
        if original is None:
            raise FileNotFoundError(f"补丁要删除的文件不存在: {relative_path}")
        return None

    if original is None:
        if not file_patch.creates_file:
            raise FileNotFoundError(f"补丁要修改的文件不存在: {relative_path}")
        work_lines: List[str] = []
        newline = "\n"
    else:
        work_lines = original.splitlines()
        newline = "\r\n" if "\r\n" in original else "\n"

    offset = 0
    for index, hunk in enumerate(file_patch.hunks, start=1):
        old_sequence = [line[1:] for line in hunk.lines if line[:1] in (" ", "-")]
        new_sequence = [line[1:] for line in hunk.lines if line[:1] in (" ", "+")]
        if len(old_sequence) != hunk.old_count:
            raise ValueError(
                f"补丁 hunk #{index} 在 {relative_path} 中声明处理 {hunk.old_count} 行，"
                f"实际提供 {len(old_sequence)} 行"
            )
        if len(new_sequence) != hunk.new_count:
            raise ValueError(
                f"补丁 hunk #{index} 在 {relative_path} 中声明产出 {hunk.new_count} 行，"
                f"实际提供 {len(new_sequence)} 行"
            )
        expected = (hunk.old_start - 1 + offset) if old_sequence else (hunk.old_start + offset)
        position = _locate_hunk(
            work_lines,
            old_sequence,
            expected,
            relative_path=relative_path,
            hunk_index=index,
        )
        work_lines[position:position + len(old_sequence)] = new_sequence
        offset += len(new_sequence) - len(old_sequence)

    if not work_lines:
        return ""
    trailing = "" if file_patch.no_newline_at_end else newline
    return newline.join(work_lines) + trailing


def apply_patch(patch: str) -> Dict[str, Any]:
    """应用代码补丁。

    采用"先全部计算、后统一写盘"的策略：任一文件应用失败则整体回滚，不留半成品。
    支持新建文件、修改文件与删除文件（``+++ /dev/null``）。

    Args:
        patch: 统一 diff 格式的补丁内容（可包含多个文件）。

    Returns:
        ``{"applied": True, "affected_files": [...], "file_count": N}``。

    Raises:
        ValueError: 补丁为空、格式错误、hunk 行数不符、上下文不匹配或多处歧义。
        FileNotFoundError: 补丁要修改/删除的文件不存在。
        PermissionError: 补丁路径越出工作目录（含 ``..`` 或绝对路径）。
    """
    if not isinstance(patch, str) or not patch.strip():
        raise ValueError("patch 不能为空")
    if "\x00" in patch:
        raise ValueError("patch 含有非法字符 NUL")

    root = _workspace_root(must_exist=False)
    file_patches = _parse_unified_diff(patch)
    if not file_patches:
        raise ValueError("补丁中未解析到任何 hunk，请确认是统一 diff（unified diff）格式")

    planned: List[Tuple[Path, Optional[str], Optional[str]]] = []
    affected: List[str] = []
    for item in file_patches:
        target = _resolve_relative(item.path, field_name="patch path", create_root=True)
        relative = _relative_display(target, root)
        if not _is_within(target, root):
            raise PermissionError(f"补丁路径越界: {item.path!r}")
        original: Optional[str] = None
        if target.exists():
            if target.is_dir():
                raise IsADirectoryError(f"补丁目标是目录: {relative}")
            original = _read_text_file(target, label=f"文件 {relative!r}", max_bytes=MAX_WRITE_BYTES)
        updated = _apply_file_patch(relative, original, item)
        planned.append((target, original, updated))
        if relative not in affected:
            affected.append(relative)

    # 全部计算成功后才落盘，保证失败时不留半成品
    for target, _original, updated in planned:
        if updated is None:
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                raise OSError(f"删除文件失败: {target}（{exc}）") from exc
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as handle:
                handle.write(updated)
        except OSError as exc:
            raise OSError(f"写入文件失败: {target}（{exc}）") from exc

    logger.info("补丁已应用到 %s 个文件: %s", len(affected), affected)
    return {"applied": True, "affected_files": affected, "file_count": len(affected)}


# ---------------------------------------------------------------------------
# 工具：增量编辑与文件管理
# ---------------------------------------------------------------------------

def edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """把文件中的一段旧文本精确替换为新文本。

    与 ``write_file`` 的区别：只改片段，不需要重写整个文件；与 ``apply_patch``
    的区别：不需要构造 unified diff。旧文本必须在文件中唯一匹配，否则拒绝执行，
    避免误改到不该改的位置。

    Args:
        path: 相对工作目录的文件路径。
        old_string: 要被替换的原文片段（需唯一匹配）。
        new_string: 替换后的新文本。
        replace_all: 为 True 时替换所有匹配；默认 False（多处匹配则报错）。

    Returns:
        替换结果说明。

    Raises:
        TypeError: ``new_string`` 不是字符串。
        ValueError: 参数为空、新旧文本相同、找不到旧文本，或多处匹配但未允许全部替换。
        FileNotFoundError: 目标文件不存在。
        IsADirectoryError: 目标是目录。
    """
    if not isinstance(old_string, str) or not old_string:
        raise ValueError("old_string 不能为空")
    if not isinstance(new_string, str):
        raise TypeError(f"new_string 必须是字符串，当前为 {type(new_string).__name__}")
    if old_string == new_string:
        raise ValueError("old_string 与 new_string 相同，无需修改")

    target = _resolve_relative(path, must_exist=True, expect_dir=False)
    content = _read_text_file(target, label=f"文件 {path!r}", max_bytes=MAX_WRITE_BYTES)

    # 兼容 CRLF 文件：先归一化为 LF 再匹配，写回时恢复原换行风格
    used_crlf = "\r\n" in content
    normalized = content.replace("\r\n", "\n") if used_crlf else content

    count = normalized.count(old_string)
    if count == 0:
        raise ValueError(
            f"在 {path!r} 中找不到 old_string 片段，无法替换。"
            f"请先用 read_file 或 search_code 确认片段原文（含缩进与空白）。"
        )
    if count > 1 and not replace_all:
        raise ValueError(
            f"old_string 在 {path!r} 中出现 {count} 次，无法确定替换位置。"
            f"请提供更长的唯一片段，或显式设置 replace_all=True。"
        )

    updated = normalized.replace(old_string, new_string)
    encoded = updated.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        raise ValueError(
            f"替换后内容过大：{len(encoded)} 字节，超过上限 {MAX_WRITE_BYTES} 字节"
        )
    if used_crlf:
        updated = updated.replace("\n", "\r\n")

    try:
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
    except OSError as exc:
        raise OSError(f"写入失败: {path!r}（{exc}）") from exc

    replaced = count if replace_all else 1
    logger.info("已在 %s 中替换 %d 处", path, replaced)
    return f"已在 {path} 中替换 {replaced} 处（{len(old_string)} → {len(new_string)} 字符）"


def file_info(path: str) -> Dict[str, Any]:
    """查看文件或目录的属性。

    Args:
        path: 相对工作目录的文件或目录路径。

    Returns:
        ``{"path", "type", "size_bytes", "mtime", "is_text", "lines"}``；
        目录额外包含 ``entries``（直接子项数量）。
    """
    root = _workspace_root()
    target = _resolve_relative(path, must_exist=True)
    try:
        stat_result = target.stat()
    except OSError as exc:
        raise OSError(f"无法读取属性: {path!r}（{exc}）") from exc

    info: Dict[str, Any] = {
        "path": _relative_display(target, root),
        "type": "dir" if target.is_dir() else "file",
        "size_bytes": stat_result.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat_result.st_mtime)),
        "is_text": False,
        "lines": None,
    }

    if target.is_dir():
        try:
            info["entries"] = len(list(target.iterdir()))
        except OSError:
            info["entries"] = None
        return info

    try:
        data = target.read_bytes()
    except OSError as exc:
        raise OSError(f"读取失败: {path!r}（{exc}）") from exc
    if b"\x00" in data:
        return info
    text = data.decode("utf-8", errors="replace")
    info["is_text"] = True
    info["lines"] = len(text.splitlines())
    return info


def delete_path(path: str) -> str:
    """删除文件或空目录。

    为避免误删，只允许删除空目录；工作目录根本身不可删除。

    Args:
        path: 相对工作目录的文件或空目录路径。

    Returns:
        删除结果说明。

    Raises:
        FileNotFoundError: 目标不存在。
        PermissionError: 试图删除工作目录根。
        OSError: 目录非空，或删除失败。
    """
    root = _workspace_root()
    target = _resolve_relative(path, must_exist=True)
    if os.path.normcase(str(target)) == os.path.normcase(str(root)):
        raise PermissionError("不能删除工作目录本身")

    if target.is_dir():
        try:
            target.rmdir()  # 只删空目录
        except OSError as exc:
            raise OSError(
                f"目录非空或无法删除: {path!r}，请先删除其中文件（{exc}）"
            ) from exc
        logger.info("已删除空目录 %s", path)
        return f"已删除空目录 {path}"

    try:
        target.unlink()
    except OSError as exc:
        raise OSError(f"删除文件失败: {path!r}（{exc}）") from exc
    logger.info("已删除文件 %s", path)
    return f"已删除文件 {path}"


def move_path(source: str, destination: str, overwrite: bool = False) -> str:
    """重命名或移动文件与目录。

    Args:
        source: 原路径（相对工作目录，必须存在）。
        destination: 目标路径（相对工作目录，父目录会自动创建）。
        overwrite: 目标已存在时是否覆盖。

    Returns:
        移动结果说明。

    Raises:
        FileNotFoundError: 源路径不存在。
        FileExistsError: 目标已存在且未允许覆盖。
        PermissionError: 试图移动工作目录根。
        ValueError: 源与目标相同。
    """
    root = _workspace_root()
    src = _resolve_relative(source, field_name="source", must_exist=True)
    dst = _resolve_relative(destination, field_name="destination", create_root=True)

    if os.path.normcase(str(src)) == os.path.normcase(str(root)):
        raise PermissionError("不能移动工作目录本身")
    if os.path.normcase(str(src)) == os.path.normcase(str(dst)):
        raise ValueError("source 与 destination 相同，无需移动")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"目标已存在: {destination!r}，如需覆盖请显式传入 overwrite=True。"
            )
        try:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        except OSError as exc:
            raise OSError(f"清理目标路径失败: {destination!r}（{exc}）") from exc

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as exc:
        raise OSError(f"移动失败: {source!r} -> {destination!r}（{exc}）") from exc

    logger.info("已移动 %s -> %s", source, destination)
    return f"已把 {source} 移动到 {_relative_display(dst, root)}"


# ---------------------------------------------------------------------------
# 工具：符号导航与标记扫描
# ---------------------------------------------------------------------------

def _iter_source_files(base: Path, root: Path, pattern: str) -> List[Path]:
    """遍历匹配的源码文件，跳过忽略目录、越界项与超过数量上限的部分。"""
    iterator = base.glob(pattern) if pattern.startswith("**/") else base.rglob(pattern)
    files: List[Path] = []
    for entry in iterator:
        if len(files) >= MAX_LISTED_FILES:
            break
        if _skip_entry(entry, root) or not entry.is_file():
            continue
        files.append(entry)
    return files


def find_symbol(
    name: str = "",
    directory: str = ".",
    file_pattern: str = "*.py",
) -> List[Dict[str, Any]]:
    """列出源码中的函数与类定义。

    与 ``search_code`` 的区别：这里做语法解析而不是文本搜索，因此能准确区分
    「定义一个函数」和「调用一个函数」。

    Args:
        name: 按符号名过滤（大小写不敏感的子串匹配），为空时列出全部。
        directory: 搜索起始目录（相对工作目录）。
        file_pattern: 限定文件类型，默认 ``*.py``。

    Returns:
        命中列表，元素为 ``{"name", "kind", "path", "line", "end_line"}``，
        按路径与行号排序；超过 ``MAX_SEARCH_HITS`` 条时末尾追加一条
        ``path="<truncated>"`` 的哨兵记录。
    """
    root = _workspace_root()
    base = _resolve_relative(directory, field_name="directory", must_exist=True)
    if not base.is_dir():
        raise NotADirectoryError(f"directory 必须是目录: {directory!r}")
    pattern = _normalize_glob(file_pattern, name="file_pattern")
    keyword = str(name or "").strip().lower()

    hits: List[Dict[str, Any]] = []
    truncated = False
    for entry in _iter_source_files(base, root, pattern):
        try:
            source = entry.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        try:
            tree = ast.parse(source, filename=str(entry))
        except SyntaxError:
            continue
        relative = _relative_display(entry, root)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if keyword and keyword not in node.name.lower():
                continue
            hits.append(
                {
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "path": relative,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", None),
                }
            )
            if len(hits) >= MAX_SEARCH_HITS:
                truncated = True
                break
        if truncated:
            break

    hits.sort(key=lambda item: (item["path"], item["line"]))
    if truncated:
        logger.warning("find_symbol 命中超过 %s 条，结果已截断", MAX_SEARCH_HITS)
        hits.append(
            {
                "name": "<truncated>",
                "kind": "",
                "path": "<truncated>",
                "line": 0,
                "end_line": None,
            }
        )
    return hits


#: 未完成标记（TODO / FIXME / XXX / HACK）
_TODO_PATTERN = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)


def list_todos(directory: str = ".", file_pattern: str = "*.py") -> List[Dict[str, Any]]:
    """扫描源码中的 TODO、FIXME 等未完成标记。

    Args:
        directory: 搜索起始目录（相对工作目录）。
        file_pattern: 限定文件类型，默认 ``*.py``。

    Returns:
        命中列表，元素为 ``{"path", "line", "marker", "content"}``；
        超过 ``MAX_SEARCH_HITS`` 条时末尾追加 ``path="<truncated>"`` 的哨兵记录。
    """
    root = _workspace_root()
    base = _resolve_relative(directory, field_name="directory", must_exist=True)
    if not base.is_dir():
        raise NotADirectoryError(f"directory 必须是目录: {directory!r}")
    pattern = _normalize_glob(file_pattern, name="file_pattern")

    hits: List[Dict[str, Any]] = []
    truncated = False
    for entry in _iter_source_files(base, root, pattern):
        try:
            text = entry.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = _relative_display(entry, root)
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = _TODO_PATTERN.search(line)
            if not match:
                continue
            snippet = line.strip()
            if len(snippet) > MAX_SEARCH_LINE_CHARS:
                snippet = snippet[:MAX_SEARCH_LINE_CHARS] + "..."
            hits.append(
                {
                    "path": relative,
                    "line": line_no,
                    "marker": match.group(1).upper(),
                    "content": snippet,
                }
            )
            if len(hits) >= MAX_SEARCH_HITS:
                truncated = True
                break
        if truncated:
            break

    if truncated:
        logger.warning("list_todos 命中超过 %s 条，结果已截断", MAX_SEARCH_HITS)
        hits.append(
            {
                "path": "<truncated>",
                "line": 0,
                "marker": "",
                "content": f"命中超过 {MAX_SEARCH_HITS} 条，仅返回前 {MAX_SEARCH_HITS} 条",
            }
        )
    return hits


# ---------------------------------------------------------------------------
# 工具：语法检查与 Git 状态
# ---------------------------------------------------------------------------

def check_syntax(path: str = ".", file_pattern: str = "*.py") -> Dict[str, Any]:
    """对源码做语法检查，报告无法解析的文件与错误位置。

    Args:
        path: 文件或目录路径（相对工作目录），默认检查整个工作目录。
        file_pattern: 当 ``path`` 是目录时限定文件类型，默认 ``*.py``。

    Returns:
        ``{"ok", "checked", "error_count",
        "errors": [{"path", "line", "offset", "message", "text"}]}``。
    """
    root = _workspace_root()
    target = _resolve_relative(path, field_name="path", must_exist=True)
    pattern = _normalize_glob(file_pattern, name="file_pattern")

    files = [target] if target.is_file() else _iter_source_files(target, root, pattern)

    errors: List[Dict[str, Any]] = []
    checked = 0
    for entry in files:
        try:
            source = entry.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        checked += 1
        try:
            compile(source, str(entry), "exec")
        except SyntaxError as exc:
            text = (exc.text or "").strip()
            errors.append(
                {
                    "path": _relative_display(entry, root),
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "message": exc.msg,
                    "text": text[:200],
                }
            )

    if errors:
        logger.info("语法检查发现 %d 个问题（共检查 %d 个文件）", len(errors), checked)
    return {
        "ok": not errors,
        "checked": checked,
        "error_count": len(errors),
        "errors": errors[:MAX_SEARCH_HITS],
    }


def git_status() -> Dict[str, Any]:
    """查看仓库工作区状态。

    返回 ``git status --porcelain`` 的解析结果，范围限定在工作目录子树内
    （工作区可能位于外层仓库内部）。

    Returns:
        ``{"clean", "count", "entries": [{"status", "path"}], "truncated"}``。

    Raises:
        RuntimeError: 工作目录不在 Git 仓库内，或 git 执行失败。
    """
    root = _workspace_root()
    probe = _run_process(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        label="git_status(probe)",
    )
    if probe["returncode"] != 0 or "true" not in probe["stdout"].lower():
        raise RuntimeError(
            f"工作目录不是 Git 仓库，无法查看状态: {root}。"
            f"请在该任务工作区执行 git init，或改用其他方式确认改动。"
        )

    result = _run_process(
        ["git", "status", "--porcelain", "--", "."],
        cwd=root,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        label="git_status",
    )
    if result["timed_out"]:
        raise RuntimeError("git status 执行超时")
    if result["returncode"] != 0:
        raise RuntimeError(
            f"git status 执行失败（{result['returncode']}）：{result['stderr'].strip()}"
        )

    entries: List[Dict[str, str]] = []
    truncated = False
    for line in result["stdout"].splitlines():
        if not line.strip():
            continue
        if len(entries) >= MAX_LISTED_FILES:
            truncated = True
            break
        code = line[:2].strip() or "?"
        path_text = line[3:].strip() if len(line) > 3 else ""
        entries.append({"status": code, "path": path_text})

    return {
        "clean": not entries,
        "count": len(entries),
        "entries": entries,
        "truncated": truncated,
    }


__all__ = [
    "read_file",
    "write_file",
    "list_files",
    "search_code",
    "find_symbol",
    "list_todos",
    "file_info",
    "edit_file",
    "delete_path",
    "move_path",
    "run_tests",
    "check_syntax",
    "run_command",
    "git_status",
    "git_diff",
    "apply_patch",
    "configure_workspace",
]
