"""实验工作副本管理。

正式实验必须保证「每次运行都从相同的初始代码状态开始」：本模块把
``benchmark/workspaces/<task_id>`` 复制到独立的实验目录，Agent 只在副本上读写，
组间与重复实验之间互不污染（约定见 benchmark/workspaces/README.md 第 5 节）。

同时提供成功判定所需的两项客观证据：
- ``snapshot_workspace`` / ``workspace_changed``：副本相对初始状态是否真的产生改动；
- ``run_workspace_tests``：在副本内运行测试（临时切换工具工作目录，用完即恢复）。

负责成员：成员 C
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from config import BENCHMARK_DIR, PROJECT_ROOT
from tools.implementations import configure_workspace, run_tests

logger = logging.getLogger(__name__)

#: 实验工作副本的默认根目录（运行时产物，不进入版本库）
DEFAULT_WORK_ROOT: Path = PROJECT_ROOT / ".experiment_workspaces"

#: 复制与快照时跳过的目录（缓存 / 版本库 / 依赖目录不属于任务初始状态）
IGNORED_DIR_NAMES: frozenset = frozenset(
    {
        ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".venv", "venv", "node_modules",
    }
)

#: 占位文件名：只有这些文件的工作区视为「尚未就绪」
PLACEHOLDER_FILES: frozenset = frozenset({".gitkeep"})

#: 评价阶段运行测试的默认超时（秒）
DEFAULT_TEST_TIMEOUT: int = 120


def resolve_path(value: Union[str, Path]) -> Path:
    """把相对路径按项目根目录解析为绝对路径。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def task_source_dir(task: Mapping[str, Any]) -> Path:
    """任务的初始工作区目录（``project_path``，缺省按 task_id 推断）。"""
    raw = task.get("project_path")
    if raw:
        return resolve_path(raw)
    return BENCHMARK_DIR / "workspaces" / str(task.get("task_id", ""))


def workspace_ready(source: Path) -> bool:
    """初始工作区是否已就绪：目录存在且含有占位文件之外的真实文件。"""
    if not source.is_dir():
        return False
    return any(
        path.is_file() and path.name not in PLACEHOLDER_FILES
        for path in source.rglob("*")
    )


def prepare_workspace(
    task: Mapping[str, Any],
    work_root: Optional[Union[str, Path]] = None,
    mode: str = "baseline",
    repeat: int = 1,
) -> Path:
    """为「任务 × 模式 × 重复序号」准备一份干净的初始工作副本。

    目标目录已存在时先删除再复制，因此本函数同时承担重复实验之间的重置职责。

    Args:
        task: benchmark 任务字典。
        work_root: 副本根目录，为空时使用 ``DEFAULT_WORK_ROOT``。
        mode: 实验模式，用于隔离不同实验组。
        repeat: 重复实验序号（从 1 开始）。

    Returns:
        副本目录的绝对路径。

    Raises:
        FileNotFoundError: 初始工作区不存在。
        ValueError: repeat 非正整数。
    """
    if repeat <= 0:
        raise ValueError(f"repeat 必须为正整数，当前为 {repeat!r}")
    task_id = str(task.get("task_id", ""))
    source = task_source_dir(task)
    if not source.is_dir():
        raise FileNotFoundError(f"任务 {task_id} 的初始工作区不存在: {source}")

    root = resolve_path(work_root) if work_root is not None else DEFAULT_WORK_ROOT
    destination = root / task_id / f"{mode}_r{repeat}"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*IGNORED_DIR_NAMES, "*.pyc"),
    )
    logger.info("任务 %s 模式 %s 第 %d 次：工作副本 %s", task_id, mode, repeat, destination)
    return destination


def snapshot_workspace(directory: Path) -> Dict[str, str]:
    """采集目录内所有文件的「相对路径 -> 内容 SHA256」快照。"""
    snapshot: Dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if any(part in IGNORED_DIR_NAMES for part in relative.parts):
            continue
        snapshot[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def workspace_changed(current: Path, initial: Path) -> bool:
    """当前目录相对初始状态是否发生了改动（新增 / 删除 / 修改均算）。"""
    return snapshot_workspace(current) != snapshot_workspace(initial)


def run_workspace_tests(
    directory: Path,
    timeout: int = DEFAULT_TEST_TIMEOUT,
) -> Dict[str, Any]:
    """在指定工作副本内运行测试，返回 ``tools.implementations.run_tests`` 的结构化结果。

    通过 ``configure_workspace`` 临时把工具根目录指向副本，调用结束后取消覆盖
    （回到 config 配置），避免影响同进程内的后续任务。
    """
    configure_workspace(str(directory))
    try:
        return run_tests(timeout=timeout)
    finally:
        configure_workspace(None)


__all__ = [
    "DEFAULT_WORK_ROOT",
    "IGNORED_DIR_NAMES",
    "PLACEHOLDER_FILES",
    "DEFAULT_TEST_TIMEOUT",
    "resolve_path",
    "task_source_dir",
    "workspace_ready",
    "prepare_workspace",
    "snapshot_workspace",
    "workspace_changed",
    "run_workspace_tests",
]