"""项目统一配置模块。

所有可变参数集中在此处管理，其他模块不要硬编码模型名称、路径、Top-K 等取值。
敏感信息（API Key 等）一律从环境变量读取，禁止写入代码或提交到仓库。

负责成员：成员 A（项目架构负责人）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
BENCHMARK_DIR: Path = PROJECT_ROOT / "benchmark"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
FIGURES_DIR: Path = RESULTS_DIR / "figures"

DEFAULT_TASKS_FILE: Path = BENCHMARK_DIR / "tasks.json"
DEFAULT_EXPECTED_TOOLS_FILE: Path = BENCHMARK_DIR / "expected_tools.json"
DEFAULT_RAW_RESULTS_FILE: Path = RESULTS_DIR / "raw_results.csv"
DEFAULT_SUMMARY_FILE: Path = RESULTS_DIR / "summary.csv"

# ---------------------------------------------------------------------------
# 枚举取值（以字符串常量形式定义，保持与 JSON / CSV 的可读性）
# ---------------------------------------------------------------------------

#: 三种实验模式
EXPERIMENT_MODES: tuple[str, ...] = ("baseline", "fixed_topk", "adaptive_topk")

#: 默认实验模式
DEFAULT_EXPERIMENT_MODE: str = "baseline"

#: 任务复杂度取值，由 TaskAnalyzer 输出
COMPLEXITY_LEVELS: tuple[str, ...] = ("simple", "medium", "complex")

#: 工具分类取值（占位，成员 B 可在此基础上扩展）
TOOL_CATEGORIES: tuple[str, ...] = ("file", "code_search", "git", "test", "shell", "other")

ExperimentMode = Literal["baseline", "fixed_topk", "adaptive_topk"]


# ---------------------------------------------------------------------------
# 环境变量读取辅助函数
# ---------------------------------------------------------------------------

def _env_optional(name: str) -> Optional[str]:
    """读取可选环境变量，空字符串视为未设置。"""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_str(name: str, default: str) -> str:
    value = _env_optional(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = _env_optional(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = _env_optional(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_optional(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    """把相对路径解析为相对项目根目录的绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else base / path


# ---------------------------------------------------------------------------
# 配置分组
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """大模型调用配置（由 LLMClient 使用）。"""

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048


@dataclass(frozen=True)
class AgentConfig:
    """Agent 执行流程配置。"""

    max_steps: int = 8
    #: 工具执行（文件操作 / 命令执行）允许的活动目录，安全边界由成员 B 实现
    workspace_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")
    allow_shell: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    """工具检索与选择配置。"""

    #: Fixed Top-K 模式的固定 K 值
    top_k: int = 5
    #: Adaptive Top-K 模式允许的工具数量下界
    adaptive_min_k: int = 3
    #: Adaptive Top-K 模式允许的工具数量上界
    adaptive_max_k: int = 8


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark 数据文件配置。"""

    tasks_file: Path = field(default_factory=lambda: DEFAULT_TASKS_FILE)
    expected_tools_file: Path = field(default_factory=lambda: DEFAULT_EXPECTED_TOOLS_FILE)


@dataclass(frozen=True)
class ResultsConfig:
    """实验结果输出配置。"""

    raw_results_file: Path = field(default_factory=lambda: DEFAULT_RAW_RESULTS_FILE)
    summary_file: Path = field(default_factory=lambda: DEFAULT_SUMMARY_FILE)
    figures_dir: Path = field(default_factory=lambda: FIGURES_DIR)


@dataclass(frozen=True)
class Settings:
    """项目整体配置聚合对象。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    results: ResultsConfig = field(default_factory=ResultsConfig)
    default_mode: str = DEFAULT_EXPERIMENT_MODE

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置（未设置的项使用默认值）。"""
        workspace_dir = _resolve_path(_env_str("AGENT_WORKSPACE_DIR", "workspace"))
        return cls(
            model=ModelConfig(
                model=_env_str("LLM_MODEL", "gpt-4o-mini"),
                api_key=_env_optional("LLM_API_KEY"),
                base_url=_env_optional("LLM_BASE_URL"),
                temperature=_env_float("LLM_TEMPERATURE", 0.0),
                max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
            ),
            agent=AgentConfig(
                max_steps=_env_int("AGENT_MAX_STEPS", 8),
                workspace_dir=workspace_dir,
                allow_shell=_env_bool("AGENT_ALLOW_SHELL", False),
            ),
            retrieval=RetrievalConfig(
                top_k=_env_int("RETRIEVAL_TOP_K", 5),
                adaptive_min_k=_env_int("ADAPTIVE_MIN_K", 3),
                adaptive_max_k=_env_int("ADAPTIVE_MAX_K", 8),
            ),
            default_mode=_env_str("DEFAULT_EXPERIMENT_MODE", DEFAULT_EXPERIMENT_MODE),
        )


_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例（首次调用时从环境变量加载）。"""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


__all__ = [
    "PROJECT_ROOT",
    "BENCHMARK_DIR",
    "RESULTS_DIR",
    "DOCS_DIR",
    "FIGURES_DIR",
    "DEFAULT_TASKS_FILE",
    "DEFAULT_EXPECTED_TOOLS_FILE",
    "DEFAULT_RAW_RESULTS_FILE",
    "DEFAULT_SUMMARY_FILE",
    "EXPERIMENT_MODES",
    "DEFAULT_EXPERIMENT_MODE",
    "COMPLEXITY_LEVELS",
    "TOOL_CATEGORIES",
    "ExperimentMode",
    "ModelConfig",
    "AgentConfig",
    "RetrievalConfig",
    "BenchmarkConfig",
    "ResultsConfig",
    "Settings",
    "get_settings",
]
