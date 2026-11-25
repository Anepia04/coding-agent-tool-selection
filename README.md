# 面向 Coding Agent 的任务感知动态工具选择方法研究

**Task-Aware Dynamic Tool Selection for Coding Agents**

## 1. 项目简介

本项目构建一个轻量级的 Python Coding Agent，用于研究：**在给定当前编程任务的前提下，如何从工具库中动态选择最合适的工具子集，再提供给大模型**。

工具既可以全量提供（简单但 Token 成本高、干扰多），也可以只提供与任务最相关的少量工具（更省 Token，但可能漏掉必需工具）。项目通过三组对照实验量化这一取舍。

| 实验模式 | 工具提供策略 | 作用 |
|---|---|---|
| Baseline | 每次提供**全部工具** | 对照组，代表"不做筛选"的基线水平 |
| Fixed Top-K | 按任务检索工具，**固定取前 K 个** | 验证"工具筛选本身"带来的收益 |
| Adaptive Top-K | 按任务检索后，**结合任务复杂度动态确定数量** | 验证"任务感知"相对固定 K 的优势 |

评价维度：**任务成功率、Token 消耗、执行效率、工具调用次数**，并用工具召回率 / 精确率辅助评价检索质量。

> 当前仓库处于**骨架阶段**：目录结构、模块职责、统一数据结构、模块间接口、配置与文档已完成；TF-IDF 检索、Cosine Similarity、Top-K 排序、自适应选择、任务分析、工具具体实现、多轮 Agent 执行等算法与业务逻辑均**尚未实现**，统一以 `NotImplementedError` 占位。

## 2. 项目目录结构

```
coding-agent-tool-selection/
├── agent/                        # Coding Agent 核心框架（成员 A）
│   ├── __init__.py
│   ├── controller.py             # AgentController：执行流程控制中心
│   ├── llm_client.py             # LLMClient：模型调用封装（LLMResponse / ToolCall）
│   └── task_analyzer.py          # TaskAnalyzer：任务结构化分析（TaskInfo）
│
├── tools/                        # 工具库（成员 B）
│   ├── __init__.py
│   ├── definitions.py            # ToolDefinition：工具元数据 + 示例占位工具
│   ├── implementations.py        # 工具函数签名（业务逻辑待实现）
│   └── registry.py               # ToolRegistry：注册 / 查询 / schema 转换 / 执行
│
├── retrieval/                    # 动态工具选择（成员 A，核心研究模块）
│   ├── __init__.py
│   ├── tfidf_retriever.py        # TFIDFRetriever：检索接口（RetrievedTool）
│   └── adaptive_selector.py      # AdaptiveSelector：固定 / 自适应 Top-K 选择接口
│
├── benchmark/                    # 实验任务与标注（成员 B）
│   ├── tasks.json                # 占位任务（当前 3 条）
│   ├── expected_tools.json       # 人工标注的 required / optional tools
│   └── workspaces/               # 任务工作区：每个任务一份独立项目环境
│       ├── README.md             # 工作区命名、隔离与重置约定
│       ├── task_001/             # 对应 task_001，当前仅 .gitkeep 占位
│       ├── task_002/             # 对应 task_002，当前仅 .gitkeep 占位
│       └── task_003/             # 对应 task_003，当前仅 .gitkeep 占位
│
├── experiments/                  # 实验运行与评价（成员 C）
│   ├── __init__.py
│   ├── baseline.py               # run_baseline
│   ├── fixed_topk.py             # run_fixed_topk
│   ├── adaptive_topk.py          # run_adaptive_topk
│   ├── evaluate.py               # 指标计算接口
│   └── schemas.py                # ExperimentConfig / ExperimentResult（辅助文件）
│
├── results/                      # 实验结果（成员 D）
│   ├── raw_results.csv           # 每次实验的原始记录（空文件占位）
│   ├── summary.csv               # 汇总结果（空文件占位）
│   └── figures/                  # 图表输出目录
│
├── demo/                         # 演示入口（成员 D）
│   └── __init__.py
│
├── docs/
│   └── architecture.md           # 系统架构与接口文档（必读）
│
├── tests/                        # 框架自检（辅助目录，仅标准库 unittest）
│   ├── __init__.py
│   └── test_skeleton.py
│
├── config.py                     # 统一配置（模型 / 路径 / Top-K / 实验模式）
├── requirements.txt              # 当前阶段无第三方依赖
├── .env.example                  # 环境变量模板
├── .gitignore
└── README.md
```

> 说明：`experiments/schemas.py` 与 `tests/` 是为保证「统一数据结构集中管理」与「框架可自检」而新增的辅助文件，未改动需求规定的核心目录职责。

## 3. 技术栈

| 分类 | 选型 | 说明 |
|---|---|---|
| 语言 | Python 3.10+ | 使用 `dataclasses`、`typing`、`pathlib`、`json`、`unittest` |
| 数据结构 | 标准库 `dataclass` | 全部跨模块数据结构均为 dataclass，并提供 `to_dict()` |
| 模型调用 | 供应商无关接口 | 后续按需接入 `openai` 或兼容网关 |
| 检索 | TF-IDF（后续实现） | 计划使用 `scikit-learn` |
| 统计与可视化 | pandas / matplotlib（后续） | 由成员 D 使用 |
| 测试 | 标准库 `unittest` | 当前阶段无第三方测试依赖 |

当前阶段 `requirements.txt` 中没有启用任何第三方依赖，项目零安装即可导入与自检。

## 4. 环境配置方法

```bash
# 1. 创建并激活虚拟环境（可选，但推荐）
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. 安装依赖（当前阶段无第三方依赖，可跳过）
pip install -r requirements.txt

# 3. 配置环境变量
#    Windows PowerShell 可手动设置，或使用后续接入的 python-dotenv 读取 .env
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
# 按需填写 LLM_MODEL / LLM_API_KEY / LLM_BASE_URL 等；.env 已被 .gitignore 忽略

# 4. 框架自检（导入检查 + 接口检查）
python -m unittest discover -s tests -v
```

约定：

- 所有命令均在**项目根目录**执行（模块使用 `import config` 等顶层导入）。
- 严禁把真实 API Key 写入代码或提交到仓库，敏感信息只能通过环境变量注入。
- 工具的文件操作与命令执行只能发生在 `AGENT_WORKSPACE_DIR`（默认 `workspace/`）内（由成员 B 实现安全边界）。

## 5. 模块职责与接口一览

| 模块 | 职责 | 关键类 / 函数 | 负责人 |
|---|---|---|---|
| `agent/llm_client.py` | 封装模型 API，隔离供应商差异，统一返回文本 + 工具调用 + Token | `LLMClient.chat()`、`LLMResponse`、`ToolCall` | A |
| `agent/task_analyzer.py` | 自然语言任务 → 结构化 `TaskInfo` | `TaskAnalyzer.analyze()`、`TaskInfo` | A |
| `agent/controller.py` | 编排任务分析 → 工具选择 → 模型调用 → 工具执行循环 | `AgentController.run()`、`AgentRunResult` | A |
| `tools/definitions.py` | 工具元数据与描述 | `ToolDefinition`、`EXAMPLE_TOOL_DEFINITIONS` | B |
| `tools/implementations.py` | 工具函数实现 | `read_file` / `write_file` / `search_code` / `run_tests` 等 | B |
| `tools/registry.py` | 注册、按名查询、schema 转换、执行 | `ToolRegistry`、`build_default_registry()` | B |
| `retrieval/tfidf_retriever.py` | 工具检索排序 | `TFIDFRetriever.fit()` / `.retrieve()`、`RetrievedTool` | A |
| `retrieval/adaptive_selector.py` | 固定 / 自适应 Top-K 选择 | `AdaptiveSelector.select()`、`SelectionConfig` | A |
| `benchmark/*.json` | 任务与预期工具标注 | - | B |
| `benchmark/workspaces/` | 各任务对应的实际项目工作区（`<task_id>/`，实验时由 C 副本化使用） | - | B（创建维护）、C（运行时使用） |
| `experiments/*` | 三组实验入口与指标计算 | `run_baseline` / `run_fixed_topk` / `run_adaptive_topk` / `calculate_*` | C |
| `results/`、`docs/`、`demo/` | 结果统计、可视化、文档与演示 | - | D |
| `config.py` | 统一配置与环境变量读取 | `Settings`、`get_settings()` | A |

详细字段定义、数据流、接口签名、三种实验模式的执行流程见 [architecture.md](docs/architecture.md)。

## 6. 四人分工

| 成员 | 负责内容 | 对应代码目录 |
|---|---|---|
| 成员 A（项目架构负责人） | Agent 框架、Task Analyzer、动态工具选择算法、系统接口设计与集成 | `agent/`、`retrieval/`、`config.py` |
| 成员 B | 工具库设计与实现、Benchmark Task 设计、Expected Tools 标注 | `tools/`、`benchmark/` |
| 成员 C | 三组实验设计与执行、实验脚本和自动化评估 | `experiments/` |
| 成员 D | 实验结果统计分析、可视化、项目文档和最终汇报 | `results/`、`docs/`、`demo/`、`presentation/` |

协作原则：

- 成员 A 负责总体架构与跨模块接口，但**不代替**其他成员实现其业务模块。
- 接口变更必须先同步到 `docs/architecture.md`，再修改调用方。
- 任何成员新增字段时，需同步更新统一数据结构与 `tests/test_skeleton.py` 中的断言。

## 7. 当前阶段完成情况

已完成：

- 全部目录与 Python 包（含 `__init__.py`）创建完成；
- 5 个统一数据结构（`TaskInfo`、`ToolDefinition`、`RetrievedTool`、`ExperimentConfig`、`ExperimentResult`）定义完成；
- 运行期结果结构 `AgentRunResult` / `AgentStep` / `LLMResponse` / `ToolCall` 定义完成；
- `config.py` 统一配置与环境变量读取完成；
- `ToolRegistry` 的注册、查询、schema 转换、执行分发可用（不含安全边界与参数校验）；
- 未实现接口全部抛出 `NotImplementedError`，并附 TODO 与负责成员说明；
- benchmark 占位数据、results 占位文件、README 与架构文档完成；
- `tests/test_skeleton.py` 通过（10 项自检用例）。

尚未实现（按要求本阶段禁止提前实现）：

1. TF-IDF 索引构建与检索、Cosine Similarity、Top-K 排序；
2. Adaptive Top-K 的复杂度规则与动态 K 决策；
3. TaskAnalyzer 的任务识别与复杂度判定；
4. 具体工具函数业务逻辑与安全边界；
5. Agent 多轮执行循环、真实模型调用；
6. 完整 Benchmark（20 个任务）与 Expected Tools 标注；
7. 实验脚本的真实运行、指标统计、结果落盘与可视化。

## 8. 后续开发计划

| 阶段 | 内容 | 主要负责 |
|---|---|---|
| 阶段一 | 补齐工具库实现与安全边界；扩充工具描述 | B |
| 阶段二 | 实现 LLMClient、TaskAnalyzer，打通单轮工具调用 | A |
| 阶段三 | 实现 TF-IDF 检索、Top-K 与自适应选择 | A |
| 阶段四 | 打通 Agent 多轮执行循环，形成三种模式入口 | A / C |
| 阶段五 | 完成 20 个 Benchmark 任务与 Expected Tools 标注 | B |
| 阶段六 | 运行三组对照实验，统计指标、生成图表与总结报告 | C / D |
