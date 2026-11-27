# 面向 Coding Agent 的任务感知动态工具选择方法研究

**Task-Aware Dynamic Tool Selection for Coding Agents**

一个轻量级 Python Coding Agent 研究原型：根据当前编程任务，从工具库中动态选择合适工具子集提供给大模型，并通过对照实验比较不同工具提供策略在任务成功率、Token 消耗与执行效率上的差异。

---

## 1. 项目简介

Coding Agent 通常把全部工具 schema 随每轮请求发送给模型。工具数量增加后，这会带来两类问题：一是 schema 占用大量上下文 Token；二是无关工具干扰模型决策。本项目研究的问题是：

> 能否在每一轮任务开始前，根据任务语义与复杂度，只向模型提供"当前任务真正需要"的工具？

项目支持三种可对照的实验模式：

| 实验模式 | 工具提供策略 | 作用 |
|---|---|---|
| `baseline` | 提供工具库中的全部工具 | 对照组，衡量动态选择的收益 |
| `fixed_topk` | TF-IDF 检索后固定选择 Top-K | 验证"一定程度筛选"的效果 |
| `adaptive_topk` | TF-IDF 检索后按任务复杂度动态确定 K | 验证"任务感知"相对固定 K 的优势 |

三种模式共用同一套 Agent 执行逻辑，唯一差异是"进入循环前向模型暴露哪些工具"。

### 当前进度

| 状态 | 内容 |
|---|---|
| 已完成 | Agent 框架（多轮 Function Calling）、LLMClient（OpenAI 兼容）、TaskAnalyzer、TF-IDF 检索、AdaptiveSelector、工具库与安全边界、真实 Qwen 联调脚本、114 项单元测试 |
| 待完成 | `experiments/` 三个实验入口与指标实现（成员 C）、Benchmark 任务扩充（成员 B）、结果统计与可视化（成员 D） |

---

## 2. 目录结构

```text
coding-agent-tool-selection/
├── agent/                     # Coding Agent 核心（成员 A）
│   ├── controller.py          # AgentController：任务编排与多轮工具调用循环
│   ├── llm_client.py          # LLMClient：OpenAI 兼容接口封装
│   └── task_analyzer.py       # TaskAnalyzer / TaskInfo：自然语言任务结构化
│
├── retrieval/                 # 动态工具选择（成员 A）
│   ├── tfidf_retriever.py     # TF-IDF + Cosine Similarity 工具检索
│   └── adaptive_selector.py   # Fixed / Adaptive Top-K 选择
│
├── tools/                     # 工具库（成员 B）
│   ├── definitions.py         # ToolDefinition 与 16 个工具元数据
│   ├── implementations.py     # 工具函数实现与工作区安全边界
│   └── registry.py            # ToolRegistry：注册、schema 生成、执行
│
├── benchmark/                 # 实验任务与标注（成员 B）
│   ├── tasks.json             # Benchmark 任务（当前 3 条，task_001 已有示例工作区）
│   ├── expected_tools.json    # required / optional 工具标注
│   └── workspaces/            # 每个任务独立的初始代码工作区
│       ├── task_001/          # 已提供 calculator 示例项目与测试
│       ├── task_002/          # 占位，待成员 B 填充
│       └── task_003/          # 占位，待成员 B 填充
│
├── experiments/               # 实验入口与评价（成员 C）
│   ├── baseline.py            # run_baseline（待实现）
│   ├── fixed_topk.py          # run_fixed_topk（待实现）
│   ├── adaptive_topk.py       # run_adaptive_topk（待实现）
│   ├── evaluate.py            # 指标接口（待实现）
│   └── schemas.py             # ExperimentConfig / ExperimentResult
│
├── results/                   # 实验结果（成员 D）
│   ├── raw_results.csv        # 每次实验原始记录（空文件占位）
│   ├── summary.csv            # 实验汇总（空文件占位）
│   └── figures/               # 统计图表
│
├── scripts/                   # 联调与诊断脚本（手动执行）
│   ├── probe_retrieval.py     # TF-IDF 检索诊断：排名、交集词、召回、噪声词
│   ├── test_qwen_api.py       # 真实 API 连通性冒烟
│   └── test_qwen_agent.py     # 真实三种模式 Function Calling 联调
│
├── tests/                     # 单元测试（114 项）
├── docs/architecture.md       # 架构与接口契约
├── workspace/                 # 工具文件操作 / 命令执行的活动目录
│
├── config.py                  # 统一配置与 .env 加载
├── requirements.txt
├── .env.example               # 环境变量模板（复制为 .env 使用）
├── .gitignore
└── README.md
```

---

## 3. 技术栈

| 类别 | 选型 | 用途 |
|---|---|---|
| 语言 | Python 3.10+ | 全部模块（依赖 `dataclass`、`X | None` 等语法） |
| 检索 | scikit-learn | `TfidfVectorizer`、`cosine_similarity` |
| 模型调用 | openai SDK | 对接 Qwen、DeepSeek 等 OpenAI 兼容接口 |
| 配置 | python-dotenv | 自动加载项目根目录 `.env` |
| 测试 | 标准库 unittest | 不引入额外测试框架 |
| 可视化 | matplotlib（暂未启用） | 成员 D 后续启用 |

项目不引入 LangChain、AutoGen 等 Agent 框架，执行循环由 `AgentController` 自行实现。

---

## 4. 环境配置

```bash
# 1. 创建并激活虚拟环境（Windows PowerShell）
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 准备环境变量
copy .env.example .env
# 编辑 .env，至少填写 LLM_API_KEY
```

`.env` 已被 `.gitignore` 忽略，**禁止提交真实密钥**。系统环境变量优先于 `.env`，便于 CI 或临时覆盖。

### 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_MODEL` | `qwen4-max`（模板） | 模型名称 |
| `LLM_API_KEY` | 空 | API Key，仅从环境变量读取 |
| `LLM_BASE_URL` | 阿里云 MaaS 兼容地址（模板） | OpenAI 兼容接口地址 |
| `LLM_TEMPERATURE` | `0.0` | 采样温度 |
| `LLM_MAX_TOKENS` | `2048` | 单次回复上限 |
| `AGENT_MAX_STEPS` | `8` | Agent 最大执行轮数 |
| `AGENT_WORKSPACE_DIR` | `workspace` | 工具文件操作 / 命令执行的活动目录 |
| `AGENT_ALLOW_SHELL` | `false` | 是否允许 `run_command`（默认关闭） |
| `RETRIEVAL_TOP_K` | `5` | Fixed Top-K 的 K |
| `ADAPTIVE_MIN_K` / `ADAPTIVE_MAX_K` | `3` / `8` | Adaptive Top-K 的取值区间 |
| `DEFAULT_EXPERIMENT_MODE` | `baseline` | 默认实验模式 |

---

## 5. 快速开始

### 5.1 装配一次 Agent 运行

```python
from agent.controller import AgentController
from agent.llm_client import LLMClient
from agent.task_analyzer import TaskAnalyzer, TaskInfo
from config import get_settings
from retrieval.adaptive_selector import AdaptiveSelector, SelectionConfig
from retrieval.tfidf_retriever import TFIDFRetriever
from tools.registry import build_default_registry

settings = get_settings()
llm = LLMClient(
    model=settings.model.model,
    api_key=settings.model.api_key,
    base_url=settings.model.base_url,
    temperature=settings.model.temperature,
)

registry = build_default_registry()
retriever = TFIDFRetriever().fit(registry.get_all_tools())
selector = AdaptiveSelector(SelectionConfig(top_k=5, min_k=3, max_k=8))

agent = AgentController(
    llm_client=llm,
    task_analyzer=TaskAnalyzer(llm),
    tool_registry=registry,
    retriever=retriever,
    selector=selector,
)

task = TaskInfo(
    task_id="demo",
    description="修复 calculator.py 的除零异常并补充测试",
    task_type="bug_fix",
    language="python",
    intents=["读取代码", "修改函数", "运行测试"],
    complexity="simple",
)

for mode in ("baseline", "fixed_topk", "adaptive_topk"):
    result = agent.run(task, mode=mode, max_steps=8)
    print(mode, result.selected_tools, result.tool_call_count,
          result.total_tokens, result.termination_reason)
```

`task` 支持三种输入：

- `TaskInfo`：跳过分析，直接使用；
- 完整 benchmark 任务字典（含 `task_id / description / task_type / language / complexity`）：直接转 `TaskInfo`，保证三种模式使用同一份任务语义；
- 自然语言字符串：调用 `TaskAnalyzer.analyze()` 分析。

### 5.2 手动联调脚本

```bash
python scripts/test_qwen_api.py       # 文本请求 + Token usage
python scripts/test_qwen_agent.py     # 三种模式的真实 Function Calling
py scripts/probe_retrieval.py         # 检索排名 / 召回 / 噪声诊断
```

以上脚本需要真实 API Key，**不进入 unittest 自动流程**。

### 5.3 运行测试

```bash
python -m unittest discover -s tests -v
```

当前 114 项测试全部通过。

---

## 6. 模块职责

| 模块 | 职责 | 不负责 |
|---|---|---|
| `agent/llm_client.py` | 消息发送、工具 schema 透传、文本 / tool_calls / Token 解析、异常统一 | 执行工具 |
| `agent/task_analyzer.py` | 自然语言 → `TaskInfo`，字段校验与 JSON 解析 | 检索、选择、执行 |
| `agent/controller.py` | 任务分析、工具选择、schema 获取、多轮工具调用、结果记录 | TF-IDF / Top-K 算法、工具业务逻辑 |
| `retrieval/tfidf_retriever.py` | 中文友好分词、工具索引、Cosine 相似度排序 | 决定最终工具数量 |
| `retrieval/adaptive_selector.py` | Fixed / Adaptive Top-K 选择 | 检索打分 |
| `tools/definitions.py` | 工具元数据（模型描述 + 检索 hints） | 实现与注册 |
| `tools/implementations.py` | 工具函数、路径围栏、命令白名单、输出截断 | 决定暴露哪些工具 |
| `tools/registry.py` | 工具注册、查询、schema 转换、执行分发 | 工具业务逻辑 |
| `experiments/` | 三组实验入口、指标计算、结果结构 | Agent 执行逻辑 |
| `benchmark/` | 任务清单、初始工作区、Expected Tools 标注 | 实验执行 |
| `results/` | 原始结果、汇总结果、图表 | 采集逻辑 |
| `config.py` | 全局配置、`.env` 加载、路径常量 | 业务逻辑 |

---

## 7. 四人分工

| 成员 | 负责内容 | 代码目录 |
|---|---|---|
| 成员 A | 项目架构、Agent 框架、任务分析、动态工具选择算法、系统接口设计与集成 | `agent/`、`retrieval/`、`config.py`、`scripts/`（联调脚本） |
| 成员 B | 工具库设计与实现、Benchmark 任务与工作区、Expected Tools 标注 | `tools/`、`benchmark/` |
| 成员 C | 三组实验设计与执行、实验入口、自动化评估 | `experiments/` |
| 成员 D | 实验结果统计与可视化、项目文档、Demo 与最终汇报 | `results/`、`docs/`、`demo/`、（后续）`presentation/` |

协作约定：

- 成员 A 负责总体架构与跨模块接口协调（`experiments/schemas.py` 等共享数据结构由 A 定义），但不代替其他成员实现业务模块；
- `experiments/schemas.py` 位于成员 C 目录，但字段契约由 A 维护，修改需同步更新 `docs/architecture.md`；
- 工具使用两条描述：`description` 面向模型（进 schema，有 Token 成本），`search_hints` 面向检索器（零 Token 成本），由成员 B 维护。

---

## 8. 后续开发计划

| 阶段 | 工作内容 | 负责成员 |
|---|---|---|
| 1 | 补齐 20 个 Benchmark 任务、对应工作区与 Expected Tools；完成 task_002 / task_003 工作区 | B |
| 2 | 实现 `run_baseline / run_fixed_topk / run_adaptive_topk`，接入工作区复制 / 重置 | C |
| 3 | 实现 `evaluate.py` 六项指标（成功率、Token、耗时、调用次数、Recall、Precision） | C |
| 4 | 运行正式三组对照实验，产出 `raw_results.csv` / `summary.csv` | C |
| 5 | 结果统计、图表、Demo 与汇报材料 | D |
| 6 | 持续维护接口契约、检索效果调优（hints / 分词）、回归测试 | A |

---

## 9. 开发约定

- 不硬编码模型名、路径与 K 值，统一走 `config.py`；
- API Key 只从环境变量读取，不打印、不提交；
- 工具文件操作与命令执行限定在 `workspace`（或实验工作副本）内，`run_command` 默认关闭；
- 新增或修改跨模块接口时，同步更新 `docs/architecture.md` 与 `tests/`；
- 真实 API 联调脚本手动执行，不加入 unittest。
