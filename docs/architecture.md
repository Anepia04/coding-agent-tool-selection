# 系统架构与接口文档

> 项目：面向 Coding Agent 的任务感知动态工具选择方法研究
> 适用范围：当前为骨架阶段，本文档定义各模块职责、统一数据结构、接口契约与协作方式；文中标注「待实现」的能力均以 `NotImplementedError` 占位。

## 1. 系统整体架构

系统分为六个层次：实验入口（experiments）、Agent 控制（agent）、工具选择（retrieval）、工具库（tools）、数据（benchmark / results）、配置与文档（config / docs）。

```
                         ┌──────────────────────────────────────────┐
                         │      experiments/  （成员 C）             │
                         │  run_baseline / run_fixed_topk           │
                         │  run_adaptive_topk / evaluate            │
                         └───────────────┬──────────────────────────┘
                                         │ agent.run(task, mode, max_steps)
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        agent/AgentController （成员 A）                        │
│                                                                               │
│  ① TaskAnalyzer.analyze ──► TaskInfo                                          │
│  ② _select_tools(task_info, mode) ──► selected_tool_names                     │
│  ③ ToolRegistry.get_schemas(names) ──► tool schemas                           │
│  ④ 循环：LLMClient.chat ──► ToolRegistry.execute_tool ──► 结果回填             │
│  ⑤ 汇总 AgentRunResult                                                        │
└───────┬───────────────────────────┬──────────────────────────┬────────────────┘
        │ TaskInfo                  │ RetrievedTool            │ ToolDefinition
        ▼                           ▼                          ▼
┌───────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ agent/            │   │ retrieval/ （成员 A）    │   │ tools/ （成员 B）        │
│ task_analyzer.py  │   │ TFIDFRetriever          │   │ ToolDefinition          │
│ llm_client.py     │   │ AdaptiveSelector        │   │ ToolRegistry            │
└───────────────────┘   └─────────────────────────┘   │ implementations         │
                                                      └─────────────────────────┘
        ▲                           ▲                          ▲
        │ benchmark/tasks.json      │                          │ workspace/（受控目录）
        └──── benchmark/expected_tools.json（成员 B）            └── results/（成员 D）
```

关键设计原则：

1. **单向依赖**：`experiments → agent → retrieval → tools`，`tools` 不反向依赖任何上层模块；`config.py` 可被所有模块依赖。
2. **接口隔离**：AgentController 只调用检索模块的 `retrieve()` / `select()`，不实现任何检索算法。
3. **数据结构统一**：跨模块传递的对象全部为 dataclass，且都提供 `to_dict()`，可直接序列化写 CSV / JSON。
4. **供应商无关**：`LLMClient` 是唯一与模型 API 耦合的位置。
5. **安全边界**：工具的路径与命令操作限定在 `AgentConfig.workspace_dir`，由成员 B 在未来实现时强制校验。

## 2. 各模块职责

### 2.1 agent/（成员 A）

| 文件 | 职责 | 不负责 |
|---|---|---|
| `llm_client.py` | 构造请求、调用模型 API、解析文本 / tool_calls / usage，返回 `LLMResponse` | 不决定使用哪些工具 |
| `task_analyzer.py` | 把自然语言任务转为 `TaskInfo`（类型、语言、意图、复杂度） | 不检索工具、不调用工具 |
| `controller.py` | 编排完整任务流程、按模式选工具、驱动多轮工具调用、记录 `AgentRunResult` | 不实现 TF-IDF / Top-K / 自适应算法 |

### 2.2 tools/（成员 B）

| 文件 | 职责 |
|---|---|
| `definitions.py` | 定义 `ToolDefinition`；维护工具描述文本与参数 JSON Schema（检索质量的关键素材） |
| `implementations.py` | 工具函数实现与安全边界（路径限制、命令白名单、超时） |
| `registry.py` | 注册工具、按名查询、批量查询、schema 转换、执行分发 |

### 2.3 retrieval/（成员 A，核心研究模块）

| 文件 | 职责 |
|---|---|
| `tfidf_retriever.py` | `fit()` 建立工具描述索引；`retrieve()` 对任务打分并返回排好序的 `RetrievedTool` 列表 |
| `adaptive_selector.py` | 在排序结果上执行 `fixed`（固定 K）或 `adaptive`（按复杂度动态 K）策略，输出最终工具名列表 |

### 2.4 experiments/（成员 C）

| 文件 | 职责 |
|---|---|
| `baseline.py` | 全量工具模式入口 |
| `fixed_topk.py` | 固定 Top-K 模式入口 |
| `adaptive_topk.py` | 自适应 Top-K 模式入口 |
| `evaluate.py` | 成功率、Token、耗时、调用次数、召回率、精确率计算 |
| `schemas.py` | `ExperimentConfig` / `ExperimentResult` 与 CSV 列定义 |

### 2.5 benchmark/、results/、docs/、demo/

- `benchmark/`：任务集（`tasks.json`）、人工标注的预期工具（`expected_tools.json`）与任务工作区（`workspaces/`）（成员 B 创建维护，成员 C 实验时副本化使用）。
- `results/`：`raw_results.csv`（逐次记录）、`summary.csv`（汇总）、`figures/`（图表）（成员 D）。
- `docs/`：架构与开发文档（成员 D 维护、成员 A 审核）。
- `demo/`：演示入口（成员 D）。

### 2.6 benchmark/workspaces/（任务工作区，成员 B 维护、成员 C 使用）

`benchmark/workspaces/` 存放每个 Benchmark Task 对应的实际 Python 项目环境，是 Agent 在实验中真正读写的代码对象。

| 约定 | 说明 |
|---|---|
| 目录命名 | 子目录名与 `tasks.json` 的 `task_id` 完全一致（当前为 `task_001` / `task_002` / `task_003`） |
| 独立结构 | 每个任务目录可自组织 `src/`、`tests/`、`pyproject.toml`、`README.md` 等，不强制统一 |
| 初始状态 | 同一任务的所有实验组与重复实验都从相同初始代码状态开始 |
| 隔离与重置 | 使用独立工作副本（推荐复制到临时实验目录）或可靠重置机制，禁止在初始工作区上直接迭代 |
| 归属 | 成员 B 创建与维护初始工作区；成员 C 在实验运行时复制 / 重置并使用；A、D 不改动其内容 |
| 当前状态 | 仅有 `task_001`~`task_003` 空目录（`.gitkeep` 占位），不含真实任务代码 |

详细约定见 `benchmark/workspaces/README.md`。

## 3. 模块之间的数据流

### 3.1 一次完整任务的数据流

```
benchmark/tasks.json
   │ 一条任务 dict
   ▼
AgentController.run(task, mode, max_steps)
   │
   ├─► TaskAnalyzer.analyze(description)
   │        └─► TaskInfo {task_id, description, task_type, language, intents, complexity}
   │
   ├─► _select_tools(task_info, mode)
   │        ├─ baseline      : ToolRegistry.tool_names
   │        ├─ fixed_topk    : TFIDFRetriever.retrieve(task_info) → AdaptiveSelector.select(strategy="fixed")
   │        └─ adaptive_topk : TFIDFRetriever.retrieve(task_info) → AdaptiveSelector.select(complexity, strategy="adaptive")
   │        └─► selected_tool_names: list[str]
   │
   ├─► ToolRegistry.get_schemas(selected_tool_names)
   │        └─► tools: list[dict]  (OpenAI function calling schema)
   │
   ├─► LLMClient.chat(messages, tools)
   │        └─► LLMResponse {content, tool_calls, input/output/total_tokens}
   │              │ 若 tool_calls 非空
   │              ├─► ToolRegistry.execute_tool(name, arguments)
   │              │        └─► 工具结果 → 追加到 messages（role="tool"）
   │              └─► 继续下一轮（最多 max_steps 轮）
   │
   └─► AgentRunResult {task_id, mode, task_success, selected_tools, steps,
                       tokens, execution_time, tool_call_count, error_message}
```

说明：任务对应的实际代码位于 `benchmark/workspaces/<task_id>/`。成员 C 在实验开始前把该目录复制为独立工作副本，Agent 的工具调用只作用于副本，因此 `benchmark/tasks.json` 提供任务语义，工作区提供任务环境。

### 3.2 从运行结果到实验记录

```
AgentRunResult
   │
   ├─► evaluate.evaluate_task_success()      → task_success
   ├─► evaluate.calculate_token_usage()      → input/output/total_tokens
   ├─► evaluate.calculate_execution_time()   → execution_time
   ├─► evaluate.calculate_tool_call_count()  → tool_call_count
   ├─► evaluate.calculate_tool_recall()      → 工具召回率（辅助指标）
   └─► evaluate.calculate_tool_precision()   → 工具精确率（辅助指标）
   │
   ▼
ExperimentResult → results/raw_results.csv
                 → results/summary.csv（按 mode 聚合）
                 → results/figures/（图表）
```

### 3.3 依赖方向

```
config.py  ◄── 所有模块
agent/controller.py  ──►  agent/llm_client.py, agent/task_analyzer.py
agent/controller.py  ──►  retrieval/*（仅调用接口）
agent/controller.py  ──►  tools/registry.py
retrieval/tfidf_retriever.py ──► tools/definitions.py, agent/task_analyzer.py（类型）
retrieval/adaptive_selector.py ──► retrieval/tfidf_retriever.py（类型）
experiments/* ──► agent/controller.py, experiments/schemas.py
tools/* ──► config.py（工作目录等配置）
```

> 为避免循环导入，`agent/controller.py` 对 `retrieval/`、`tools/` 的引用只用于类型标注（`TYPE_CHECKING`）。

## 4. 统一数据结构

所有结构均为 `dataclass`，并提供 `to_dict()`；字段名即 CSV / JSON 中的字段名。

### 4.1 TaskInfo（`agent/task_analyzer.py`）

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `task_id` | `str` | 是 | 任务唯一标识，与 `benchmark/tasks.json` 对应 |
| `description` | `str` | 是 | 自然语言任务描述，检索的查询文本 |
| `task_type` | `str` | 是 | 任务类型：`bug_fix` / `feature_add` / `refactor` / `test_write` / `doc` |
| `language` | `str` | 是 | 主要编程语言，如 `python` |
| `intents` | `list[str]` | 是 | 意图关键词，供检索增强与自适应选择使用 |
| `complexity` | `str` | 是 | 复杂度，取值 `simple` / `medium` / `complex` |

### 4.2 ToolDefinition（`tools/definitions.py`）

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `name` | `str` | 是 | 工具唯一名称，检索与执行标识 |
| `description` | `str` | 是 | 自然语言描述，TF-IDF 的核心输入 |
| `category` | `str` | 是 | 分类，取值见 `config.TOOL_CATEGORIES` |
| `parameters` | `dict` | 是 | JSON Schema（`type=object`，含 `properties` / `required`） |
| `function_name` | `str` | 是 | 对应 `tools.implementations` 中的函数名，默认同 `name` |

### 4.3 RetrievedTool（`retrieval/tfidf_retriever.py`）

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `tool_name` | `str` | 是 | 工具名称，必须能在 `ToolRegistry` 中查到 |
| `score` | `float` | 是 | 相关性分数，越大越相关（具体量纲由实现约定） |
| `rank` | `int` | 是 | 排名，从 1 开始，1 为最相关 |

### 4.4 ExperimentConfig（`experiments/schemas.py`）

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `mode` | `str` | 是 | 实验模式：`baseline` / `fixed_topk` / `adaptive_topk` |
| `top_k` | `int \| None` | 否 | Fixed Top-K 的 K；其他模式为 `None` |
| `max_steps` | `int` | 否 | Agent 最大执行轮数，默认 8 |
| `task_id` | `str` | 是 | 本次执行的任务 ID |

### 4.5 ExperimentResult（`experiments/schemas.py`）

| 字段 | 类型 | 含义 |
|---|---|---|
| `task_id` | `str` | 任务 ID |
| `experiment_mode` | `str` | 实验模式 |
| `task_success` | `bool` | 是否成功完成任务 |
| `input_tokens` | `int` | 输入 Token 总数 |
| `output_tokens` | `int` | 输出 Token 总数 |
| `total_tokens` | `int` | Token 总数 |
| `execution_time` | `float` | 执行耗时（秒） |
| `tool_call_count` | `int` | 工具调用次数 |
| `selected_tools` | `list[str]` | 提供给模型的工具名列表（顺序即提供给模型的顺序） |
| `error_message` | `str \| None` | 失败原因，成功为 `None` |

### 4.6 运行期辅助结构

- `LLMResponse` / `ToolCall`（`agent/llm_client.py`）：模型响应与工具调用请求。
- `AgentStep` / `AgentRunResult`（`agent/controller.py`）：逐步执行轨迹与运行汇总，`AgentRunResult` 是 `ExperimentResult` 的数据来源。
- `SelectionConfig`（`retrieval/adaptive_selector.py`）：`top_k` / `min_k` / `max_k`。

### 4.7 模块间传递约定

1. `TaskInfo`：`TaskAnalyzer` 产出 → `AgentController` 与 `TFIDFRetriever` 消费。
2. `ToolDefinition`：`ToolRegistry.get_all_tools()` 产出 → `TFIDFRetriever.fit()` 与 `get_schemas()` 消费。
3. `RetrievedTool`：`TFIDFRetriever.retrieve()` 产出 → `AdaptiveSelector.select()` 消费。
4. `list[str]`：`AdaptiveSelector.select()` / Baseline 的 `tool_names` 产出 → `ToolRegistry.get_schemas(names)` 消费。
5. `AgentRunResult`：`AgentController.run()` 产出 → `experiments/evaluate.py` 消费并转成 `ExperimentResult`。
6. `ExperimentResult`：`evaluate.py` / `experiments` 产出 → `results/*.csv` 与统计脚本消费。

## 5. 各模块接口定义

### 5.1 agent/llm_client.py

```python
class LLMClient:
    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, **options): ...
    def chat(self, messages: Sequence[Message],
             tools: Sequence[dict] | None = None, **kwargs) -> LLMResponse: ...
```

- 输入：OpenAI 风格消息列表；`tools` 为 `ToolRegistry.get_schemas()` 输出。
- 输出：`LLMResponse(content, tool_calls, input_tokens, output_tokens, total_tokens, raw)`。
- 约束：不得在代码中出现真实密钥；异常必须以清晰异常类型抛出，不得静默返回空响应。

### 5.2 agent/task_analyzer.py

```python
class TaskAnalyzer:
    def __init__(self, llm_client: LLMClient | None = None): ...
    def analyze(self, task_description: str, task_id: str | None = None) -> TaskInfo: ...
```

### 5.3 agent/controller.py

```python
class AgentController:
    def __init__(self, llm_client, task_analyzer, tool_registry,
                 retriever=None, selector=None, default_max_steps: int = 8): ...
    def run(self, task: TaskInfo | Mapping[str, Any] | str,
            mode: str = "baseline", max_steps: int | None = None) -> AgentRunResult: ...
    def _select_tools(self, task_info: TaskInfo, mode: str) -> list[str]: ...
```

- `run()` 必须校验 `mode ∈ config.EXPERIMENT_MODES`，非法模式抛 `ValueError`。
- `run()` 不得直接实现检索 / 排序算法，只能调用 retrieval 接口。
- `mode="fixed_topk"` / `"adaptive_topk"` 时 `retriever` 必填；`"adaptive_topk"` 时 `selector` 必填，缺失应抛 `ValueError`（待实现）。

### 5.4 tools/registry.py

```python
class ToolRegistry:
    def register_tool(self, definition: ToolDefinition, implementation: Callable) -> None
    def get_tool(self, name: str) -> tuple[ToolDefinition, Callable]
    def get_all_tools(self) -> list[ToolDefinition]
    def get_tools_by_names(self, names: Iterable[str]) -> list[ToolDefinition]
    def get_schemas(self, names: Iterable[str] | None = None) -> list[dict]
    def execute_tool(self, name: str, arguments: dict | None = None) -> Any
    @property
    def tool_names(self) -> list[str]
```

- `get_all_tools()` 用于 Baseline；`get_tools_by_names()` / `get_schemas(names)` 用于 Fixed / Adaptive Top-K。
- 未注册工具统一抛 `KeyError`；重复注册抛 `ValueError`。
- `execute_tool()` 待成员 B 增加参数校验与 `workspace_dir` 安全边界。

### 5.5 retrieval/tfidf_retriever.py

```python
class TFIDFRetriever:
    def fit(self, tool_definitions: Sequence[ToolDefinition]) -> "TFIDFRetriever": ...
    def retrieve(self, task_info: TaskInfo | str,
                 top_k: int | None = None) -> list[RetrievedTool]: ...
    @property
    def is_fitted(self) -> bool: ...
```

- `fit()` 必须缓存索引；未 `fit()` 直接 `retrieve()` 应抛 `RuntimeError`（待实现）。
- `retrieve()` 返回结果必须按 `score` 降序、`rank` 从 1 连续编号。

### 5.6 retrieval/adaptive_selector.py

```python
class AdaptiveSelector:
    def __init__(self, config: SelectionConfig | dict | None = None): ...
    def select(self, ranked_tools: Sequence[RetrievedTool],
               complexity: str | None = None,
               strategy: str = "fixed") -> list[str]: ...
```

- `strategy="fixed"`：取 `ranked_tools[: config.top_k]` 的工具名。
- `strategy="adaptive"`：K 由 `complexity` 与配置区间 `[min_k, max_k]` 决定。
- 输出顺序必须与 `ranked_tools` 一致，供 `get_schemas(names)` 使用。

### 5.7 experiments/

```python
def run_baseline(agent: AgentController, task: Mapping[str, Any]) -> AgentRunResult
def run_fixed_topk(agent: AgentController, task: Mapping[str, Any], top_k: int = 5) -> AgentRunResult
def run_adaptive_topk(agent: AgentController, task: Mapping[str, Any]) -> AgentRunResult
```

```python
def evaluate_task_success(result, task=None) -> bool
def calculate_token_usage(result) -> dict[str, int]
def calculate_execution_time(result) -> float
def calculate_tool_call_count(result) -> int
def calculate_tool_recall(selected_tools, expected_tools) -> float
def calculate_tool_precision(selected_tools, expected_tools) -> float
```

### 5.8 config.py

```python
class Settings:            # 聚合配置
    model: ModelConfig     # 模型名 / key / base_url / temperature / max_tokens
    agent: AgentConfig     # max_steps / workspace_dir / allow_shell
    retrieval: RetrievalConfig  # top_k / adaptive_min_k / adaptive_max_k
    benchmark: BenchmarkConfig  # tasks_file / expected_tools_file
    results: ResultsConfig      # raw_results_file / summary_file / figures_dir
    default_mode: str
def get_settings() -> Settings     # 单例，从环境变量加载
```

## 6. 三种实验模式的执行流程

### 6.1 Baseline

```
task ─► TaskAnalyzer ─► TaskInfo
      ─► selected = ToolRegistry.tool_names（全部工具）
      ─► schemas = ToolRegistry.get_schemas()（names=None）
      ─► 多轮 chat / execute_tool 循环
      ─► AgentRunResult
```

特点：不做任何筛选，Token 消耗最高，作为对照组。

### 6.2 Fixed Top-K

```
task ─► TaskAnalyzer ─► TaskInfo
      ─► ranked = TFIDFRetriever.retrieve(TaskInfo)
      ─► selected = AdaptiveSelector.select(ranked, strategy="fixed")   # 前 K 个
      ─► schemas = ToolRegistry.get_schemas(selected)
      ─► 多轮 chat / execute_tool 循环
      ─► AgentRunResult（selected_tools 长度恒为 K，不足 K 时取全部）
```

特点：筛选程度固定，用于观察"工具筛选本身"的收益；K 由实验网格（如 3/5/8）设定。

### 6.3 Adaptive Top-K

```
task ─► TaskAnalyzer ─► TaskInfo（含 complexity）
      ─► ranked = TFIDFRetriever.retrieve(TaskInfo)
      ─► selected = AdaptiveSelector.select(ranked, complexity=TaskInfo.complexity,
                                            strategy="adaptive")      # K 动态确定
      ─► schemas = ToolRegistry.get_schemas(selected)
      ─► 多轮 chat / execute_tool 循环
      ─► AgentRunResult（selected_tools 长度随任务变化）
```

特点：简单任务少给工具以降低 Token 与干扰，复杂任务多给工具以降低召回风险。

### 6.4 三类模式共用约定

| 约定 | 说明 |
|---|---|
| 统一入口 | 都通过 `AgentController.run(task, mode=...)` 执行 |
| 统一记录 | 都产出 `AgentRunResult`，再转为 `ExperimentResult` |
| 统一评价 | 都由 `experiments/evaluate.py` 计算同一组指标 |
| 唯一差异 | 仅 `selected_tool_names` 的来源不同 |
| 可复现 | 实验配置（K、max_steps、模型名）需写入 `ExperimentConfig` 与结果文件 |

## 7. 四人协作关系

```
                 ┌─────────────────────────────┐
                 │ 成员 A：架构 + agent/retrieval│
                 │  - 定义数据结构与接口          │
                 │  - 实现 Agent 与工具选择算法    │
                 └───────┬─────────────┬───────┘
                         │ 接口契约     │ 接口契约
          ┌──────────────▼───┐     ┌───▼──────────────────┐
          │ 成员 B：tools/    │     │ 成员 C：experiments/  │
          │ benchmark/        │     │  - 三组实验入口        │
          │  - 工具元数据与实现│     │  - 指标计算           │
          │  - 任务与标注      │     └──────────┬───────────┘
          └──────────────┬───┘                │ 结果数据
                         │ 工具清单 / 数据      │
                         ▼                    ▼
                 ┌────────────────────────────────────┐
                 │ 成员 D：results/ docs/ demo/        │
                 │  - 统计、可视化、文档、汇报          │
                 └────────────────────────────────────┘
```

协作规则：

1. **接口先行**：任何跨模块接口变更先改本文件，再改代码与调用方。
2. **数据结构唯一来源**：`TaskInfo` / `ToolDefinition` / `RetrievedTool` / `ExperimentConfig` / `ExperimentResult` 不得在别处重复定义。
3. **工具名一致性**：`benchmark/expected_tools.json` 中的工具名必须能在 `tools/definitions.py` 注册表中找到（自检测试已校验）。
4. **归属不重复**：成员 A 只定义并调用接口，不实现 B/C 的业务逻辑；B 不修改检索算法；C 不修改数据结构；D 不改业务代码。
5. **合并前自检**：每人提交前运行 `python -m unittest discover -s tests -v`。
6. **占位规范**：未完成能力必须 `raise NotImplementedError("待成员 X 实现：...")`，并保留 TODO 说明预期功能，禁止用 `pass` 或空实现冒充完成。
7. **工作区隔离**：任务工作区由成员 B 维护在 `benchmark/workspaces/<task_id>/`；成员 C 实验时必须使用独立副本或可靠重置，禁止直接修改初始工作区，保证实验可复现、互不污染。
