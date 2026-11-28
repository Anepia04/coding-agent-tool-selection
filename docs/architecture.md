# 架构与接口契约

本文档描述当前代码的真实架构、模块边界、统一数据结构与跨模块接口。任何跨模块接口变更都必须同步更新本文档与对应测试。

---

## 1. 系统总体架构

```text
┌──────────────────────────────────────────────────────────────┐
│ 数据层  benchmark/ tasks.json · expected_tools.json · workspaces/ │
└──────────────────────────────┬───────────────────────────────┘
                               │ 任务定义 / 初始代码
┌──────────────────────────────▼───────────────────────────────┐
│ 实验层  experiments/                                          │
│   runner.py 编排：工作副本 → 三模式 → 指标 → CSV              │
│   run_baseline / run_fixed_topk / run_adaptive_topk           │
│   evaluate.py 六项指标 · workspace.py 副本 · schemas.py 结构   │
└──────────────────────────────┬───────────────────────────────┘
                               │ AgentController.run(task, mode, max_steps)
┌──────────────────────────────▼───────────────────────────────┐
│ 编排层  agent/                                                │
│   TaskAnalyzer ──> TaskInfo                                   │
│   AgentController：选工具 → get_schemas → chat → 执行 → 回填   │
│   LLMClient：OpenAI 兼容接口封装                               │
└───────────────┬──────────────────────────────┬───────────────┘
                │ 检索 / 选择                   │ schema / 执行
┌───────────────▼──────────────┐  ┌────────────▼───────────────┐
│ 能力层  retrieval/            │  │ 能力层  tools/              │
│   TFIDFRetriever              │  │   ToolRegistry              │
│   AdaptiveSelector            │  │   definitions / implementations │
└───────────────┬──────────────┘  └────────────┬───────────────┘
                │ 配置 / 路径 / 环境变量        │ workspace 安全边界
┌───────────────▼──────────────────────────────▼───────────────┐
│ 基础设施  config.py（.env、路径、K、模式、workspace）          │
└──────────────────────────────────────────────────────────────┘
```

依赖方向自上而下：实验层依赖编排层，编排层依赖能力层与基础设施；`tools/` 与 `retrieval/` 不反向依赖 `agent/` 或 `experiments/`。

---

## 2. 模块职责与边界

| 文件 | 职责 | 边界 |
|---|---|---|
| `config.py` | 路径常量、枚举、分组配置、`.env` 加载、全局单例 | 不含业务逻辑；敏感信息只读环境变量 |
| `agent/llm_client.py` | 消息发送、工具透传、文本 / tool_calls / Token 解析 | 不执行工具、不做任务分析 |
| `agent/task_analyzer.py` | Prompt 约束 + JSON 解析 → `TaskInfo` | 不检索、不选择、不执行 |
| `agent/controller.py` | 任务解析、工具选择、多轮循环、轨迹记录 | 不实现 TF-IDF / Top-K / 工具业务 |
| `retrieval/tfidf_retriever.py` | 中文友好分词、TF-IDF 索引、Cosine 排序 | 不构造查询、不决定最终 K |
| `retrieval/query_utils.py` | 分析阶段的查询构造：否定约束清洗、正向关键词拼接 | 不检索、不选择工具 |
| `retrieval/adaptive_selector.py` | fixed / adaptive 两种选择策略 | 不检索打分 |
| `tools/definitions.py` | `ToolDefinition` 与 16 个工具元数据 | 不实现工具函数 |
| `tools/implementations.py` | 工具函数、路径围栏、命令白名单、输出截断 | 不决定暴露哪些工具 |
| `tools/registry.py` | 注册、查询、schema 生成、执行分发 | 参数业务校验由实现侧负责（当前为 TODO） |
| `experiments/baseline.py` 等 | 三个实验入口，只决定模式与 K，执行逻辑全部交给 Controller | 不实现 Agent 循环 |
| `experiments/evaluate.py` | 六项指标与任务成功判定 | 不执行 Agent、不修改工作区 |
| `experiments/workspace.py` | 工作副本复制 / 重置、改动快照、副本内测试执行 | 不决定成功标准 |
| `experiments/runner.py` | 对照实验编排、结果聚合与 CSV 落盘、CLI | 不绕过 Controller 调用工具 |
| `experiments/schemas.py` | `ExperimentConfig` / `ExperimentResult` / CSV 列 | 结构由 A 维护，C 使用 |
| `benchmark/` | 任务、工作区、Expected Tools | 当前 3 条占位任务 |
| `scripts/` | 检索诊断、任务预标注、真实 API 联调 | 手动执行，不进入 unittest |

---

## 3. 数据流

### 3.1 任务输入

`AgentController.run(task, mode, max_steps)` 支持三种任务输入：

```text
TaskInfo
  -> 直接使用

完整 benchmark 映射
  （含 task_id / description / task_type / language / complexity，
    以及预标注的 retrieval_query / negative_constraints）
  -> TaskInfo.from_dict(task)

自然语言字符串或不完整映射
  -> TaskAnalyzer.analyze(description, task_id) -> TaskInfo
```

预标注字段在实验前由一次性脚本生成并冻结；正式实验中三种模式都走 `TaskInfo.from_dict`，不会再调用 LLM 分析，避免任务语义漂移。若某个 `TaskInfo` 缺少 `retrieval_query`，controller 会在挑选工具前调用一次 TaskAnalyzer 补齐，保证检索器拿到的查询始终非空。

完整映射直接转换的目的是：三种实验模式共享同一份任务语义，避免每种模式重新调用 LLM 造成分析结果漂移。

### 3.2 完整执行链路

```text
task
  -> TaskInfo
  -> _select_tools(task_info, mode)
       baseline      : ToolRegistry.tool_names
       fixed_topk    : TFIDFRetriever.retrieve(task_info)
                       -> AdaptiveSelector.select(..., strategy="fixed")
       adaptive_topk : TFIDFRetriever.retrieve(task_info)
                       -> AdaptiveSelector.select(..., complexity=..., strategy="adaptive")
  -> ToolRegistry.get_schemas(selected_tools)
  -> messages = [system, user(description)]
  -> 循环（最多 max_steps 轮）：
       LLMClient.chat(messages, tools=schemas)
       -> 无 tool_calls：termination_reason="completed"，结束
       -> 有 tool_calls：
            messages += assistant(tool_calls)
            逐个 ToolRegistry.execute_tool(name, arguments)
            messages += tool(tool_call_id, name, result)
            继续下一轮
  -> AgentRunResult
```

### 3.3 会话消息格式

工具调用回填遵循 OpenAI 规范：

```text
assistant:
  {"role": "assistant", "content": ..., "tool_calls": [
      {"id": ..., "type": "function",
       "function": {"name": ..., "arguments": "<JSON 字符串>"}}]}

tool:
  {"role": "tool", "tool_call_id": ..., "name": ..., "content": "<文本结果>"}
```

一个响应中的多个 ToolCall 全部执行，并逐条按 `tool_call_id` 关联回填。

---

## 4. 统一数据结构

### 4.1 TaskInfo

定义于 `agent/task_analyzer.py`。

```text
task_id              str   任务唯一标识，与 benchmark tasks.json 对应
description          str   自然语言任务描述（发给模型；检索查询由分析阶段单独构造）
task_type            str   bug_fix / feature_add / refactor / test_write / doc
language             str   主要编程语言，未知为 unknown
retrieval_query      str   面向检索器的正向关键词；禁止否定词与工具名，缺失时由 controller 调 TaskAnalyzer 补齐
negative_constraints list[str]  任务明确禁止的操作，供评价 / 过滤 / 提示使用
complexity           str   simple / medium / complex（构造时校验）
```

提供 `to_dict()` / `from_dict()`。`retrieval_query` 与 `negative_constraints` 由预标注流程写入 `tasks.json`，实验运行时直接读取，不调用 LLM 重新分析。

### 4.2 ToolDefinition

定义于 `tools/definitions.py`。

```text
name          str   全局唯一工具名
description   str   面向模型的功能描述（进入 schema，有 Token 成本）
category      str   分类，见 config.TOOL_CATEGORIES
parameters    dict  JSON Schema（至少含 type）
function_name str   实现函数名，为空时默认等于 name
search_hints  list[str]  面向检索器的场景词 / 同义词（不进入 schema，零 Token 成本）
```

**描述职责划分**：`description` 只给模型看；`search_hints` 只给检索器看。TF-IDF 文档由 `name + category + search_hints` 构建；旧定义没有 hints 时才回退到 `description`。

### 4.3 RetrievedTool

定义于 `retrieval/tfidf_retriever.py`。

```text
tool_name  str    工具名
score      float  余弦相似度（越大越相关）
rank       int    排名，从 1 开始
```

### 4.4 AgentStep

定义于 `agent/controller.py`。

```text
step_index      int   步骤序号（从 0 开始）
step_type       str   "assistant" | "tool"
content         str   模型文本（assistant）
tool_name       str   工具名（tool）
tool_arguments  dict  工具参数
tool_result     Any   工具返回值
error_message   str   该步骤错误信息
```

### 4.5 AgentRunResult

定义于 `agent/controller.py`。

```text
task_id             str
mode                str   baseline / fixed_topk / adaptive_topk
task_success        bool  Agent 流程是否无错误（不代表任务最终正确性）
selected_tools      list[str]  提供给模型的工具
steps               list[AgentStep]
input_tokens        int
output_tokens       int
total_tokens        int
execution_time      float  秒
tool_call_count     int
error_message       str | None
termination_reason  str | None  completed / max_steps / error
```

### 4.6 ExperimentConfig / ExperimentResult

定义于 `experiments/schemas.py`。

```text
ExperimentConfig:
  mode: str, task_id: str, top_k: int | None, max_steps: int

ExperimentResult:
  task_id, experiment_mode, task_success,
  input_tokens, output_tokens, total_tokens,
  execution_time, tool_call_count, selected_tools, error_message
```

`RESULT_CSV_FIELDS` 与 `ExperimentResult` 字段顺序一致，用于 `results/raw_results.csv`。
落盘时 `selected_tools` 序列化为 JSON 数组字符串（读取时 `json.loads` 还原），`execution_time` 保留 4 位小数。

Recall / Precision 与 `termination_reason` 不在 `ExperimentResult` 中（raw 列契约由 A 冻结），
由 `experiments/runner.py` 的 `RunRecord` 承载，只进入 `summary.csv` 与日志。

`summary.csv` 按实验模式聚合，列顺序见 `runner.SUMMARY_CSV_FIELDS`：

```text
experiment_mode, task_count, success_count, success_rate,
avg_input_tokens, avg_output_tokens, avg_total_tokens,
avg_execution_time, avg_tool_call_count, avg_selected_tools,
avg_tool_recall, avg_tool_precision
```

### 4.7 跨模块契约

```text
AgentRunResult（运行期轨迹）
  -> experiments.evaluate 抽取：task_success / tokens / 耗时 / 调用次数 / selected_tools
  -> ExperimentResult（可落盘结果）+ RunRecord（Recall / Precision / 终止原因）
  -> results/raw_results.csv（逐次运行）
  -> results/summary.csv（按模式聚合）
```

`AgentRunResult.task_success` 与最终任务成功是两个概念：前者只看 Agent 流程是否报错，后者由 `experiments/evaluate.py` 结合工作副本状态（改动 + 测试结果）判定。

---

## 5. 接口定义

### 5.1 LLMClient

```python
class LLMClient:
    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None, **options) -> None
    def chat(self, messages: Sequence[Message],
             tools: Sequence[dict] | None = None, **kwargs) -> LLMResponse
```

- Key 优先级：显式参数 → `LLM_API_KEY` → `OPENAI_API_KEY`；
- `LLMResponse`：`content / tool_calls / input_tokens / output_tokens / total_tokens / raw`；
- 统一异常 `LLMClientError`；缺少 Key 抛 `ValueError`。

### 5.2 TaskAnalyzer

```python
class TaskAnalyzer:
    def __init__(self, llm_client: LLMClient | None = None) -> None
    def analyze(self, task_description: str, task_id: str | None = None) -> TaskInfo
```

异常：`TaskAnalysisError`（调用失败、非 JSON、字段缺失或枚举非法）。输出除基础字段外，还包含 `retrieval_query`（正向检索关键词）与 `negative_constraints`（禁止操作），供预标注流程写入 `tasks.json`。

### 5.3 TFIDFRetriever

```python
class TFIDFRetriever:
    def fit(self, tool_definitions: Sequence[ToolDefinition]) -> TFIDFRetriever
    def retrieve(self, task_info: TaskInfo | str,
                 top_k: int | None = None) -> list[RetrievedTool]
```

- 分词：英文标识符整体 + 下划线子词；中文连续片段输出单字 + 相邻双字；
- 只消费已构造查询：`TaskInfo` 必须包含非空 `retrieval_query`，为空直接抛 `ValueError`；检索器不读 description，也不做否定清洗；
- 查询词由分析阶段构造（`retrieval/query_utils.py`）：清洗否定约束后作为检索查询；
- 排序稳定，同分保持注册顺序；
- 未 `fit` 调用 `retrieve` 抛 `RuntimeError`；空工具库 / 空词表抛 `ValueError`。

否定清洗是分析阶段的规则启发式，不处理复杂隐含否定；若 `TaskInfo` 缺少 `retrieval_query`，由 `AgentController` 调用一次 TaskAnalyzer 补齐，而不是在检索器里静默回退。

### 5.4 AdaptiveSelector

```python
class AdaptiveSelector:
    def __init__(self, config: SelectionConfig | dict | None = None) -> None
    def select(self, ranked_tools: Sequence[RetrievedTool],
               complexity: str | None = None,
               strategy: str = "fixed") -> list[str]
```

- `fixed`：取前 `config.top_k`；
- `adaptive`：`simple→3 / medium→5 / complex→7`，再用 `[min_k, max_k]` 夹取；
- 实际数量不超过候选工具数；空列表返回 `[]`。

### 5.5 ToolRegistry

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

`get_schemas(None)` 返回全部工具；schema 形如：

```json
{"type": "function",
 "function": {"name": "...", "description": "...", "parameters": {...}}}
```

### 5.6 AgentController

```python
class AgentController:
    def __init__(self, llm_client, task_analyzer, tool_registry,
                 retriever=None, selector=None, default_max_steps=8) -> None
    def run(self, task: TaskInfo | Mapping | str,
            mode: str = "baseline",
            max_steps: int | None = None) -> AgentRunResult
```

- `baseline` 不需要 retriever / selector；
- `fixed_topk` / `adaptive_topk` 缺少依赖时在调用模型前抛 `ValueError`；
- 工具集合在整次任务中保持不变，schema 只获取一次。

### 5.7 experiments

三个实验入口只决定「模式与 K」，执行逻辑全部由 `AgentController` 承担：

```python
run_baseline(agent, task) -> AgentRunResult
run_fixed_topk(agent, task, top_k=5) -> AgentRunResult
run_adaptive_topk(agent, task) -> AgentRunResult
```

- `run_fixed_topk`：临时把 `agent.selector` 换成 `top_k` 生效的 `AdaptiveSelector`，运行结束（含异常）后恢复原对象，避免污染同一 agent 上的其他实验组；`top_k <= 0` 抛 `ValueError`；
- `run_adaptive_topk`：不覆盖 K，实际 K 即 `len(result.selected_tools)`。

指标接口（前四项同时接受 `AgentRunResult` 与 `ExperimentResult`）：

```python
evaluate_task_success(result, task=None) -> bool
calculate_token_usage(result) -> dict[str, int]   # input / output / total
calculate_execution_time(result) -> float
calculate_tool_call_count(result) -> int
calculate_tool_recall(selected_tools, expected_tools) -> float
calculate_tool_precision(selected_tools, expected_tools) -> float
```

`evaluate_task_success` 依次短路判定：

1. 流程失败（`error_message` 非空 / `termination_reason == "error"` / `task_success` 为假）→ False；
2. 未提供 `task` → 退化为流程级结论；
3. 提供 `task` → 在待评价工作区（`task["workspace_dir"]`，缺省回退 `project_path`）上核验：
   相对初始工作区**确有改动**，且**全部测试通过**（`run_tests` 的 `ok` 为真且 `no_tests_found` 为假）。

`success_criteria` 中超出测试覆盖的语义要求（如「函数复杂度下降」）需人工复核。
Expected Tools 只用于 Recall / Precision，**不参与成功判定**。

空集合约定：`recall` 的 `expected_tools` 为空 → 1.0（无必需工具）；`precision` 的 `selected_tools` 为空 → 0.0（未提供工具）。

编排与落盘接口（`experiments/runner.py`）：

```python
RunnerConfig(modes, top_k, max_steps, repeats, work_root, keep_workspaces,
             tasks_file, expected_tools_file, raw_results_file, summary_file)
run_experiment(config=None, agent=None, tasks=None, expected_tools=None) -> list[RunRecord]
build_agent(settings=None, llm_client=None) -> AgentController
load_tasks(path=None) -> list[dict]
load_expected_tools(path=None) -> dict[str, dict]
write_raw_results(records, path=None) -> Path
write_summary(records, path=None) -> Path
```

命令行：`python -m experiments.runner [--modes ...] [--task-id ...] [--top-k N] [--max-steps N] [--repeats N] [--keep-workspaces]`，
退出码 0 成功 / 2 缺少 API Key / 3 无可运行任务。

### 5.8 工具实现

`tools/implementations.py` 提供 16 个函数：`read_file / write_file / list_files / file_info / edit_file / apply_patch / delete_path / move_path / search_code / find_symbol / list_todos / run_tests / check_syntax / git_status / git_diff / run_command`。

实验侧可通过 `configure_workspace(path)` 在同一进程内切换工具根目录（不同任务工作副本）。

---

## 6. 三种实验模式的执行流程

三种模式共用同一入口、同一循环、同一结果结构，唯一差异是工具选择：

### baseline

```text
selected = tool_registry.tool_names
```

不做任何检索与筛选，schema 成本最高，作为对照组。

### fixed_topk

```text
ranked = retriever.retrieve(task_info)          # 全部工具已排序
selected = selector.select(ranked, strategy="fixed")
```

固定 K（默认 5，受 `SelectionConfig.top_k` 控制），不随任务复杂度变化。

### adaptive_topk

```text
ranked = retriever.retrieve(task_info)
selected = selector.select(ranked,
                           complexity=task_info.complexity,
                           strategy="adaptive")
```

K 由任务复杂度决定（simple→3 / medium→5 / complex→7），并夹在配置区间 `[min_k, max_k]` 内。

### 共同部分

```text
get_schemas(selected) -> chat -> [执行 tool_calls -> 回填 -> chat] -> AgentRunResult
```

selected_tools、schema、循环轮数上限与结果记录方式完全一致，保证对照公平。

### 编排流程（experiments/runner.py）

```text
run_experiment(config, agent, tasks, expected_tools)
  -> load_tasks / load_expected_tools（未显式传入时）
  -> 逐任务：
       workspace_ready：初始工作区只有 .gitkeep 等占位时跳过并告警（不计入成功率）
       _freeze_task：缺 retrieval_query 时只调用一次 TaskAnalyzer 补齐，
                     三种模式与多次重复共用同一份任务语义
  -> 逐「模式 × 重复序号」：
       prepare_workspace：复制 benchmark/workspaces/<task_id> 到
                          .experiment_workspaces/<task_id>/<mode>_r<n>（已存在先删除）
       configure_workspace(副本)
         -> run_baseline / run_fixed_topk / run_adaptive_topk
       configure_workspace(None)
       evaluate_task_success(run_result, task + workspace_dir=副本)
       六项指标 -> ExperimentResult + RunRecord
       默认删除副本（keep_workspaces=True 时保留供人工复核）
  -> write_raw_results（逐次运行） / write_summary（按模式聚合）
```

`configure_workspace` 是**进程级**开关，因此实验串行执行；每次运行与每次评价都在
`try/finally` 中取消覆盖（回到 config 配置），避免副本路径泄漏到后续任务。

---

## 7. 执行与终止语义

| 情况 | task_success | termination_reason | 说明 |
|---|---|---|---|
| 模型返回纯文本 | True | `completed` | 正常结束 |
| 达到 max_steps 仍未结束 | True | `max_steps` | 非流程错误 |
| 工具执行失败 | 仍可为 True | 由后续轮次决定 | 错误以 tool 消息回填，模型可修复 |
| 分析 / 检索 / 模型 / 控制器异常 | False | `error` | `error_message` 记录原因 |

工具失败只写入对应 `AgentStep.error_message`，不直接判定任务失败；最终正确性由评价层判定。

---

## 8. 安全边界

由 `tools/implementations.py` 实现（成员 B）：

1. **路径围栏**：所有路径必须是相对路径，解析后必须落在 `AGENT_WORKSPACE_DIR` 内；拒绝绝对路径、盘符、`..`、越界符号链接；
2. **命令执行默认关闭**：`AGENT_ALLOW_SHELL=false` 时 `run_command` 抛 `PermissionError`；开启后仅允许白名单程序，且不经过 shell 解析；
3. **输出截断**：工具输出限制在 `MAX_TOOL_OUTPUT_CHARS`，避免撑爆上下文；
4. **敏感环境隔离**：子进程不继承名称含 `API_KEY / TOKEN / SECRET / PASSWORD` 等变量；
5. **工作副本**：实验层（`experiments/workspace.py`）把 `benchmark/workspaces/<task_id>` 复制到 `.experiment_workspaces/<task_id>/<mode>_r<n>` 后运行，避免组间污染；该目录为运行时产物，不进入版本库。

---

## 9. 四人协作关系

| 成员 | 负责范围 | 对外契约 |
|---|---|---|
| A | `agent/`、`retrieval/`、`config.py`、`scripts/`（联调）、共享数据结构协调 | `LLMClient`、`TaskAnalyzer`、`AgentController`、`TFIDFRetriever`、`AdaptiveSelector`、`AgentRunResult` |
| B | `tools/`、`benchmark/` | `ToolDefinition`（含 description / search_hints）、`ToolRegistry`、工具实现、任务工作区 |
| C | `experiments/` | 三个 `run_*` 入口、六项指标、实验结果字段消费 |
| D | `results/`、`docs/`、`demo/`、`presentation/` | 结果文件、图表、文档与展示 |

变更规则：

1. 跨模块接口变更必须同步修改本文档与相关测试；
2. `description` / `search_hints` 的内容由 B 维护，检索逻辑由 A 维护；新增工具必须提供 hints；
3. `benchmark/tasks.json` 的 `project_path` 必须指向 `benchmark/workspaces/<task_id>`；
4. 实验层只消费 `AgentRunResult`，不得绕过 `AgentController` 直接调用工具；
5. 真实 API Key、`.env`、`.devlogs/` 不进入版本库；
6. 任务预标注由 B 执行：`retrieval_query` 只能从 `description` 生成（不得参考 `expected_tools`、不得含否定词或工具名），否定要求写入 `negative_constraints`；默认只补缺失字段，人工复核后提交，描述变更时需重新生成。

---

## 10. 测试与联调现状

| 项目 | 状态 |
|---|---|
| 单元测试 | 168 项，`python -m unittest discover -s tests -v` 全部通过 |
| 覆盖范围 | LLMClient（Mock）、TaskAnalyzer、TF-IDF、AdaptiveSelector、AgentController（单轮 / 多轮 / 异常）、Benchmark 结构、工具注册与 schema、实验入口 / 六项指标 / 工作副本 / 编排落盘（`tests/test_experiments.py`） |
| 真实 Qwen4-Max | 已完成文本调用与三种模式 Function Calling 冒烟（`scripts/test_qwen_api.py`、`scripts/test_qwen_agent.py`） |
| 检索诊断 | `scripts/probe_retrieval.py`：排名、交集词、recall@K、跨工具噪声词 |
| 实验入口 | 已实现（成员 C）：`run_baseline / run_fixed_topk / run_adaptive_topk` |
| 指标计算 | 已实现（成员 C）：六项指标 + 基于工作副本的成功判定 |
| 正式对照实验 | 待运行：入口为 `python -m experiments.runner`，阻塞于 Benchmark 任务扩充（B）与 `LLM_API_KEY` |

真实 API 脚本手动执行、不进入 unittest；运行环境需可访问模型服务。
