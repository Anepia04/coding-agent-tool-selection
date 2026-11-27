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
│   run_baseline / run_fixed_topk / run_adaptive_topk（待实现）  │
│   evaluate.py 指标接口（待实现） · schemas.py 结果结构         │
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
| `retrieval/tfidf_retriever.py` | 中文友好分词、TF-IDF 索引、Cosine 排序 | 不决定最终 K |
| `retrieval/adaptive_selector.py` | fixed / adaptive 两种选择策略 | 不检索打分 |
| `tools/definitions.py` | `ToolDefinition` 与 16 个工具元数据 | 不实现工具函数 |
| `tools/implementations.py` | 工具函数、路径围栏、命令白名单、输出截断 | 不决定暴露哪些工具 |
| `tools/registry.py` | 注册、查询、schema 生成、执行分发 | 参数业务校验由实现侧负责（当前为 TODO） |
| `experiments/*.py` | 三组实验入口与指标 | 当前为接口占位，未实现 |
| `experiments/schemas.py` | `ExperimentConfig` / `ExperimentResult` / CSV 列 | 结构由 A 维护，C 使用 |
| `benchmark/` | 任务、工作区、Expected Tools | 当前 3 条占位任务 |
| `scripts/` | 检索诊断、真实 API 联调 | 手动执行，不进入 unittest |

---

## 3. 数据流

### 3.1 任务输入

`AgentController.run(task, mode, max_steps)` 支持三种任务输入：

```text
TaskInfo
  -> 直接使用

完整 benchmark 映射
  （含 task_id / description / task_type / language / complexity）
  -> TaskInfo.from_dict(task)

自然语言字符串或不完整映射
  -> TaskAnalyzer.analyze(description, task_id) -> TaskInfo
```

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
task_id      str   任务唯一标识，与 benchmark tasks.json 对应
description  str   自然语言任务描述（检索主输入）
task_type    str   bug_fix / feature_add / refactor / test_write / doc
language     str   主要编程语言，未知为 unknown
intents      list[str]  操作意图关键词
complexity   str   simple / medium / complex（构造时校验）
```

提供 `to_dict()` / `from_dict()`。

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

### 4.7 跨模块契约

```text
AgentRunResult（运行期轨迹）
  -> experiments 抽取：task_success / tokens / 耗时 / 调用次数 / selected_tools
  -> ExperimentResult（可落盘结果）
  -> results/raw_results.csv
```

`AgentRunResult.task_success` 与最终任务成功是两个概念：前者只看 Agent 流程是否报错，后者由 `experiments/evaluate.py` 结合 `success_criteria` / 测试结果判定。

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

异常：`TaskAnalysisError`（调用失败、非 JSON、字段缺失或枚举非法）。

### 5.3 TFIDFRetriever

```python
class TFIDFRetriever:
    def fit(self, tool_definitions: Sequence[ToolDefinition]) -> TFIDFRetriever
    def retrieve(self, task_info: TaskInfo | str,
                 top_k: int | None = None) -> list[RetrievedTool]
```

- 分词：英文标识符整体 + 下划线子词；中文连续片段输出单字 + 相邻双字；
- 排序稳定，同分保持注册顺序；
- 未 `fit` 调用 `retrieve` 抛 `RuntimeError`；空工具库 / 空词表抛 `ValueError`。

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

当前为接口占位（`NotImplementedError`，待成员 C）：

```python
run_baseline(agent, task) -> AgentRunResult
run_fixed_topk(agent, task, top_k=5) -> AgentRunResult
run_adaptive_topk(agent, task) -> AgentRunResult

evaluate_task_success(result, task=None) -> bool
calculate_token_usage(result) -> dict[str, int]
calculate_execution_time(result) -> float
calculate_tool_call_count(result) -> int
calculate_tool_recall(selected_tools, expected_tools) -> float
calculate_tool_precision(selected_tools, expected_tools) -> float
```

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
5. **工作副本**：实验层负责把 `benchmark/workspaces/<task_id>` 复制到独立目录后运行，避免组间污染。

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
5. 真实 API Key、`.env`、`.devlogs/` 不进入版本库。

---

## 10. 测试与联调现状

| 项目 | 状态 |
|---|---|
| 单元测试 | 114 项，`python -m unittest discover -s tests -v` 全部通过 |
| 覆盖范围 | LLMClient（Mock）、TaskAnalyzer、TF-IDF、AdaptiveSelector、AgentController（单轮 / 多轮 / 异常）、Benchmark 结构、工具注册与 schema |
| 真实 Qwen4-Max | 已完成文本调用与三种模式 Function Calling 冒烟（`scripts/test_qwen_api.py`、`scripts/test_qwen_agent.py`） |
| 检索诊断 | `scripts/probe_retrieval.py`：排名、交集词、recall@K、跨工具噪声词 |
| 实验入口 | 未实现（成员 C） |
| 指标计算 | 未实现（成员 C） |

真实 API 脚本手动执行、不进入 unittest；运行环境需可访问模型服务。
