# Architecture and Interface Contract

## 1. Ownership boundaries

- Member A: `agent/`, `retrieval/`, and `config.py`. Owns the agent framework, task analysis, retrieval, selection, and cross-module interfaces.
- Member B: `tools/` and `benchmark/`. Owns tool definitions, implementations, workspaces, and expected-tool labels.
- Member C: `experiments/`. Owns experiment entry points, task acceptance, and metric calculation.
- Member D: `results/`, `docs/`, and `demo/`. Owns aggregation, visualization, and presentation material.

## 2. Agent data flow

```text
TaskInfo, benchmark mapping, or natural-language string
  -> TaskInfo.from_dict or TaskAnalyzer.analyze
  -> TFIDFRetriever.retrieve(TaskInfo)
  -> AdaptiveSelector.select(...)
  -> ToolRegistry.get_schemas(selected_tools)
  -> LLMClient.chat(messages, tools)
  -> ToolRegistry.execute_tool(name, arguments)
  -> append tool message and continue the loop
  -> AgentRunResult
```

A complete benchmark mapping is converted directly to `TaskInfo`. An incomplete mapping or a string is analyzed by `TaskAnalyzer`. All three experiment modes use `AgentController.run(task, mode, max_steps)`.

## 3. Core data structures

### TaskInfo

```text
task_id, description, task_type, language, intents, complexity
```

### ToolDefinition

```text
name            unique tool name
description     model-facing function description; included in the schema
category        tool category
parameters      JSON Schema for arguments
function_name   implementation function name
search_hints    retrieval phrases and synonyms; excluded from the schema
```

TF-IDF builds each tool document from `name`, `category`, and `search_hints`. Older definitions without hints temporarily fall back to `description`; new tools should provide hints.

### AgentRunResult

The result records:

```text
task_id, mode, task_success, selected_tools, steps,
input_tokens, output_tokens, total_tokens, execution_time,
tool_call_count, error_message, termination_reason
```

`task_success` describes whether the agent flow completed without a controller-level error. It is not the final benchmark acceptance result. Member C's evaluation layer owns task acceptance.

## 4. Loop and error behavior

The controller supports multiple assistant and tool rounds up to `max_steps`. Tool failures are recorded in `AgentStep.error_message` and returned to the model as a tool message so the model can recover or stop.

`termination_reason` values currently include:

- `completed`: the model returned a response without tool calls;
- `max_steps`: the loop reached its configured limit;
- `error`: task analysis, retrieval, model, or controller failure.

## 5. Experiment modes

- `baseline`: expose `ToolRegistry.tool_names`.
- `fixed_topk`: retrieve tools and select `SelectionConfig.top_k`.
- `adaptive_topk`: retrieve tools and select K from task complexity within the configured bounds.

The agent executes and records the run. The experiment layer consumes `AgentRunResult` and produces `ExperimentResult`.

## 6. Workspace and change rules

Tool implementations own workspace enforcement. The experiment runner owns copying or resetting benchmark workspaces. The controller does not modify benchmark source directories directly. Any cross-module interface change must update this document and its tests.
