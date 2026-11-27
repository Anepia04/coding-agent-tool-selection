# Task-Aware Dynamic Tool Selection for Coding Agents

A research prototype for task-aware tool selection in a lightweight Python coding agent.

## Current status

Member A builds and maintains the agent framework and retrieval layer:

- `agent/`: `LLMClient`, `TaskAnalyzer`, `AgentController`, and multi-turn tool calling.
- `retrieval/`: TF-IDF, cosine similarity, Fixed Top-K, and Adaptive Top-K.
- Tool retrieval uses `ToolDefinition.search_hints` first. `description` is reserved for model-facing schemas.
- `AgentRunResult` records selected tools, execution steps, token usage, duration, tool calls, errors, and termination reason.
- A complete benchmark mapping can be converted directly to `TaskInfo`, so the three modes share the same task analysis.

Member B builds and maintains the tool library and benchmark tasks in `tools/` and `benchmark/`. Member C writes and runs the experiments and implements the evaluation metrics in `experiments/`. Member D analyzes the results and prepares the charts, demo, and presentation materials.

## Experiment modes

| Mode | Tool exposure |
|---|---|
| `baseline` | Expose every registered tool. |
| `fixed_topk` | Rank tools with TF-IDF and expose a fixed number. |
| `adaptive_topk` | Rank tools with TF-IDF and choose K from task complexity. |

All modes use the same entry point:

```python
agent.run(task, mode="baseline")
agent.run(task, mode="fixed_topk")
agent.run(task, mode="adaptive_topk")
```

`task` may be a `TaskInfo`, a complete benchmark mapping, or a natural-language string. A complete mapping contains `task_id`, `description`, `task_type`, `language`, and `complexity`; it is converted directly to `TaskInfo`. A plain string is analyzed by `TaskAnalyzer`.

## Repository layout

- `agent/`: Member A, agent framework and model interface.
- `retrieval/`: Member A, tool retrieval and selection.
- `tools/`: Member B, tool definitions, implementations, and registry.
- `benchmark/`: Member B, tasks, workspaces, and expected-tool labels.
- `experiments/`: Member C, experiment entry points and evaluation.
- `results/`: experiment outputs and figures.
- `docs/`: architecture and interface contracts.
- `tests/`: development and regression tests.

## Setup and tests

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Provide model credentials through `.env` or environment variables. Never commit API keys.
