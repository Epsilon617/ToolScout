# ToolScout

> ToolScout: Execution-aware tool routing for LLM agents.

[简体中文](README.zh-CN.md)

![ToolScout architecture overview](assets/toolscout-architecture-overview.png)

ToolScout helps LLM agents work with hundreds or thousands of tools without putting the entire tool catalog into every prompt. It retrieves and reranks the most relevant tools using semantic similarity, skill routing, tool dependencies, and execution feedback, which reduces prompt size, latency, and bad tool choices.

## Why This Exists

LLM agents often degrade when the prompt contains a very large tool list:

- more tokens
- slower responses
- worse tool selection
- more semantically plausible but unreliable tool picks

ToolScout addresses this by narrowing the tool set first, then letting the agent reason over a smaller and higher-quality candidate list.

## When to Use ToolScout

- **Enterprise Tools**: when your internal API catalog grows to hundreds of endpoints or more.
- **High-Stakes Tasks**: when picking the wrong tool, such as `delete_user` instead of `get_user`, is unacceptable.
- **Cost-Sensitive Apps**: when you want smaller or cheaper models to operate over a compact tool shortlist instead of a full catalog.

## Quick Demo

From the repository root:

```bash
pip install -e .
toolscout search "latest Nvidia news"
```

Expected output:

```text
Top tools:
1 news_search
2 web_search
3 finance_api
```

More quick commands:

```bash
toolscout search "weather in Tokyo" --execution-aware
toolscout skill "latest Nvidia news"
python examples/retrieval_demo.py
```

## Architecture

```text
User Query
   |
   v
Skill Routing
   |
   v
Tool Retrieval
   |
   v
Execution-aware Reranking
   |
   v
Selected Tools
   |
   v
LLM Agent
   |
   v
Tool Execution
   |
   v
Execution Feedback
```

What each step does:

- `Skill Routing`: maps the query to a higher-level workflow such as `news_research`.
- `Tool Retrieval`: pulls the top semantic candidates from the tool catalog.
- `Execution-aware Reranking`: prefers tools that are both relevant and historically reliable.
- `Selected Tools`: keeps the prompt small by passing only top candidates downstream.
- `LLM Agent`: plans with a constrained tool set instead of the full library.
- `Tool Execution`: runs the chosen tool or a simulator-backed equivalent.
- `Execution Feedback`: stores success and latency so future ranking improves.

## Key Features

- Semantic tool retrieval over large tool libraries
- Skill-based routing before low-level tool search
- Tool dependency graph with graph-aware expansion
- Execution-aware reranking using success rate and latency
- MCP-compatible tool loading
- Offline evaluation framework with simulator-backed execution
- NumPy fallback path for local runs without FAISS or sentence-transformers

## Why ToolScout?

Traditional tool retrieval usually ranks tools by semantic similarity alone. That is useful, but tool descriptions do not capture real-world reliability: two tools can look equally relevant while one fails more often or runs much slower.

ToolScout adds execution-aware routing. It combines semantic matching with historical execution outcomes so the system can favor tools that actually work in practice, not just tools that read well in metadata.

## Installation

```bash
git clone https://github.com/Epsilon617/toolscout.git
cd toolscout
pip install -e .
```

Optional dependencies:

- `faiss-cpu` for vector indexing
- `sentence-transformers` for dense embeddings
- `openai` only for the optional OpenAI demo or judge mode

If those are unavailable, ToolScout falls back to its deterministic local backend so examples and benchmarks still run offline.

## Core Commands

```bash
toolscout search "latest Nvidia news"
toolscout search "weather in Tokyo" --execution-aware
toolscout search "latest Nvidia news" --graph-aware
toolscout skill "latest Nvidia news"
toolscout simulate-runs --queries datasets/tool_queries.json --passes 5 --clear
toolscout feedback-stats
toolscout load-mcp path/to/mcp_tools.json
```

## Evaluation

ToolScout evaluates tool routing at multiple layers.

### Evaluation Layers

- `Retrieval Quality`: `Recall@k` and `Precision@1`
- `Decision Quality`: hard-negative benchmark with semantically confusing distractors
- `Execution Success`: end-to-end `Pass@1`, `Pass@k`, and simulated execution outcomes
- `Robustness`: schema mutation, tool sprawl scaling, and cross-lingual queries

Example offline summary from `python benchmark/generate_eval_report.py --top-k 5`:

| Method | Precision@1 | Pass@1 (Success) | Avg Latency |
| ------ | ----------: | ----------------: | ----------: |
| Random Retrieval | 0.417 | 0.000 | 133.289 ms |
| Semantic Retrieval | 0.833 | 0.750 | 117.555 ms |
| **ToolScout** | **1.000** | **0.917** | **111.454 ms** |

In this offline benchmark snapshot, ToolScout improves both task success and latency over pure semantic retrieval.

All evaluation scripts run offline by default:

- `python benchmark/hard_negative_eval.py --top-k 4`
- `python benchmark/judge_eval.py --judge-mode mock --top-k 5`
- `python benchmark/e2e_eval.py --top-k 5`
- `python benchmark/robustness_eval.py --top-k 5`
- `python benchmark/generate_eval_report.py --top-k 5`

`judge_eval.py` supports `openai` mode as an optional external judge, but the default offline path uses `mock` mode. End-to-end evaluation uses `ToolExecutionSimulator` when real APIs are unavailable.

## Dataset Summary

Evaluation dataset:

- 12 primary benchmark queries
- 1000-tool synthetic catalog
- 12 hard-negative cases
- 1 explicit multi-step task: `news_search -> news_summary`
- 6 Chinese cross-lingual robustness queries

The datasets are intentionally lightweight so the full evaluation stack can run on a standard local Python environment.

Supported tool categories:

- `weather`
- `finance`
- `news`
- `maps`
- `math`
- `code`
- `translation`
- `search`
- `social`
- `calendar`

## Evaluation Flow

```text
Query
   |
   v
ToolScout Retrieval
   |
   v
Top-K Tools
   |
   v
Execution Simulator
   |
   v
Success / Failure Logging
   |
   v
Execution-aware Ranking Update
```

This loop lets ToolScout move beyond one-shot retrieval metrics and measure whether the selected tool actually succeeds.

## Project Layout

```text
toolscout/
  cli.py
  execution_feedback.py
  mcp_adapter.py
  skill_registry.py
  skill_retriever.py
  tool_graph.py
  tool_simulator.py
  encoder/
  executor/
  index/
  registry/
  retriever/
examples/
benchmark/
datasets/
data/
```

## Roadmap

- Larger and more diverse benchmark datasets
- Better learning from execution feedback
- Real API execution benchmarks
- Deeper integration with agent runtimes such as LangGraph and MCP ecosystems

## License

MIT
