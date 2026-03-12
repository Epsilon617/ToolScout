# ToolScout

ToolScout retrieves the most relevant tools for LLM agents from large tool libraries before the model sees them.

When an agent has hundreds of tools, placing the whole tool catalog in the prompt leads to:

- token explosion
- lower tool selection accuracy
- higher latency

ToolScout solves this by embedding tool metadata, indexing the vectors, and retrieving the top-k tools for a user query.

## Features

- Tool registry with `name`, `description`, `args`, optional tags, examples, and executable handlers
- Tool encoder powered by `sentence-transformers`
- Vector search with FAISS
- End-to-end retriever for `query -> top-k tools`
- Tool executor for validated handler calls
- Simple agent demo with a mock LLM planner
- OpenAI demo with retrieved-tool prompting
- Benchmark scripts for retrieval accuracy, token cost reduction, and scaling to 1000 tools

ToolScout prefers `sentence-transformers` and FAISS when those packages are installed. For lightweight local runs, it also includes a deterministic NumPy-based fallback encoder/index so the examples and benchmarks can still run in constrained environments.

## Project Layout

```text
ToolScout/
├── benchmark/
│   ├── retrieval_accuracy.py
│   ├── scaling_test.py
│   └── token_cost_eval.py
├── datasets/
│   ├── generate_tools.py
│   ├── tools_1000.json
│   └── synthetic_tools.json
├── examples/
│   ├── retrieval_demo.py
│   ├── openai_agent_demo.py
│   └── simple_agent.py
├── toolscout/
│   ├── cli.py
│   ├── encoder/
│   │   └── tool_encoder.py
│   ├── executor/
│   │   └── tool_executor.py
│   ├── index/
│   │   └── tool_index.py
│   ├── registry/
│   │   └── tool_registry.py
│   └── retriever/
│       └── tool_retriever.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## Installation

```bash
cd /mnt/iusers01/fatpou01/compsci01/r90629yl/src/ToolScout
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you want GPU-backed FAISS, replace `faiss-cpu` with an appropriate FAISS build for your environment.

## Quickstart

```python
from toolscout import ToolRegistry, ToolRetriever

registry = ToolRegistry()
registry.register_tool(
    name="weather_api",
    description="Get the weather forecast for a city",
    args=["city"],
)
registry.register_tool(
    name="news_search",
    description="Fetch recent news coverage for a topic or company",
    args=["topic"],
)

retriever = ToolRetriever(registry=registry)
retriever.fit()

results = retriever.search("latest news about Nvidia", top_k=2)
for result in results:
    print(result.rank, result.tool.name, round(result.score, 4))
```

Generate the large synthetic catalog:

```bash
python datasets/generate_tools.py
```

## Core API

### Register tools

```python
from toolscout import ToolRegistry

registry = ToolRegistry()
registry.register_tool(
    name="weather_api",
    description="Get weather for a city",
    args=["city"],
)
```

### Retrieve tools

```python
from toolscout import ToolRetriever

retriever = ToolRetriever(registry=registry)
retriever.fit()
hits = retriever.search("latest news about Nvidia", top_k=5)
```

### Execute tools

```python
from toolscout import ToolExecutor

executor = ToolExecutor(registry)
result = executor.execute("weather_api", {"city": "London"})
```

## Example Scripts

Run the simple end-to-end demo:

```bash
python examples/simple_agent.py --query "latest news about Nvidia"
```

Run the large-scale retrieval demo:

```bash
python examples/retrieval_demo.py
```

Search from the CLI:

```bash
toolscout search "latest Nvidia news"
```

Run the OpenAI-backed demo:

```bash
export OPENAI_API_KEY=your_key_here
python examples/openai_agent_demo.py --model gpt-4.1-mini --query "latest news about Nvidia"
```

The OpenAI demo retrieves the top-k candidate tools first, then asks the model to choose one tool and fill its arguments using only those candidates.

## Large-Scale Tool Retrieval Demo

The repository includes a generated 1000-tool catalog spanning weather, finance, news, maps, math, code, translation, search, social, and calendar tools.

Example:

```text
Query:
"latest Nvidia news"

Retrieved tools:
1 news_api
2 web_search
3 finance_api
```

## Benchmarks

Retrieval accuracy on the synthetic dataset:

```bash
python benchmark/retrieval_accuracy.py --top-k 5
```

Prompt/token cost reduction:

```bash
python benchmark/token_cost_eval.py --top-k 5
```

Scaling behavior up to 1000 tools:

```bash
python benchmark/scaling_test.py --runs 50
```

Example scaling results:

| tools | latency | token reduction |
|------|--------|----------------|
| 10 | 0.035 ms | 50.1% |
| 100 | 0.043 ms | 94.8% |
| 1000 | 0.074 ms | 99.5% |

These example numbers were measured on the built-in keyword encoder plus NumPy index fallback in this repository's current environment. The same benchmark also reports `recall@5`, which reached `1.000` at 10, 100, and 1000 tools on the generated benchmark queries.

## Synthetic Dataset

`datasets/synthetic_tools.json` contains:

- a synthetic tool catalog
- example queries
- expected tools for each query

This is enough to exercise retrieval quality and scaling without external services.

`datasets/tools_1000.json` is the larger generated catalog used by the new retrieval demo, CLI, and multi-scale benchmark. It contains 1000 synthetic tools and benchmark queries covering the ten supported categories.

## Notes

- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- Tool metadata embeddings concatenate the tool name, description, arguments, and examples into one embedding string.
- The included fallback backend is intended for development and smoke tests, not final production accuracy measurements.

## License

MIT
