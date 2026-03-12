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
│   └── synthetic_tools.json
├── examples/
│   ├── openai_agent_demo.py
│   └── simple_agent.py
├── toolscout/
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

Run the OpenAI-backed demo:

```bash
export OPENAI_API_KEY=your_key_here
python examples/openai_agent_demo.py --model gpt-4.1-mini --query "latest news about Nvidia"
```

The OpenAI demo retrieves the top-k candidate tools first, then asks the model to choose one tool and fill its arguments using only those candidates.

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
python benchmark/scaling_test.py --target-tools 1000 --runs 50
```

## Synthetic Dataset

`datasets/synthetic_tools.json` contains:

- a synthetic tool catalog
- example queries
- expected tools for each query

This is enough to exercise retrieval quality and scaling without external services.

## Notes

- The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- Tool metadata embeddings combine the tool name, description, arguments, tags, and examples.
- The included fallback backend is intended for development and smoke tests, not final production accuracy measurements.

## License

MIT

