# ToolScout

> 面向 LLM Agent 的执行感知型工具路由器。

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[English](README.md)

![ToolScout architecture overview](assets/toolscout-architecture-overview.png)

当 LLM Agent 需要面对几百到几千个工具时，把整个工具目录塞进每一次 prompt 往往会增加延迟、浪费上下文，并让工具选择更不稳定。ToolScout 通过语义检索、skill 路由、依赖扩展和执行反馈重排，把真正相关的一小组工具交给 Agent。

## 快速体验

在仓库根目录执行：

```bash
pip install -e .
toolscout search "latest Nvidia news"
```

示例输出：

```text
Top tools:
1 news_search
2 web_search
3 finance_api
```

更多常用命令：

```bash
toolscout search "weather in Tokyo" --execution-aware
toolscout skill "latest Nvidia news"
python examples/retrieval_demo.py
```

## 为什么要做 ToolScout

ToolScout 面向的是 Agent 系统从 demo 走向真实复杂度时出现的几个典型问题：

- `工具爆炸`：内部 API 或外部工具目录太大，无法优雅地全部塞进 prompt
- `上下文压力`：工具列表越长，token 成本越高，规划越慢
- `描述失真`：工具描述看起来相关，不代表工具在真实执行中稳定、快速、可靠

核心思路是：不要只按语义相似度选工具。ToolScout 把语义相关性和执行反馈一起纳入排序，让系统更偏向选择“既相关又更可能成功”的工具。

## 什么时候特别适合用 ToolScout

- **企业内部工具库**：当内部 API、微服务或平台工具达到几百个以上时。
- **高风险任务**：当选错工具的代价很高，例如误调用 `delete_user` 而不是 `get_user`。
- **成本敏感应用**：当你希望更小、更便宜的模型也能在受限上下文里稳定完成工具选择。

## 架构概览

```text
Query
  |
  v
Skill Router
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
Agent Execution
  |
  v
Execution Feedback
```

各阶段作用：

- `Skill Router`：先把查询映射到更高层的工作流，例如 `news_research`
- `Tool Retrieval`：从大工具库中做语义召回
- `Execution-aware Reranking`：把语义相关性、成功率和延迟结合起来做重排
- `Selected Tools`：只把少量候选工具交给下游 Agent
- `Agent Execution`：让 LLM 在压缩后的工具集合上做规划与调用
- `Execution Feedback`：记录执行结果，反哺下一轮排序

## 核心特性

- 面向大规模工具库的语义检索
- 基于 skill 的路由
- 工具依赖图和 graph-aware 扩展
- 执行感知重排
- MCP 兼容工具加载
- 可离线运行的评测框架
- 支持 FAISS，也支持轻量级 NumPy fallback

## 安装

先克隆仓库，再安装默认轻量依赖：

```bash
git clone https://github.com/Epsilon617/toolscout.git
cd toolscout
pip install -e .
```

可选 extras：

- `pip install -e .[embeddings]`：安装 FAISS 和 sentence-transformers
- `pip install -e .[openai]`：安装 OpenAI demo 和 judge 所需依赖
- `pip install -e .[full]`：安装完整依赖

默认安装刻意保持轻量，会使用本地 NumPy + keyword fallback 后端。即使没有这些可选依赖，ToolScout 也可以离线跑通示例和评测。

## 使用方式

### CLI

```bash
toolscout search "weather in Tokyo"
toolscout search "weather in Tokyo" --execution-aware
toolscout search "latest Nvidia news" --graph-aware
toolscout skill "latest Nvidia news"
toolscout simulate-runs --queries datasets/tool_queries.json --passes 5 --clear
toolscout feedback-stats
toolscout load-mcp path/to/mcp_tools.json
```

### Python

```python
from toolscout import ToolRegistry, ToolRetriever

registry = ToolRegistry()
registry.register_tool(
    name="weather_api",
    description="Get weather for a city",
    args=["city"],
)
registry.register_tool(
    name="news_search",
    description="Fetch recent news coverage for a topic or company",
    args=["topic"],
)

retriever = ToolRetriever(registry=registry)
retriever.fit()

results = retriever.search("latest Nvidia news", top_k=2)
for hit in results:
    print(hit.rank, hit.tool.name, round(hit.score, 4))
```

### 更多示例

- `python examples/simple_agent.py --query "latest news about Nvidia"`
- `python examples/retrieval_demo.py`
- `python examples/langgraph_agent.py --query "summarize recent Nvidia articles"`
- `python examples/openai_agent_demo.py --model gpt-4.1-mini --query "latest news about Nvidia"`

## 评测

ToolScout 的评测不只看检索，还看决策质量、执行成功率和鲁棒性。

### 评测层次

- `Retrieval Quality`：`Recall@k`、`Precision@1`
- `Decision Quality`：hard-negative 干扰下是否还能选对
- `Execution Success`：端到端 `Pass@1`、`Pass@k`
- `Robustness`：schema mutation、tool sprawl、跨语言查询

离线示例结果，来自 `python benchmark/generate_eval_report.py --top-k 5`：

| Method | Precision@1 | Pass@1（任务成功） | Avg Latency |
| ------ | ----------: | ----------------: | ----------: |
| Random Retrieval | 0.417 | 0.000 | 133.289 ms |
| Semantic Retrieval | 0.833 | 0.750 | 117.555 ms |
| **ToolScout** | **1.000** | **0.917** | **111.454 ms** |

在这组离线结果里，ToolScout 相比纯语义检索同时提升了成功率，并且延迟更低。

以下评测脚本都可以离线运行：

- `python benchmark/hard_negative_eval.py --top-k 4`
- `python benchmark/judge_eval.py --judge-mode mock --top-k 5`
- `python benchmark/e2e_eval.py --top-k 5`
- `python benchmark/robustness_eval.py --top-k 5`
- `python benchmark/generate_eval_report.py --top-k 5`

其中 `judge_eval.py` 默认使用 `mock` 模式；端到端执行默认使用 `ToolExecutionSimulator`。

## 评测数据集

评测数据包括：

- 12 条 benchmark 查询
- 1000 个合成工具
- 12 条 hard-negative 样本
- 1 条显式多步工具链：`news_search -> news_summary`
- 6 条中文跨语言鲁棒性查询

数据集刻意保持轻量，目的是让完整评测可以在普通本地 Python 环境下离线复现。

支持的工具类别：

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

### 评测流程

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

这条链路体现了 ToolScout 的核心区别：不仅看“检索像不像”，还看“执行成不成功”。

## 仓库结构

```text
toolscout/    核心库、CLI、注册表、检索、图结构、反馈模块
benchmark/    检索、决策、执行和鲁棒性评测
datasets/     合成工具、skills、查询集、hard negatives
examples/     simple agent、retrieval demo、LangGraph demo、OpenAI demo
assets/       README 图片资源
```

## 贡献方式

欢迎贡献。

- Fork 仓库
- 创建功能分支
- 运行与你改动相关的离线示例或 benchmark
- 提交带清晰说明的 Pull Request

## 路线图

- 更大的 benchmark 数据集
- 更强的执行反馈学习机制
- 真实 API 的执行基准
- 与 LangGraph、MCP 等 Agent 框架更深入集成

## License

MIT
