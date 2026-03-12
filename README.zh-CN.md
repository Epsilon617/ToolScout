# ToolScout

> ToolScout：面向 LLM Agent 的执行感知型工具路由器。

[English](README.md)

ToolScout 用来解决一个很实际的问题：当 Agent 拥有几百到几千个工具时，如果把所有工具都塞进提示词，通常会带来更高延迟、更差的工具选择，以及更多看起来合理但实际上不可靠的调用。ToolScout 先检索、再重排，只把最相关且更可靠的工具交给 Agent。

## ToolScout 解决什么问题

大规模工具库会让 Agent 出现几个典型问题：

- Prompt 过长，token 成本上升
- 模型响应变慢
- 工具选择精度下降
- 工具描述相似时容易选到“不稳定但看起来很像”的工具

ToolScout 的核心思路是先缩小候选集合，再结合历史执行反馈做排序。

## 快速体验

在仓库根目录执行：

```bash
pip install -e .
toolscout search "latest Nvidia news"
```

预期输出：

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

## 架构

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

各阶段作用：

- `Skill Routing`：先做高层技能路由，例如 `news_research`
- `Tool Retrieval`：从大工具库里做语义召回
- `Execution-aware Reranking`：结合历史成功率和延迟做重排
- `Selected Tools`：只把少量候选工具交给下游 Agent
- `LLM Agent`：在受限工具集合上进行规划和参数生成
- `Tool Execution`：执行真实工具或模拟器工具
- `Execution Feedback`：记录成功率与延迟，反哺下一轮排序

## 核心特性

- 面向大规模工具库的语义检索
- 基于 skill 的路由
- 工具依赖图和 graph-aware 扩展
- 执行感知重排
- MCP 兼容工具加载
- 可完全离线运行的评测框架
- 没有 FAISS 或 sentence-transformers 时可使用本地回退实现

## 为什么 ToolScout 不一样

传统工具检索大多只看语义相似度，但语义相似并不等于真实可用。一个工具描述写得很好，不代表它延迟低、成功率高、在真实任务里稳定。

ToolScout 把执行反馈也纳入排序依据。这样系统不只是“找到看起来相关的工具”，而是更倾向于选择“既相关又更可能成功”的工具。

## 安装

```bash
git clone https://github.com/Epsilon617/toolscout.git
cd toolscout
pip install -e .
```

可选依赖：

- `faiss-cpu`
- `sentence-transformers`
- `openai`，仅用于可选的 OpenAI demo 或 judge 模式

即使没有这些依赖，ToolScout 也可以用本地回退后端跑通示例和评测。

## 常用命令

```bash
toolscout search "latest Nvidia news"
toolscout search "weather in Tokyo" --execution-aware
toolscout search "latest Nvidia news" --graph-aware
toolscout skill "latest Nvidia news"
toolscout simulate-runs --queries datasets/tool_queries.json --passes 5 --clear
toolscout feedback-stats
toolscout load-mcp path/to/mcp_tools.json
```

## 评测

ToolScout 的评测分为四层：

- `Retrieval Quality`：`Recall@k`、`Precision@1`
- `Decision Quality`：hard-negative 干扰集下是否还能选对
- `Execution Success`：端到端 `Pass@1`、`Pass@k`
- `Robustness`：schema mutation、tool sprawl、跨语言查询

离线示例结果，来自 `python benchmark/generate_eval_report.py --top-k 5`：

| Method | Precision@1 | Pass@1 | Avg Latency |
| ------ | ----------: | -----: | ----------: |
| Random Retrieval | 0.417 | 0.000 | 133.289 ms |
| Semantic Retrieval | 0.833 | 0.750 | 117.555 ms |
| ToolScout | 1.000 | 0.917 | 111.454 ms |

以下评测脚本都可以离线运行：

- `python benchmark/hard_negative_eval.py --top-k 4`
- `python benchmark/judge_eval.py --judge-mode mock --top-k 5`
- `python benchmark/e2e_eval.py --top-k 5`
- `python benchmark/robustness_eval.py --top-k 5`
- `python benchmark/generate_eval_report.py --top-k 5`

其中 `judge_eval.py` 默认使用 `mock` 模式；端到端执行默认使用 `ToolExecutionSimulator`。

## 数据集概览

评测数据包括：

- 12 条主 benchmark 查询
- 1000 个合成工具
- 12 条 hard-negative 样本
- 1 条显式多步任务：`news_search -> news_summary`
- 6 条中文跨语言鲁棒性查询

数据集刻意保持轻量，目的是让完整评测可以在普通本地 Python 环境下离线复现。

工具类别包括：

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

## 评测流程

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

## 路线图

- 更大的 benchmark 数据集
- 更强的执行反馈学习机制
- 真实 API 的执行基准
- 与 LangGraph、MCP 等 Agent 框架更深入集成

## License

MIT
